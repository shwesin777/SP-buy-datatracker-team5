from pathlib import Path
import numpy as np
import pandas as pd


def clean_table(df):
    df = df.copy()

    for column in df.select_dtypes(include=["object", "string"]):
        df[column] = df[column].str.strip().replace("", pd.NA)

    df["country_code"] = df["country_code"].str.upper()
    return df.drop_duplicates()


def parse_order_dates(series):
    dates = series.str.replace("/", "-", regex=False)
    numeric = pd.to_datetime(
        dates, format="%Y-%m-%d", errors="coerce", utc=True
    )
    named = pd.to_datetime(
        dates, format="%Y-%b-%d", errors="coerce", utc=True
    )
    return numeric.fillna(named)


def prepare_batch(folder, batch="base"):
    folder = Path(folder)
    tables = {}

    for name in ["customers", "orders", "labels"]:
        df = pd.read_csv(
            folder / f"{name}_{batch}.csv",
            dtype="string"
        )
        tables[name] = clean_table(df)

    customers, orders, labels = (
        tables["customers"], tables["orders"], tables["labels"]
    )

    orders["payment_method"] = (
        orders["payment_method"].str.lower().fillna("unknown")
    )
    orders["order_date"] = parse_order_dates(orders["order_date"])
    customers["first_order_datetime"] = pd.to_datetime(
        customers["first_order_datetime"],
        format="mixed", errors="coerce", utc=True
    )
    mobile = customers["mobile_verified"].str.lower()

    customers["mobile_verified"] = mobile.map({
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False
    }).astype("boolean")

    unexpected = mobile.notna() & customers["mobile_verified"].isna()

    if unexpected.any():
        raise ValueError(
            "Unexpected mobile_verified values: "
            f"{mobile.loc[unexpected].unique().tolist()}"
        )

    labels["is_fraud"] = pd.to_numeric(
        labels["is_fraud"], errors="coerce"
    )

    if not labels["is_fraud"].isin([0, 1]).all():
        raise ValueError("Fraud labels must contain only 0 or 1.")

    labels["is_fraud"] = labels["is_fraud"].astype("int64")

    data = orders.merge(
        labels, on=["country_code", "order_id"],
        how="inner", validate="one_to_one"
    ).merge(
        customers, on=["country_code", "customer_id"],
        how="left", validate="many_to_one", indicator=True
    )

    data["customer_record_missing"] = data.pop("_merge").eq("left_only")

    for column in [
        "num_orders_last_50days", "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days", "total_payment_last_50days",
        "num_associated_customers", "order_value",
        "num_items_ordered", "refund_value"
    ]:
        values = pd.to_numeric(data[column], errors="coerce")
        data[column] = values.mask((values < 0) | ~np.isfinite(values))
        data[f"{column}_missing"] = data[column].isna()
        data[f"{column}_log"] = np.log1p(data[column].astype(float))

    age = (
        data["order_date"].dt.normalize()
        - data["first_order_datetime"].dt.normalize()
    ).dt.days

    data["customer_age_invalid"] = age.lt(0)
    data["customer_age_days"] = age.mask(age.lt(0))
    data["first_order_datetime_missing"] = data["first_order_datetime"].isna()
    data["order_day_of_week"] = data["order_date"].dt.dayofweek
    data["order_month"] = data["order_date"].dt.month

    return data

