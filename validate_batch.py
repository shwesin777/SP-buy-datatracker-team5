import csv
from pathlib import Path
import pandas as pd
from clean_data import parse_order_dates



EXPECTED_COLUMNS = {
    "customers": (
        "country_code",
        "customer_id",
        "mobile_verified",
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "total_payment_last_50days",
        "num_associated_customers",
        "first_order_datetime",
    ),
    "orders": (
        "country_code",
        "order_id",
        "collect_type",
        "payment_method",
        "order_value",
        "num_items_ordered",
        "refund_value",
        "order_date",
    ),
    "labels": (
        "country_code",
        "order_id",
        "customer_id",
        "is_fraud",
    ),
}


def validate_batch_headers(folder, batch):
    folder = Path(folder)

    for table, expected in EXPECTED_COLUMNS.items():
        path = folder / f"{table}_{batch}.csv"

        with path.open(encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), [])

        missing = sorted(set(expected) - set(header))
        unexpected = sorted(set(header) - set(expected))

        if missing or unexpected or len(header) != len(set(header)):
            raise ValueError(
                f"{path.name} has an invalid header. "
                f"Missing columns: {missing}. "
                f"Unexpected columns: {unexpected}. "
                f"Duplicate column names: {len(header) != len(set(header))}."
            )

REQUIRED_IDS = {
    "customers": ("country_code", "customer_id"),
    "orders": ("country_code", "order_id"),
    "labels": ("country_code", "order_id", "customer_id"),
}
COUNT_COLUMNS = {
    "customers": (
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "num_associated_customers",
    ),
    "orders": ("num_items_ordered",),
    "labels": (),
}

AMOUNT_COLUMNS = {
    "customers": ("total_payment_last_50days",),
    "orders": ("order_value", "refund_value"),
    "labels": (),
}

def validate_batch(folder, batch):
    folder = Path(folder)
    validate_batch_headers(folder, batch)

    for table, id_columns in REQUIRED_IDS.items():
        path = folder / f"{table}_{batch}.csv"
        count_columns = COUNT_COLUMNS[table]
        numeric_columns = count_columns + AMOUNT_COLUMNS[table]
        columns_to_check = [*id_columns, *numeric_columns]

        if table == "customers":
            columns_to_check.append("first_order_datetime")
        elif table == "orders":
            columns_to_check.append("order_date")
        else:
            columns_to_check.append("is_fraud")

        data = pd.read_csv(
            path,
            usecols=columns_to_check,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        if table == "customers":
            column = "first_order_datetime"
            raw = data[column].str.strip()
            parsed = pd.to_datetime(
                raw, format="mixed", errors="coerce", utc=True
            )
        elif table == "orders":
            column = "order_date"
            raw = data[column].str.strip()
            parsed = parse_order_dates(raw)

        if table in ("customers", "orders"):
            invalid_count = int((raw.ne("") & parsed.isna()).sum())
            if invalid_count:
                raise ValueError(
                    f"{path.name}: {invalid_count} invalid date value(s) in {column}"
                )
        for column in numeric_columns:
            raw = data[column].str.strip()
            values = pd.to_numeric(raw, errors="coerce")

            invalid = raw.ne("") & (
                values.isna() | values.isin([float("inf"), float("-inf")])
            )

            if column in count_columns:
                invalid |= values.notna() & values.mod(1).ne(0)

            invalid_count = int(invalid.sum())
            if invalid_count:
                raise ValueError(
                    f"{path.name}: {invalid_count} invalid numeric value(s) in {column}"
                )
            
        for column in id_columns:
            blank_count = int(data[column].str.strip().eq("").sum())
            if blank_count:
                raise ValueError(
                    f"{path.name}: {blank_count} blank value(s) in {column}"
                )

        if table == "labels":
            labels = pd.to_numeric(data["is_fraud"].str.strip(), errors="coerce")
            invalid_count = int((~labels.isin([0, 1])).sum())
            if invalid_count:
                raise ValueError(
                    f"{path.name}: {invalid_count} value(s) in is_fraud "
                    "are missing or are not 0 or 1"
                )

def validate_order_label_matches(folder, batch):
    folder = Path(folder)
    keys = ["country_code", "order_id"]
    batch_keys = {}

    for table in ("orders", "labels"):
        data = pd.read_csv(
            folder / f"{table}_{batch}.csv",
            usecols=keys,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )

        data["country_code"] = data["country_code"].str.strip().str.upper()
        data["order_id"] = data["order_id"].str.strip()
        batch_keys[table] = data.drop_duplicates()

    matches = batch_keys["orders"].merge(
        batch_keys["labels"], on=keys, how="outer", indicator=True
    )

    orders_without_labels = int(matches["_merge"].eq("left_only").sum())
    labels_without_orders = int(matches["_merge"].eq("right_only").sum())

    if orders_without_labels or labels_without_orders:
        raise ValueError(
            f"Batch {batch!r}: {orders_without_labels} order(s) without a label "
            f"and {labels_without_orders} label key(s) without an order."
        )