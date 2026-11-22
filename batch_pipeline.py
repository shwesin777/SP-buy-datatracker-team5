import argparse
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from clean_data import prepare_batch
from validate_batch import validate_batch
import subprocess
import sys
from time import perf_counter


def prepare_batch_from_urls(customers_url, orders_url, labels_url):
    urls = {
        "customers": customers_url,
        "orders": orders_url,
        "labels": labels_url,
    }

    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir)

        for table, url in urls.items():
            path = folder / f"{table}_incoming.csv"
            with urlopen(url, timeout=30) as response, path.open("wb") as file:
                copyfileobj(response, file)

        validate_batch(folder, "incoming")
        print("Downloaded batch passed validation.")

        return prepare_batch(folder, "incoming")

def count_existing_orders(cleaned):
    from sqlalchemy import column, func, select, table, tuple_

    from database.db_config import get_db_engine

    keys = list(
        cleaned[["country_code", "order_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    fraud_data = table(
        "cleaned_fraud_data",
        column("country_code"),
        column("order_id"),
    )
    query = select(func.count()).select_from(fraud_data).where(
        tuple_(fraud_data.c.country_code, fraud_data.c.order_id).in_(keys)
    )

    engine = get_db_engine()
    try:
        with engine.connect() as connection:
            return connection.execute(query).scalar_one()
    finally:
        engine.dispose()

def load_batch_to_rds(cleaned):
    from database.db_config import get_db_engine

    rows = cleaned.copy()

    # MySQL DATETIME stores values without a timezone. The cleaner uses UTC.
    for column in ("order_date", "first_order_datetime"):
        rows[column] = rows[column].dt.tz_convert(None)

    # Store boolean flags as 0 or 1 and preserve missing values as NULL.
    for column in rows.select_dtypes(include=["bool", "boolean"]):
        rows[column] = rows[column].astype("Int8")

    engine = get_db_engine()
    try:
        with engine.begin() as connection:
            rows.to_sql(
                "cleaned_fraud_data",
                con=connection,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=2000,
            )
    finally:
        engine.dispose()

    return len(rows)
def retrain_models():
    repo_dir = Path(__file__).resolve().parent

    for script in ("train_ernest.py", "train_jaime.py", "train_shwesin.py"):
        started = perf_counter()
        print(f"Running {script}...", flush=True)
        subprocess.run(
            [sys.executable, str(repo_dir / "models" / script)],
            cwd=repo_dir,
            check=True,
        )
        print(f"{script} finished in {perf_counter() - started:.1f}s.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers-url", required=True)
    parser.add_argument("--orders-url", required=True)
    parser.add_argument("--labels-url", required=True)
    parser.add_argument("--load-rds", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    pipeline_started = perf_counter()
    cleaned = prepare_batch_from_urls(
        args.customers_url, args.orders_url, args.labels_url
    )
    print(f"Cleaned {len(cleaned):,} rows.")

    if args.load_rds:
        inserted = load_batch_to_rds(cleaned)
        print(f"Inserted {inserted:,} rows into RDS.")

    existing_orders = count_existing_orders(cleaned)
    print(f"Orders already in RDS: {existing_orders:,}")

    if args.retrain:
        if cleaned.empty or existing_orders != len(cleaned):
            raise SystemExit("Retraining requires a nonempty batch already loaded in RDS.")

        retrain_models()
        print(f"Pipeline finished in {perf_counter() - pipeline_started:.1f}s.")