import os
import time
import pandas as pd
from sqlalchemy import text
from db_config import ensure_database_exists, get_db_engine

# Path from database/ to cleaned_data.csv in CA1/
CSV_PATH = os.path.join("..", "..", "cleaned_data.csv")
TABLE_NAME = "cleaned_fraud_data"
CHUNK_SIZE = 25000


def create_cleaned_table(engine):
    """Creates the cleaned_fraud_data table matching all 39 engineered columns."""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        country_code VARCHAR(10) NOT NULL,
        order_id VARCHAR(64) NOT NULL,
        collect_type VARCHAR(50),
        payment_method VARCHAR(50),
        order_value DOUBLE,
        num_items_ordered INT,
        refund_value DOUBLE,
        order_date DATETIME NULL,
        customer_id VARCHAR(64),
        is_fraud TINYINT,
        mobile_verified TINYINT,
        num_orders_last_50days INT,
        num_cancelled_orders_last_50days INT,
        num_refund_orders_last_50days INT,
        total_payment_last_50days DOUBLE,
        num_associated_customers INT,
        first_order_datetime DATETIME NULL,
        customer_record_missing TINYINT,
        num_orders_last_50days_missing TINYINT,
        num_orders_last_50days_log DOUBLE,
        num_cancelled_orders_last_50days_missing TINYINT,
        num_cancelled_orders_last_50days_log DOUBLE,
        num_refund_orders_last_50days_missing TINYINT,
        num_refund_orders_last_50days_log DOUBLE,
        total_payment_last_50days_missing TINYINT,
        total_payment_last_50days_log DOUBLE,
        num_associated_customers_missing TINYINT,
        num_associated_customers_log DOUBLE,
        order_value_missing TINYINT,
        order_value_log DOUBLE,
        num_items_ordered_missing TINYINT,
        num_items_ordered_log DOUBLE,
        refund_value_missing TINYINT,
        refund_value_log DOUBLE,
        customer_age_invalid TINYINT,
        customer_age_days DOUBLE,
        first_order_datetime_missing TINYINT,
        order_day_of_week INT,
        order_month INT,
        PRIMARY KEY (country_code, order_id)
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    print(f"Table '{TABLE_NAME}' verified/created.")


def load_csv_in_chunks(engine):
    """Reads the 328MB CSV in chunks and bulk-inserts into AWS RDS MySQL."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find CSV at: {os.path.abspath(CSV_PATH)}")

    # Clear existing rows so re-runs don't hit duplicate primary key errors
    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLE_NAME};"))
        conn.commit()

    bool_cols = [
        "mobile_verified",
        "customer_record_missing",
        "num_orders_last_50days_missing",
        "num_cancelled_orders_last_50days_missing",
        "num_refund_orders_last_50days_missing",
        "total_payment_last_50days_missing",
        "num_associated_customers_missing",
        "order_value_missing",
        "num_items_ordered_missing",
        "refund_value_missing",
        "customer_age_invalid",
        "first_order_datetime_missing",
    ]

    total_rows = 0
    start_time = time.time()

    print(f"Starting chunked upload from {CSV_PATH}...")
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        # Parse datetimes cleanly
        chunk["order_date"] = pd.to_datetime(chunk["order_date"], errors="coerce")
        chunk["first_order_datetime"] = pd.to_datetime(chunk["first_order_datetime"], errors="coerce")

        # Convert TRUE/FALSE boolean flags to 1/0 integers for MySQL TINYINT
        for col in bool_cols:
            if col in chunk.columns:
                chunk[col] = (
                    chunk[col]
                    .replace({True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0})
                    .fillna(0)
                    .astype(int)
                )

        # method="multi" batches INSERT rows for faster network throughput to AWS RDS
        chunk.to_sql(
            TABLE_NAME,
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2000,
        )

        total_rows += len(chunk)
        elapsed = time.time() - start_time
        print(f"Batch {i}: Uploaded {total_rows:,} rows ({elapsed:.1f}s elapsed)")

    with engine.connect() as conn:
        db_count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME};")).scalar()

    print(f"\nDone! Total rows in RDS '{TABLE_NAME}': {db_count:,}")


if __name__ == "__main__":
    ensure_database_exists()
    db_engine = get_db_engine()
    create_cleaned_table(db_engine)
    load_csv_in_chunks(db_engine)