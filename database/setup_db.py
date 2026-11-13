import pandas as pd
from sqlalchemy import text
from db_config import ensure_database_exists, get_db_engine


def create_tables(engine):
    """Creates the customers, orders, and labels tables in MySQL."""
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS customers (
            country_code VARCHAR(10) NOT NULL,
            customer_id VARCHAR(64) NOT NULL,
            mobile_verified VARCHAR(20),
            num_orders_last_50days INT,
            num_cancelled_orders_last_50days INT,
            num_refund_orders_last_50days INT,
            total_payment_last_50days DOUBLE,
            num_associated_customers INT,
            first_order_datetime DATETIME NULL,
            PRIMARY KEY (country_code, customer_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS orders (
            country_code VARCHAR(10) NOT NULL,
            order_id VARCHAR(64) NOT NULL,
            collect_type VARCHAR(50),
            payment_method VARCHAR(50),
            order_value DOUBLE,
            num_items_ordered INT,
            refund_value DOUBLE,
            order_date DATETIME NULL,
            PRIMARY KEY (country_code, order_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS labels (
            country_code VARCHAR(10) NOT NULL,
            order_id VARCHAR(64) NOT NULL,
            customer_id VARCHAR(64) NOT NULL,
            is_fraud TINYINT,
            PRIMARY KEY (country_code, order_id)
        );
        """
    ]

    with engine.connect() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))
        conn.commit()
    print("Tables 'customers', 'orders', and 'labels' created successfully!")


def test_connection_and_load(engine):
    """Tests inserting a sample row from each CSV and reading it back."""
    print("Testing sample insert & read...")

    # Read 5 sample rows from the base CSVs one folder up
    df_cust = pd.read_csv("../customers_base.csv", nrows=5, low_memory=False)
    df_ord = pd.read_csv("../orders_base.csv", nrows=5, low_memory=False)
    df_lbl = pd.read_csv("../labels_base.csv", nrows=5, low_memory=False)

    # Convert datetime strings cleanly
    df_cust["first_order_datetime"] = pd.to_datetime(df_cust["first_order_datetime"], errors="coerce")
    df_ord["order_date"] = pd.to_datetime(df_ord["order_date"], errors="coerce")

    # Clear test rows if re-running, then insert sample rows
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM labels;"))
        conn.execute(text("DELETE FROM orders;"))
        conn.execute(text("DELETE FROM customers;"))
        conn.commit()

    df_cust.to_sql("customers", con=engine, if_exists="append", index=False)
    df_ord.to_sql("orders", con=engine, if_exists="append", index=False)
    df_lbl.to_sql("labels", con=engine, if_exists="append", index=False)

    # Verify read-back
    with engine.connect() as conn:
        cust_count = conn.execute(text("SELECT COUNT(*) FROM customers;")).scalar()
        ord_count = conn.execute(text("SELECT COUNT(*) FROM orders;")).scalar()
        lbl_count = conn.execute(text("SELECT COUNT(*) FROM labels;")).scalar()

    print(f"Verification passed! Row counts -> customers: {cust_count}, orders: {ord_count}, labels: {lbl_count}")


if __name__ == "__main__":
    ensure_database_exists()
    db_engine = get_db_engine()
    create_tables(db_engine)
    test_connection_and_load(db_engine)