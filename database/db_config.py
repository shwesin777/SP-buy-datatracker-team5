import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

RDS_HOST = os.getenv("RDS_HOST")
RDS_PORT = os.getenv("RDS_PORT", "3306")
RDS_USER = os.getenv("RDS_USER")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_DB = os.getenv("RDS_DB", "spbuy_db")


def get_server_engine():
    """Connects to the MySQL server directly (used to create the database if missing)."""
    url = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}"
    return create_engine(url)


def get_db_engine():
    """Connects directly to the target spbuy_db database."""
    url = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB}"
    return create_engine(url)


def ensure_database_exists():
    """Creates spbuy_db if it wasn't created during AWS RDS setup."""
    engine = get_server_engine()
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{RDS_DB}`;"))
        conn.commit()
    print(f"Database '{RDS_DB}' verified/created.")