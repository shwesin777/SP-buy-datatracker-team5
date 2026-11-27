import os
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from db_config import get_db_engine  # type: ignore

load_dotenv()

# Support local Prefect server by default, or cloud Prefect server via PREFECT_API_URL in .env (PBI 14)
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")


@task(name="Jaime-Query-1-Basket-Size-Fraud", retries=2, retry_delay_seconds=5)
def task_basket_size_fraud() -> pd.DataFrame:
    logger = get_run_logger()
    query = """
    SELECT
        LEAST(num_items_ordered, 5) AS basket_size,
        COUNT(*) AS labelled_orders,
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_orders,
        ROUND(100.0 * AVG(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END), 2) AS fraud_rate_pct
    FROM cleaned_fraud_data
    WHERE is_fraud IN (0, 1)
      AND num_items_ordered >= 1
    GROUP BY LEAST(num_items_ordered, 5)
    ORDER BY basket_size
    """
    engine = get_db_engine()
    try:
        results = pd.read_sql(query, con=engine)
    finally:
        engine.dispose()

    logger.info("\nQuery 1 — Fraud by basket size:\n" + results.to_string(index=False))
    return results


@task(name="Jaime-Query-2-Order-Value-Fraud", retries=2, retry_delay_seconds=5)
def task_order_value_fraud() -> pd.DataFrame:
    logger = get_run_logger()
    value_query = """
    SELECT
        value_band,
        COUNT(*) AS labelled_orders,
        SUM(is_fraud) AS fraud_orders,
        ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN order_value ELSE 0 END), 2) AS fraud_order_value
    FROM (
        SELECT
            order_value,
            is_fraud,
            CASE
                WHEN order_value < 25 THEN '1. Under 25'
                WHEN order_value < 50 THEN '2. 25 to under 50'
                WHEN order_value < 100 THEN '3. 50 to under 100'
                ELSE '4. 100 and above'
            END AS value_band
        FROM cleaned_fraud_data
        WHERE is_fraud IN (0, 1)
          AND order_value >= 0
    ) AS banded_orders
    GROUP BY value_band
    ORDER BY value_band
    """
    engine = get_db_engine()
    try:
        value_results = pd.read_sql(value_query, con=engine)
    finally:
        engine.dispose()

    logger.info("\nQuery 2 — Fraud by order value:\n" + value_results.to_string(index=False))
    return value_results


@task(name="Jaime-Query-3-Refunds-By-Fraud", retries=2, retry_delay_seconds=5)
def task_refunds_by_fraud() -> pd.DataFrame:
    logger = get_run_logger()
    refund_query = """
    SELECT
        CASE WHEN is_fraud = 1 THEN 'Fraud' ELSE 'Non-fraud' END AS order_group,
        COUNT(*) AS total_orders,
        COUNT(usable_refund) AS orders_with_refund_data,
        COUNT(*) - COUNT(usable_refund) AS missing_or_invalid,
        SUM(CASE WHEN usable_refund = 0 THEN 1 ELSE 0 END) AS zero_refunds,
        SUM(CASE WHEN usable_refund > 0 THEN 1 ELSE 0 END) AS refunded_orders,
        ROUND(
            100.0 * SUM(CASE WHEN usable_refund > 0 THEN 1 ELSE 0 END)
            / NULLIF(COUNT(usable_refund), 0),
            2
        ) AS refund_rate_pct,
        ROUND(SUM(usable_refund), 2) AS total_refund_value,
        ROUND(AVG(CASE WHEN usable_refund > 0 THEN usable_refund END), 2)
            AS avg_refund_per_refunded_order
    FROM (
        SELECT
            is_fraud,
            CASE
                WHEN refund_value_missing = 0 AND refund_value >= 0
                THEN refund_value
            END AS usable_refund
        FROM cleaned_fraud_data
        WHERE is_fraud IN (0, 1)
    ) AS refund_data
    GROUP BY is_fraud
    ORDER BY is_fraud DESC
    """
    engine = get_db_engine()
    try:
        refund_results = pd.read_sql(refund_query, con=engine)
    finally:
        engine.dispose()

    logger.info("\nQuery 3 — Refunds by fraud label:\n" + refund_results.to_string(index=False))
    return refund_results


@task(name="Jaime-Generate-Dashboard")
def task_generate_dashboard(
    results: pd.DataFrame, value_results: pd.DataFrame, refund_results: pd.DataFrame
):
    logger = get_run_logger()
    output_dir = Path(__file__).resolve().parent.parent / "fraud_analysis_outputs"
    output_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(
        "SP-Buy Trust & Safety: Fraud Trend & Risk Segment Analysis",
        fontsize=19,
        fontweight="bold",
    )

    palette = ["#38BDF8", "#2DD4BF", "#A78BFA", "#FBBF24", "#FB7185"]

    def draw_bars(ax, labels, values, title, ylabel, colours, fmt="%.2f%%"):
        bars = ax.bar(labels, values, color=colours, width=0.65)
        ax.bar_label(bars, fmt=fmt, padding=4)
        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(0, max(1, values.max() * 1.2))
        ax.xaxis.grid(False)

    basket_labels = results["basket_size"].map(
        lambda size: "5+" if size == 5 else str(int(size))
    )
    value_labels = value_results["value_band"].str.split(". ", n=1, regex=False).str[-1]

    draw_bars(
        axes[0, 0], basket_labels, results["fraud_rate_pct"],
        "Fraud Rate by Basket Size", "Fraud rate (%)", palette,
    )
    axes[0, 0].set_xlabel("Items per order")

    draw_bars(
        axes[0, 1], value_labels, value_results["fraud_rate_pct"],
        "Fraud Rate by Order Value", "Fraud rate (%)", palette,
    )

    draw_bars(
        axes[1, 0], value_labels, value_results["fraud_order_value"],
        "Total Value of Fraud-Labelled Orders", "Total order value", palette,
        fmt="{:,.0f}",
    )

    draw_bars(
        axes[1, 1], refund_results["order_group"], refund_results["refund_rate_pct"],
        "Recorded Refund Rate by Fraud Label", "Orders with a positive refund (%)",
        ["#FB7185", "#2DD4BF"],
    )
    axes[1, 1].set_ylim(0, 110)
    axes[1, 1].set_yticks(range(0, 101, 20))

    dashboard_path = output_dir / "SP_Buy_Fraud_Dashboard.png"
    fig.savefig(dashboard_path, dpi=200)
    plt.close(fig)
    logger.info(f"Dashboard saved to: {dashboard_path}")


@flow(name="spbuy-fraud-analytics-jaime", log_prints=True)
def run_jaime_fraud_analytics_flow():
    logger = get_run_logger()
    start_time = time.time()
    logger.info("Starting SP-buy Automated Fraud Analytics Prefect Flow (Jaime)...")

    results = task_basket_size_fraud()
    value_results = task_order_value_fraud()
    refund_results = task_refunds_by_fraud()
    task_generate_dashboard(results, value_results, refund_results)

    elapsed = time.time() - start_time
    logger.info(f"Completed Jaime's fraud analytics flow in {elapsed:.2f}s (Target: < 180s).")


if __name__ == "__main__":
    run_jaime_fraud_analytics_flow()