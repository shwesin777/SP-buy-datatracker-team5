import os
import time
from urllib.parse import quote_plus
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from sqlalchemy import create_engine, text

load_dotenv()

# Support local Prefect server by default, or cloud Prefect server via PREFECT_API_URL in .env (PBI 14)
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

OUTPUT_DIR = "analysis/fraud_analysis_shwesin"
TEAM_OUTPUT_DIR = "fraud_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEAM_OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def get_rds_engine():
    rds_user = os.getenv("RDS_USER")
    rds_password = quote_plus(os.getenv("RDS_PASSWORD", ""))
    rds_host = os.getenv("RDS_HOST")
    rds_port = os.getenv("RDS_PORT", "3306")
    rds_db = os.getenv("RDS_DB", "spbuy_db")
    db_url = f"mysql+pymysql://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_db}"
    return create_engine(db_url)


@task(name="Query 1: Payment & Collection Fraud Exposure", retries=2, retry_delay_seconds=5)
def task_payment_collect_fraud() -> pd.DataFrame:
    logger = get_run_logger()
    q1_sql = """
    SELECT
        payment_method,
        collect_type,
        COUNT(*) AS total_orders,
        SUM(is_fraud) AS fraud_orders,
        ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN order_value ELSE 0 END), 2) AS fraud_loss_sgd
    FROM cleaned_fraud_data
    GROUP BY payment_method, collect_type
    ORDER BY fraud_rate_pct DESC;
    """
    engine = get_rds_engine()
    with engine.connect() as conn:
        df_q1 = pd.read_sql(text(q1_sql), conn)

    logger.info("\n=== Q1: Fraud Exposure by Payment Method & Collect Type ===\n" + df_q1.to_string(index=False))

    top_methods = df_q1.groupby("payment_method")["fraud_rate_pct"].max().nlargest(10).index
    df_q1_top = df_q1[df_q1["payment_method"].isin(top_methods)]

    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(data=df_q1_top, y="payment_method", x="fraud_rate_pct", hue="collect_type", order=top_methods)
    ax1.set_title("Top 10 Payment Methods by Fraud Rate (%) & Collection Type", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Fraud Rate (%)")
    ax1.set_ylabel("Payment Method")
    plt.legend(title="Collect Type")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q1_payment_collect_fraud.png", dpi=300)
    plt.close()
    return df_q1


@task(name="Query 2: High-Risk Customer Account Clusters", retries=2, retry_delay_seconds=5)
def task_customer_risk_clusters() -> pd.DataFrame:
    logger = get_run_logger()
    q2_sql = """
    SELECT
        mobile_verified,
        CASE
            WHEN num_associated_customers = 0 THEN '0 Linked Accounts'
            WHEN num_associated_customers <= 2 THEN '1-2 Linked Accounts'
            ELSE '3+ Linked Accounts (Ring Risk)'
        END AS account_cluster,
        COUNT(*) AS total_orders,
        SUM(is_fraud) AS fraud_orders,
        ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct,
        ROUND(AVG(order_value), 2) AS avg_order_value_sgd
    FROM cleaned_fraud_data
    GROUP BY mobile_verified, account_cluster
    ORDER BY fraud_rate_pct DESC;
    """
    engine = get_rds_engine()
    with engine.connect() as conn:
        df_q2 = pd.read_sql(text(q2_sql), conn)

    logger.info(
        "\n=== Q2: High-Risk Customer Clusters (Verification & Linked Accounts) ===\n"
        + df_q2.to_string(index=False)
    )

    plt.figure(figsize=(10, 5))
    ax2 = sns.barplot(data=df_q2, x="account_cluster", y="fraud_rate_pct", hue="mobile_verified")
    ax2.set_title("Fraud Rate by Linked Account Cluster & Mobile Verification", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_xlabel("Customer Account Cluster")
    plt.legend(title="Mobile Verified")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q2_customer_risk_clusters.png", dpi=300)
    plt.close()
    return df_q2


@task(name="Query 3: Refund Abuse Behavior", retries=2, retry_delay_seconds=5)
def task_refund_abuse_patterns() -> pd.DataFrame:
    logger = get_run_logger()
    q3_sql = """
    SELECT
        is_fraud,
        COUNT(*) AS total_orders,
        ROUND(AVG(order_value), 2) AS avg_order_value_sgd,
        ROUND(AVG(refund_value), 2) AS avg_refund_value_sgd,
        ROUND(AVG(num_refund_orders_last_50days), 2) AS avg_50d_refund_count,
        ROUND(AVG(num_cancelled_orders_last_50days), 2) AS avg_50d_cancel_count
    FROM cleaned_fraud_data
    GROUP BY is_fraud;
    """
    engine = get_rds_engine()
    with engine.connect() as conn:
        df_q3 = pd.read_sql(text(q3_sql), conn)

    df_q3["Status"] = df_q3["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
    display_cols = [
        "Status",
        "total_orders",
        "avg_order_value_sgd",
        "avg_refund_value_sgd",
        "avg_50d_refund_count",
        "avg_50d_cancel_count",
    ]
    logger.info(
        "\n=== Q3: Refund & Cancellation Behavior (Legitimate vs. Fraud) ===\n"
        + df_q3[display_cols].to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.barplot(
        data=df_q3, x="Status", y="avg_refund_value_sgd", hue="Status", legend=False, ax=axes[0], palette=["#4c72b0", "#dd8452"]
    )
    axes[0].set_title("Average Refund Value (SGD)", fontweight="bold")
    axes[0].set_ylabel("SGD")

    sns.barplot(
        data=df_q3, x="Status", y="avg_50d_refund_count", hue="Status", legend=False, ax=axes[1], palette=["#4c72b0", "#dd8452"]
    )
    axes[1].set_title("Avg Refunded Orders in Last 50 Days", fontweight="bold")
    axes[1].set_ylabel("Order Count")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q3_refund_abuse_patterns.png", dpi=300)
    plt.close()
    return df_q3


@task(name="Generate Unified Dashboard PNG", retries=1)
def task_generate_dashboard_png() -> str:
    logger = get_run_logger()
    q1_img = mpimg.imread(f"{OUTPUT_DIR}/q1_payment_collect_fraud.png")
    q2_img = mpimg.imread(f"{OUTPUT_DIR}/q2_customer_risk_clusters.png")
    q3_img = mpimg.imread(f"{OUTPUT_DIR}/q3_refund_abuse_patterns.png")

    fig = plt.figure(figsize=(16, 11), facecolor="#f8fafc")
    fig.suptitle(
        "SP-buy Trust & Safety — Fraud Exposure, Account Clusters & Refund Abuse Dashboard (Shwe Sin)",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08, top=0.91, bottom=0.04, left=0.03, right=0.97)
    for ax, img in zip([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])], [q1_img, q2_img, q3_img]):
        ax.imshow(img)
        ax.axis("off")

    for out_path in [f"{OUTPUT_DIR}/shwesin_dashboard.png", f"{TEAM_OUTPUT_DIR}/shwesin_dashboard.png"]:
        plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    logger.info("Updated unified shwesin_dashboard.png in analysis/fraud_analysis_shwesin/ and fraud_analysis_outputs/.")
    return f"{TEAM_OUTPUT_DIR}/shwesin_dashboard.png"


@flow(name="spbuy-fraud-analytics-shwesin", log_prints=True)
def run_fraud_analytics_flow():
    logger = get_run_logger()
    start_time = time.time()
    logger.info("Starting SP-buy Automated Fraud Analytics Prefect Flow (Shwe Sin)...")

    task_payment_collect_fraud()
    task_customer_risk_clusters()
    task_refund_abuse_patterns()
    task_generate_dashboard_png()

    elapsed = time.time() - start_time
    logger.info(f"Completed all 4 fraud analytics tasks and updated dashboard in {elapsed:.2f}s (Target: < 180s).")


if __name__ == "__main__":
    run_fraud_analytics_flow()