import os
from urllib.parse import quote_plus
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Connect to AWS RDS MySQL using your team's .env variables
RDS_USER = os.getenv("RDS_USER")
RDS_PASSWORD = quote_plus(os.getenv("RDS_PASSWORD", ""))
RDS_HOST = os.getenv("RDS_HOST")
RDS_PORT = os.getenv("RDS_PORT", "3306")
RDS_DB = os.getenv("RDS_DB", "spbuy_db")

DATABASE_URL = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB}"
engine = create_engine(DATABASE_URL)

OUTPUT_DIR = "analysis/fraud_analysis_shwesin"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def run_query(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ---------------------------------------------------------
# QUERY 1: Fraud Rate & SGD Loss by Payment Method & Collect Type
# ---------------------------------------------------------
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
df_q1 = run_query(q1_sql)
print("\n=== Q1: Fraud Exposure by Payment Method & Collect Type ===")
print(df_q1.to_string(index=False))

plt.figure(figsize=(10, 5))
ax1 = sns.barplot(data=df_q1, x="payment_method", y="fraud_rate_pct", hue="collect_type")
ax1.set_title("Fraud Rate (%) by Payment Method & Collection Type", fontsize=13, fontweight="bold")
ax1.set_ylabel("Fraud Rate (%)")
ax1.set_xlabel("Payment Method")
plt.legend(title="Collect Type")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/q1_payment_collect_fraud.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# QUERY 2: High-Risk Customer Clusters (Mobile Verification & Linked Accounts)
# ---------------------------------------------------------
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
df_q2 = run_query(q2_sql)
print("\n=== Q2: High-Risk Customer Clusters (Verification & Linked Accounts) ===")
print(df_q2.to_string(index=False))

plt.figure(figsize=(10, 5))
ax2 = sns.barplot(data=df_q2, x="account_cluster", y="fraud_rate_pct", hue="mobile_verified")
ax2.set_title("Fraud Rate by Linked Account Cluster & Mobile Verification", fontsize=13, fontweight="bold")
ax2.set_ylabel("Fraud Rate (%)")
ax2.set_xlabel("Customer Account Cluster")
plt.legend(title="Mobile Verified")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/q2_customer_risk_clusters.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# QUERY 3: Refund Abuse Behavior (Legitimate vs. Fraudulent Orders)
# ---------------------------------------------------------
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
df_q3 = run_query(q3_sql)
df_q3["Status"] = df_q3["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
print("\n=== Q3: Refund & Cancellation Behavior (Legitimate vs. Fraud) ===")
print(df_q3[["Status", "total_orders", "avg_order_value_sgd", "avg_refund_value_sgd", "avg_50d_refund_count", "avg_50d_cancel_count"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.barplot(data=df_q3, x="Status", y="avg_refund_value_sgd", ax=axes[0], palette=["#4c72b0", "#dd8452"])
axes[0].set_title("Average Refund Value (SGD)", fontweight="bold")
axes[0].set_ylabel("SGD")

sns.barplot(data=df_q3, x="Status", y="avg_50d_refund_count", ax=axes[1], palette=["#4c72b0", "#dd8452"])
axes[1].set_title("Avg Refunded Orders in Last 50 Days", fontweight="bold")
axes[1].set_ylabel("Order Count")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/q3_refund_abuse_patterns.png", dpi=300)
plt.close()

print(f"\nSuccess! 3 summary tables printed above and 3 charts saved to {OUTPUT_DIR}/")