import pandas as pd
from dotenv import load_dotenv
from db_config import get_db_engine
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

load_dotenv()

query = """
SELECT
    LEAST(num_items_ordered, 5) AS basket_size,
    COUNT(*) AS labelled_orders,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_orders,
    ROUND(100.0 * AVG(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END), 2)
        AS fraud_rate_pct
FROM cleaned_fraud_data
WHERE is_fraud IN (0, 1)
  AND num_items_ordered >= 1
GROUP BY LEAST(num_items_ordered, 5)
ORDER BY basket_size
"""

value_query = """
SELECT
    value_band,
    COUNT(*) AS labelled_orders,
    SUM(is_fraud) AS fraud_orders,
    ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN order_value ELSE 0 END), 2)
        AS fraud_order_value
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
    results = pd.read_sql(query, con=engine)
    print("\nQuery 1 — Fraud by basket size")
    print(results.to_string(index=False))

    value_results = pd.read_sql(value_query, con=engine)
    print("\nQuery 2 — Fraud by order value")
    print(value_results.to_string(index=False))

    refund_results = pd.read_sql(refund_query, con=engine)
    print("\nQuery 3 — Refunds by fraud label")
    print(refund_results.to_string(index=False))
    
finally:
    engine.dispose()

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

print(f"Dashboard saved to: {dashboard_path}")