# Trust & Safety Fraud Trend Analysis Report

## 1. High-Risk Geographic & Payment Corridors (Query 1)
- **Highest Risk Segment:** `PK` using `jazzcash wallet` (`pickup`) recorded a **69.84%** fraud rate across 63 orders (**SGD 95.50** total fraud exposure).
- **Actionable Insight:** Apply dynamic risk thresholds by `country_code` and `payment_method` rather than a single global rule.

## 2. Account Maturity & Verification Impact (Query 2)
- Comparing `< 1 Day` accounts against `> 30 Days` established accounts shows that while brand-new accounts carry elevated risk, **unverified mobile numbers (`mobile_verified = 0`)** and **missing/invalid customer records** exhibit disproportionately high fraud rates even among older accounts.
- **Actionable Insight:** Replace the rigid legacy rule (`order_value > $X AND age < 1 day`) with ML probability scoring that weighs `mobile_verified` and historical ratios.

## 3. Repeat Refund Abuse & Linked Account Rings (Query 3)
- Identified **27,951 repeat customers** with confirmed fraudulent orders; their median 50-day refund ratio (`num_refund_orders_last_50days / num_orders_last_50days`) and `num_associated_customers` count substantially exceed legitimate multi-order buyers.
- **Actionable Insight:** Automatically route accounts with multiple associated customer IDs or elevated 50-day refund velocity to pre-shipment manual review.

## 4. Transaction Timing Patterns (Query 4)
- **Peak Fraud Window:** Month **3** on **Sat** reached a peak fraud rate of **13.33%** across 22,224 orders.
- **Actionable Insight:** Increase Trust & Safety analyst staffing and tighten automated fraud thresholds during identified peak day-of-week and seasonal windows.
