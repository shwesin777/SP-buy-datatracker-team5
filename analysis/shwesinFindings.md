# SP-buy Trust & Safety Fraud Analytics Findings
**Author:** Shwe Sin (2508315)  
**Data Source:** AWS RDS MySQL (`cleaned_fraud_data`)  
**Pipeline Scripts:** `database/fraud_analytics_shwesin.py` and `database/prefect_fraud_analytics_shwesin.py`  
**Dashboard:** [View Shwe Sin’s dashboard](../fraud_analysis_outputs/shwesin/shwesin_dashboard.png)

---

## 1. Fraud Exposure by Payment Method & Collection Type
**Business Question:** Which combinations of payment method and collection type exhibit the highest fraud rate (%) and total financial loss (SGD) on SP-buy?

### Top Findings & Metrics
| Payment Method | Collect Type | Total Orders | Fraud Orders | Fraud Rate (%) | Fraud Loss (SGD) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| jazzcash wallet | pickup | 63 | 44 | 69.84% | $95.50 |
| antfinancial bkash | pickup | 1,749 | 1,019 | 58.26% | $3,450.95 |
| payment on delivery | pickup | 164 | 81 | 49.39% | $1,121.78 |
| jazzcash wallet | Delivery | 2,446 | 1,183 | 48.36% | $5,830.86 |
| generic creditcard | pickup | 1,699 | 628 | 36.96% | $2,940.23 |
| credit card | pickup | 2,808 | 982 | 34.97% | $2,686.58 |
| no payment | Delivery | 576 | 191 | 33.16% | $1,502.23 |
| antfinancial bkash | Delivery | 16,140 | 3,629 | 22.48% | $17,634.67 |
| generic creditcard | Delivery | 70,626 | 13,425 | 19.01% | $104,385.97 |
| payment on delivery | Delivery | 312,349 | 50,173 | 16.06% | $312,128.00 |
| credit card | Delivery | 111,439 | 8,336 | 7.48% | $82,209.41 |

* **Key Insight:** Fraud risk concentrates heavily in specific payment-fulfillment pairs rather than uniformly across checkout channels. Pickup orders using mobile wallets (`jazzcash wallet` at 69.84%, `antfinancial bkash` at 58.26%) and `payment on delivery` (49.39%) have the highest fraud rates, while high-volume delivery channels (`payment on delivery` at $312,128.00 loss and `generic creditcard` at $104,385.97 loss) drive the largest total SGD losses[cite: 5].
* **Actionable Recommendation:** Replace blanket order-value thresholds with segmented pre-shipment review queues prioritized by `(payment_method, collect_type)` risk and SGD exposure.

---

## 2. High-Risk Customer Account Clusters & Mobile Verification
**Business Question:** How does fraud risk vary across customer account clusters segmented by mobile verification status and the number of linked/associated accounts?

### Cluster Summary Table
| Mobile Verified | Account Cluster | Total Orders | Fraud Orders | Fraud Rate (%) | Avg Order Value (SGD) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Unverified (`NaN`) | 3+ Linked Accounts (Ring Risk) | 7 | 2 | 28.57% | $5.02 |
| Verified (`1.0`) | 3+ Linked Accounts (Ring Risk) | 302,974 | 50,555 | 16.69% | $6.37 |
| Unverified (`0.0`) | 3+ Linked Accounts (Ring Risk) | 372 | 53 | 14.25% | $7.20 |
| Verified (`1.0`) | 1-2 Linked Accounts | 646,356 | 53,804 | 8.32% | $7.20 |
| Unverified (`0.0`) | 1-2 Linked Accounts | 269 | 21 | 7.81% | $7.07 |

* **Key Insight:** Accounts linked to 3 or more associated customer profiles (`3+ Linked Accounts (Ring Risk)`) exhibit roughly **2x the fraud rate** (~14.25%–28.57%) of single/dual-account profiles (`1-2 Linked Accounts` at ~7.81%–8.32%), indicating coordinated multi-account fraud rings even when mobile numbers are verified[cite: 5].
* **Actionable Recommendation:** Enforce step-up verification (secondary OTP or device-fingerprint checks) and automated order holds whenever `num_associated_customers >= 3`, while fast-tracking verified `1-2 Linked Accounts` orders to minimize false positives.

---

## 3. Refund & Cancellation Abuse Behavior
**Business Question:** How do order value, refund value, and 50-day historical refund/cancellation counts differ between legitimate and fraudulent orders?

### Behavioral Comparison Table
| Order Status | Total Orders | Avg Order Value (SGD) | Avg Refund Value (SGD) | Avg 50-Day Refund Count | Avg 50-Day Cancel Count |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Legitimate (`is_fraud = 0`) | 845,543 | $6.83 | $2.51 | 0.89 | 0.62 |
| Fraudulent (`is_fraud = 1`) | 104,435 | $7.78 | $0.00 | 1.13 | 0.92 |

* **Key Insight:** Fraudulent orders have higher average order values ($7.78 vs. $6.83) and noticeably higher 50-day refund velocity (1.13 vs. 0.89) and cancellation velocity (0.92 vs. 0.62) than legitimate orders, while showing $0.00 recorded average refund value due to pre-refund fraud interception/chargeback blocking[cite: 5].
* **Actionable Recommendation:** Flag repeat high-velocity accounts whose 50-day refund or cancellation counts exceed baseline thresholds (`> 1.0`) for manual review before approving instant refunds or repeat high-value checkouts.