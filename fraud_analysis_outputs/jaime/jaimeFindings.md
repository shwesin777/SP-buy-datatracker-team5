# SP-Buy Trust & Safety: Fraud Trend & Risk Segment Analysis

[View the dashboard](SP_Buy_Fraud_Dashboard.png)

## Fraud increases most noticeably at five items

Fraud rates stay within a narrow range of **9.31% to 10.00%** for orders with one to four items, then rise to **16.14%** for orders with five or more. The **6.14 percentage-point** jump from the four-item group is based on a sizeable group of **204,389 orders**. Because the dashboard groups all baskets of five or more items together, it cannot show whether the rate continues to rise beyond five items. Comparing basket sizes within each order-value band would help determine whether basket size adds information beyond the price of the order.

## The highest fraud rate and the highest total fraud value occur in different bands

The fraud rate rises across every order-value band, from **10.71%** for orders under 25 to **57.59%** for orders of 100 or more. The 100+ group has the highest proportion of fraud-labelled orders, with **110 out of 191**. Orders under 25 account for **615,058.30** in fraud-labelled order value, around **76% of the total across all four bands**, because that band contains far more orders. For Trust & Safety, the 100+ group stands out when prioritising individual orders by observed fraud rate, while the under-25 group matters most when tracking aggregate fraud-labelled value. The value chart sums order amounts. Estimating actual losses would require separate loss data.

## Recorded refunds differ sharply by fraud label

None of the **104,422 fraud-labelled orders** had a positive recorded refund, compared with **68.83%** of non-fraud orders. Both groups had usable refund data for every order in the comparison. The difference may reflect how refunds are approved or recorded after an order is flagged. Check when refund amounts become available before using them as a model input. If they are only available after the fraud decision, including them in a model that scores orders earlier would cause data leakage.

*Source: `cleaned_fraud_data` in the team’s AWS RDS MySQL database. The charts compare groups within this dataset.*
