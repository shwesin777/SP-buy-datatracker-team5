# SP-buy Fraud Detection — 10-Minute Presentation & Live Demo Storyboard (Team 5)

**Total Time Budget:** Strictly ≤ 10 minutes (Slides: ~5.5 mins | Live Demo: ~4.0 mins | Buffer: 0.5 mins)

---

## Part 1: Slide Presentation (~5 mins 30 secs)

| Slide | Topic | Speaker | Target Time | Key Points to Cover |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Title, Problem & Team Contributions** | **Shwe Sin** | 0:30 | SP-buy manual fraud rule limitations; team contribution split (e.g., 33% / 33% / 34%). |
| **2** | **EDA Findings & Automated Cleaning** | **Jaime** | 1:15 | Data issues found in `customers`, `orders`, `labels` (missing values, dtype mismatches, class imbalance) & `clean_data.py` fixes. |
| **3** | **Feature Engineering & Model Iteration** | **Ernest** | 1:30 | Engineered features (refund/cancel ratios, 50-day spend velocity); v1 → v2 comparison across Random Forest (Champion), Extra Trees, and HistGradientBoosting. |
| **4** | **AWS Cloud & MLflow Architecture** | **Shwe Sin** | 1:15 | AWS RDS MySQL (`spbuy-fraud-db`), S3 artifact bucket (`spbuy-mlflow-artifacts-team5`), Elastic Beanstalk MLflow server, and `.env` config management. |
| **5** | **End-to-End Automated ETL Pipeline** | **Jaime** | 1:00 | How `batch_pipeline.py` ingests dynamic URLs, validates schema, bulk-loads to RDS, retrains all 3 models, and logs to MLflow in **2m 20s** (< 3 min limit). |

---

## Part 2: Live Demonstration Walkthrough (~4 mins)

### Scene 1: Live MLflow Dashboard & Champion Model (1:00) — *Lead: Ernest*
1. Open the live MLflow tracking dashboard in the browser (do **not** use screenshots).
2. Filter experiment runs to show v1 vs. v2 model comparisons (F1-score, ROC-AUC, Precision/Recall).
3. Navigate to the **Models (Model Registry)** tab and show the Top 2 registered models and the promoted **Champion** model (`train_ernest` Random Forest v2) with its artifact stored in S3.

### Scene 2: On-the-Spot Batch ETL & Retraining Execution (2:30) — *Lead: Shwe Sin*
1. Receive the 3 new `*_XXX.csv` batch URLs (`customers_XXX.csv`, `orders_XXX.csv`, `labels_XXX.csv`) from the module tutor during the interview.
2. Run the automated pipeline from the terminal:
   ```bash
   python batch_pipeline.py --customers <URL_1> --orders <URL_2> --labels <URL_3>