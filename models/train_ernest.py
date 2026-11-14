import os
import sys
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
)

# Allow importing db_config from ../database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_config import get_db_engine  # type: ignore

warnings.filterwarnings("ignore")
load_dotenv()

# Point to deployed MLflow tracking URI (falls back to local server if not set in .env)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SP-Buy-Fraud-Detection-Ernest")


def load_data_from_rds() -> pd.DataFrame:
    """Fetches cleaned_fraud_data from AWS RDS MySQL."""
    print("Querying cleaned_fraud_data from AWS RDS MySQL...")
    engine = get_db_engine()
    query = "SELECT * FROM cleaned_fraud_data;"
    df = pd.read_sql(query, con=engine)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from RDS.")
    return df


def engineer_ernest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-specific fraud features for Model v2 focusing on:
    1. Legacy rule modernization (new account + high order value interaction)
    2. Combined problem-order velocity (cancellations + refunds in last 50 days)
    3. Unverified mobile + high spend risk interaction
    4. Associated customer network risk & cyclical temporal encoding
    """
    df = df.copy()
    eps = 1e-5

    # Ensure mobile_verified is numeric 0/1
    if df["mobile_verified"].dtype == object or df["mobile_verified"].dtype == bool:
        df["mobile_verified"] = (
            df["mobile_verified"]
            .map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
            .fillna(0)
            .astype(int)
        )

    # 1. Combined bad-behavior rate over the last 50 days (refunds + cancellations)
    df["problem_orders_50d"] = (
        df["num_refund_orders_last_50days"] + df["num_cancelled_orders_last_50days"]
    )
    df["problem_order_rate_50d"] = df["problem_orders_50d"] / (
        df["num_orders_last_50days"] + 1.0
    )

    # 2. Modernized soft version of Trust & Safety's legacy rule ("account < 1 day old & high value")
    df["is_brand_new_account"] = (df["customer_age_days"] <= 1).astype(int)
    df["new_account_order_value_risk"] = df["order_value_log"] / (
        np.log1p(df["customer_age_days"].clip(lower=0)) + 1.0
    )

    # 3. Unverified phone number combined with order value
    df["unverified_order_risk"] = (1 - df["mobile_verified"]) * df["order_value_log"]

    # 4. Associated customer network risk (fraud rings often share linked accounts)
    df["has_associated_accounts"] = (df["num_associated_customers"] > 0).astype(int)
    df["associated_to_order_ratio"] = df["num_associated_customers"] / (
        df["num_orders_last_50days"] + 1.0
    )

    # 5. Spend per historical order vs current order value
    hist_avg_order = df["total_payment_last_50days"] / (df["num_orders_last_50days"] + 1.0)
    df["order_to_hist_avg_diff"] = df["order_value"] - hist_avg_order

    # 6. Cyclical encoding of day of week (captures weekend/weekday transitions smoothly)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["order_day_of_week"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["order_day_of_week"] / 7.0)

    return df


def find_optimal_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Finds the probability threshold that maximizes F1-score on validation predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])  # last element has no corresponding threshold
    return float(np.clip(thresholds[best_idx], 0.15, 0.85))


def evaluate_predictions(y_true, y_prob, threshold=0.5) -> dict:
    """Computes fraud classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def main(df: pd.DataFrame = None, batch_tag: str = None):
    """
    Runs Ernest's v1 and v2 ML experiments.
    Can be called standalone OR imported by the automated ETL pipeline with a pre-loaded df and batch_tag.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if batch_tag is None:
        batch_tag = os.getenv("DATA_BATCH_TAG", "base")

    if df is None:
        df = load_data_from_rds()

    df = df.replace([np.inf, -np.inf], np.nan)

    target_col = "is_fraud"
    drop_cols = ["order_id", "customer_id", "order_date", "first_order_datetime", target_col]
    cat_cols = ["country_code", "collect_type", "payment_method"]

    # Raw skewed columns that already have _log counterparts in the 39 columns
    raw_skewed_cols = [
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "total_payment_last_50days",
        "num_associated_customers",
        "order_value",
        "num_items_ordered",
        "refund_value",
    ]

    y = df[target_col].astype(int)

    # =========================================================================
    # MODEL V1: Baseline Decision Tree (Unweighted, Non-Collinear Base Features)
    # Diagnosis Goal: Show how unweighted shallow trees suffer from low Recall/F1
    # on the 11% minority fraud class without feature engineering or class weights.
    # =========================================================================
    X_v1 = df.drop(columns=[c for c in drop_cols + raw_skewed_cols if c in df.columns])

    X_train_v1, X_test_v1, y_train, y_test = train_test_split(
        X_v1, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols_v1 = [c for c in X_train_v1.columns if c not in cat_cols]

    preprocessor_v1 = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols_v1,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    pipeline_v1 = Pipeline(
        steps=[
            ("prep", preprocessor_v1),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=25,
                    class_weight=None,  # Deliberately unweighted baseline to diagnose imbalance impact
                    random_state=42,
                ),
            ),
        ]
    )

    run_name_v1 = f"v1_baseline_dtree_ernest_{batch_tag}"
    with mlflow.start_run(run_name=run_name_v1):
        mlflow.set_tag("author", "ernest")
        mlflow.set_tag("model_version", "v1_baseline")
        mlflow.set_tag("data_batch", batch_tag)
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("feature_set", f"non_collinear_base_{len(X_v1.columns)}_features")
        mlflow.log_param("class_weight", "None")
        mlflow.log_param("max_depth", 6)

        print(f"\nTraining {run_name_v1}...")
        pipeline_v1.fit(X_train_v1, y_train)
        y_prob_v1 = pipeline_v1.predict_proba(X_test_v1)[:, 1]
        metrics_v1 = evaluate_predictions(y_test, y_prob_v1, threshold=0.5)

        mlflow.log_metrics(metrics_v1)
        mlflow.sklearn.log_model(
            pipeline_v1, artifact_path="model_v1", serialization_format="cloudpickle"
        )
        print("Model v1 Metrics:", {k: round(v, 4) for k, v in metrics_v1.items()})

    # =========================================================================
    # MODEL V2: Feature-Engineered + Tuned Random Forest + Threshold Optimization
    # Optimization Goal: Address v1's variance and class imbalance using
    # behavioral risk ratios, balanced_subsample weights, and F1 threshold tuning.
    # =========================================================================
    df_eng = engineer_ernest_features(df).replace([np.inf, -np.inf], np.nan)
    X_v2 = df_eng.drop(columns=[c for c in drop_cols + raw_skewed_cols if c in df_eng.columns])

    X_train_v2, X_test_v2, _, _ = train_test_split(
        X_v2, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split training further into sub-train and validation for clean threshold optimization
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_v2, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    num_cols_v2 = [c for c in X_train_v2.columns if c not in cat_cols]

    preprocessor_v2 = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols_v2),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    base_rf = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    pipeline_v2 = Pipeline(steps=[("prep", preprocessor_v2), ("clf", base_rf)])

    param_dist = {
        "clf__n_estimators": [120, 180],
        "clf__max_depth": [12, 16, 20],
        "clf__min_samples_leaf": [10, 25, 50],
        "clf__max_features": ["sqrt", 0.5],
    }

    search = RandomizedSearchCV(
        pipeline_v2,
        param_distributions=param_dist,
        n_iter=4,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    run_name_v2 = f"v2_tuned_rf_fe_ernest_{batch_tag}"
    with mlflow.start_run(run_name=run_name_v2):
        mlflow.set_tag("author", "ernest")
        mlflow.set_tag("model_version", "v2_optimized_fe")
        mlflow.set_tag("data_batch", batch_tag)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("feature_set", f"engineered_{len(X_v2.columns)}_features")
        mlflow.log_param("tuning_metric", "average_precision (PR-AUC) + F1 Threshold")

        print(f"\nTuning & Training {run_name_v2}...")
        search.fit(X_subtrain, y_subtrain)
        best_model_v2 = search.best_estimator_

        # Find optimal probability threshold on held-out validation set (no test leakage!)
        val_probs = best_model_v2.predict_proba(X_val)[:, 1]
        best_threshold = find_optimal_f1_threshold(y_val, val_probs)

        # Refit best estimator on full training data before final test evaluation
        best_model_v2.fit(X_train_v2, y_train)
        y_prob_v2 = best_model_v2.predict_proba(X_test_v2)[:, 1]
        metrics_v2 = evaluate_predictions(y_test, y_prob_v2, threshold=best_threshold)

        mlflow.log_params(search.best_params_)
        mlflow.log_param("optimal_probability_threshold", round(best_threshold, 4))
        mlflow.log_metrics(metrics_v2)
        mlflow.sklearn.log_model(
            best_model_v2, artifact_path="model_v2", serialization_format="cloudpickle"
        )
        print("Best Params v2:", search.best_params_)
        print(f"Optimal Decision Threshold: {best_threshold:.4f}")
        print("Model v2 Metrics:", {k: round(v, 4) for k, v in metrics_v2.items()})


if __name__ == "__main__":
    main()