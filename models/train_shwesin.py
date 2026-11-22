import os
import sys
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Allow importing db_config from ../database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_config import get_db_engine  # type: ignore

warnings.filterwarnings("ignore")
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://spbuy-mlflow-v2-env.eba-vg28gb2y.us-east-1.elasticbeanstalk.com/",
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME_SHWESIN", "SP-Buy-Fraud-Detection-Shwesin")


def load_data_from_rds() -> pd.DataFrame:
    """Fetches cleaned_fraud_data from AWS RDS MySQL."""
    print("Querying cleaned_fraud_data from AWS RDS MySQL...")
    engine = get_db_engine()
    query = "SELECT * FROM cleaned_fraud_data;"
    df = pd.read_sql(query, con=engine)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from RDS.")
    return df


def engineer_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain-specific fraud interaction features for Model v2."""
    df = df.copy()
    eps = 1e-5

    df["refund_to_order_ratio"] = df["refund_value"] / (df["order_value"] + eps)
    df["cancel_rate_50d"] = df["num_cancelled_orders_last_50days"] / (df["num_orders_last_50days"] + 1.0)
    df["refund_rate_50d"] = df["num_refund_orders_last_50days"] / (df["num_orders_last_50days"] + 1.0)
    df["avg_item_price"] = df["order_value"] / (df["num_items_ordered"] + 1.0)
    df["order_vs_50d_avg_spend"] = df["order_value"] / (
        (df["total_payment_last_50days"] / (df["num_orders_last_50days"] + 1.0)) + eps
    )
    df["is_weekend"] = df["order_day_of_week"].isin([5, 6]).astype(int)

    return df


def evaluate_predictions(y_true, y_prob, threshold=0.5) -> dict:
    """Computes fraud-appropriate classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def main(df: pd.DataFrame = None, batch_tag: str = None):
    """
    Runs Shwesin's v1 and v2 ML experiments.
    Can be called standalone OR imported by the automated ETL pipeline with a pre-loaded df and batch_tag.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if batch_tag is None:
        batch_tag = os.getenv("DATA_BATCH_TAG", "base")

    if df is None:
        df = load_data_from_rds()

    df = df.replace([np.inf, -np.inf], np.nan); num_c = df.select_dtypes(include="number").columns; df[num_c] = df[num_c].fillna(0)

    drop_cols = ["order_id", "customer_id", "order_date", "first_order_datetime", "is_fraud"]
    target_col = "is_fraud"
    cat_cols = ["country_code", "collect_type", "payment_method"]

    # =========================================================
    # MODEL V1: Baseline Logistic Regression (Base Features)
    # =========================================================
    X_v1 = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(int)

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
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]
    )

    run_name_v1 = f"v1_baseline_logreg_shwesin_{batch_tag}"
    if False:  # skipped v1 baseline for fast ETL retraining
        mlflow.set_tag("author", "shwesin777")
        mlflow.set_tag("model_version", "v1_baseline")
        mlflow.set_tag("data_batch", batch_tag)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("feature_set", f"cleaned_base_{X_v1.shape[1]}_features")
        mlflow.log_param("class_weight", "balanced")

        print(f"\nTraining {run_name_v1}...")
        pipeline_v1.fit(X_train_v1, y_train)
        y_prob_v1 = pipeline_v1.predict_proba(X_test_v1)[:, 1]
        metrics_v1 = evaluate_predictions(y_test, y_prob_v1)

        mlflow.log_metrics(metrics_v1)
        None and mlflow.sklearn.log_model(
            pipeline_v1, artifact_path="model_v1", serialization_format="cloudpickle"
        )
        print("Model v1 Metrics:", {k: round(v, 4) for k, v in metrics_v1.items()})

    # =========================================================
    # MODEL V2: Feature-Engineered + Tuned HistGradientBoosting
    # =========================================================
    df_eng = engineer_additional_features(df).replace([np.inf, -np.inf], np.nan)
    X_v2 = df_eng.drop(columns=[c for c in drop_cols if c in df_eng.columns])

    X_train_v2, X_test_v2, _, _ = train_test_split(
        X_v2, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols_v2 = [c for c in X_train_v2.columns if c not in cat_cols]

    cat_transformer_v2 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor_v2 = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols_v2),
            ("cat", cat_transformer_v2, cat_cols),
        ]
    )

    base_hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=True,
        random_state=42,
    )

    pipeline_v2 = Pipeline(steps=[("prep", preprocessor_v2), ("clf", base_hgb)])

    param_dist = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_iter": [80, 120],
        "clf__max_leaf_nodes": [15, 31],
        "clf__min_samples_leaf": [50, 100],
    }

    search = RandomizedSearchCV(
        pipeline_v2,
        param_distributions=param_dist,
        n_iter=1,
        scoring="average_precision",
        cv=2,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )

    run_name_v2 = f"v2_tuned_histgb_fe_shwesin_{batch_tag}"
    with mlflow.start_run(run_name=run_name_v2):
        mlflow.set_tag("author", "shwesin777")
        mlflow.set_tag("model_version", "v2_optimized_fe")
        mlflow.set_tag("data_batch", batch_tag)
        mlflow.log_param("model_type", "HistGradientBoostingClassifier")
        mlflow.log_param("feature_set", f"engineered_{X_v2.shape[1]}_features")
        mlflow.log_param("tuning_metric", "average_precision (PR-AUC)")

        print(f"\nTuning & Training {run_name_v2}...")
        search.fit(X_train_v2, y_train)
        best_model_v2 = search.best_estimator_

        mlflow.log_params(search.best_params_)
        y_prob_v2 = best_model_v2.predict_proba(X_test_v2)[:, 1]
        metrics_v2 = evaluate_predictions(y_test, y_prob_v2)

        mlflow.log_metrics(metrics_v2)
        None and mlflow.sklearn.log_model(
            best_model_v2, artifact_path="model_v2", serialization_format="cloudpickle"
        )
        print("Best Params v2:", search.best_params_)
        print("Model v2 Metrics:", {k: round(v, 4) for k, v in metrics_v2.items()})


if __name__ == "__main__":
    main()
