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
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
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
    "http://spbuy-mlflow-env.eba-yspwy5bb.us-east-1.elasticbeanstalk.com/",
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME_JAIME", "SP-Buy-Fraud-Detection-Jaime")

CATEGORICAL = ["country_code", "collect_type", "payment_method"]
BASE_FEATURES = CATEGORICAL + [
    "mobile_verified",
    "num_orders_last_50days",
    "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days",
    "total_payment_last_50days",
    "num_associated_customers",
    "order_value",
    "num_items_ordered",
    "customer_age_days",
    "order_day_of_week",
    "order_month",
]


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
    out = df.copy()
    count = out["num_orders_last_50days"].clip(lower=0)
    age = out["customer_age_days"].clip(lower=0)

    out["orders_per_account_day"] = count / (age + 1)
    out["spend_per_account_day"] = out["total_payment_last_50days"] / (age + 1)
    out["associated_per_account_day"] = out["num_associated_customers"] / (age + 1)
    out["value_per_item"] = out["order_value"] / out["num_items_ordered"].clip(lower=1)
    out["cancel_rate_50d"] = out["num_cancelled_orders_last_50days"] / (count + 1)
    out["refund_rate_50d"] = out["num_refund_orders_last_50days"] / (count + 1)
    out["unverified_account_velocity"] = (1 - out["mobile_verified"]) * out["orders_per_account_day"]
    out["weekend_order"] = out["order_day_of_week"].isin([5, 6]).astype(int)

    return out.replace([np.inf, -np.inf], np.nan)


def build_pipeline(features: pd.DataFrame, n_jobs: int = -1, **parameters) -> Pipeline:
    """Builds preprocessing and ExtraTreesClassifier pipeline."""
    num_cols = [c for c in features.columns if c not in CATEGORICAL]

    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, CATEGORICAL),
        ]
    )

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", ExtraTreesClassifier(random_state=42, n_jobs=n_jobs, **parameters)),
        ]
    )


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
    Runs Jaime's v1 and v2 ML experiments.
    Can be called standalone OR imported by the automated ETL pipeline with a pre-loaded df and batch_tag.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if batch_tag is None:
        batch_tag = os.getenv("DATA_BATCH_TAG", "base")

    if df is None:
        df = load_data_from_rds()

    df = df.replace([np.inf, -np.inf], np.nan)

    y = pd.to_numeric(df["is_fraud"], errors="raise").astype(int)
    X_v1 = df[BASE_FEATURES].copy()
    X_v1["mobile_verified"] = X_v1["mobile_verified"].replace(
        {True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0}
    )
    for col in X_v1.columns:
        if col not in CATEGORICAL:
            X_v1[col] = pd.to_numeric(X_v1[col], errors="raise")
    X_v1 = X_v1.replace([np.inf, -np.inf], np.nan)

    X_train_v1, X_test_v1, y_train, y_test = train_test_split(
        X_v1, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================================================
    # MODEL V1: Baseline Extra Trees (Base Features)
    # =========================================================
    pipeline_v1 = build_pipeline(
        X_train_v1,
        n_jobs=1,
        n_estimators=30,
        max_depth=8,
        min_samples_leaf=20,
        class_weight=None,
    )

    run_name_v1 = f"v1_baseline_extratrees_jaime_{batch_tag}"
    if False:  # skipped v1 baseline for fast ETL retraining
        mlflow.set_tag("author", "JaimeHoyr")
        mlflow.set_tag("model_version", "v1_baseline")
        mlflow.set_tag("data_batch", batch_tag)
        mlflow.log_param("model_type", "ExtraTreesClassifier")
        mlflow.log_param("feature_set", f"base_{X_v1.shape[1]}_features")
        mlflow.log_param("class_weight", "None")

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
    # MODEL V2: Feature-Engineered + Tuned Extra Trees
    # =========================================================
    X_v2 = engineer_additional_features(X_v1)
    X_train_v2, X_test_v2, _, _ = train_test_split(
        X_v2, y, test_size=0.2, random_state=42, stratify=y
    )

    # Keep inner n_jobs=1 so RandomizedSearchCV(n_jobs=1) doesn't cause Windows CPU thrashing,
    # and cap max_depth at 20 (avoiding None) so the .pkl artifact stays small for fast S3 upload.
    base_et = build_pipeline(X_train_v2, n_jobs=1, class_weight="balanced")

    param_dist = {
        "clf__n_estimators": [30],
        "clf__max_depth": [8],
        "clf__min_samples_leaf": [10, 20, 30],
        "clf__max_features": ["sqrt", 0.7],
    }

    search = RandomizedSearchCV(
        base_et,
        param_distributions=param_dist,
        n_iter=1,
        scoring="average_precision",
        cv=StratifiedKFold(2, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        verbose=1,
    )

    run_name_v2 = f"v2_tuned_extratrees_fe_jaime_{batch_tag}"
    with mlflow.start_run(run_name=run_name_v2):
        mlflow.set_tag("author", "JaimeHoyr")
        mlflow.set_tag("model_version", "v2_optimized_fe")
        mlflow.set_tag("data_batch", batch_tag)
        mlflow.log_param("model_type", "ExtraTreesClassifier")
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
