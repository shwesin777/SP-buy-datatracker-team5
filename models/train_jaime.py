"""PBI 6: Jaime's baseline and optimized Extra Trees experiments.

Run from project root: python models/train_jaime.py
Use shared MLflow instead of local recording: python models/train_jaime.py --remote
Optional local data: python models/train_jaime.py --csv cleaned_base.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
CATEGORICAL = ['country_code', 'collect_type', 'payment_method']
# Use only fields expected to be available when the order is placed.
# Current-order refund fields are excluded because refunds may occur later.
BASE_FEATURES = CATEGORICAL + [
    'mobile_verified', 'num_orders_last_50days',
    'num_cancelled_orders_last_50days', 'num_refund_orders_last_50days',
    'total_payment_last_50days', 'num_associated_customers',
    'order_value', 'num_items_ordered', 'customer_age_days',
    'order_day_of_week', 'order_month',
]


def engineer_features(frame):
    out = frame.copy()
    count = out['num_orders_last_50days'].clip(lower=0)
    age = out['customer_age_days'].clip(lower=0)
    out['orders_per_account_day'] = count / (age + 1)
    out['spend_per_account_day'] = out['total_payment_last_50days'] / (age + 1)
    out['associated_per_account_day'] = out['num_associated_customers'] / (age + 1)
    out['value_per_item'] = out['order_value'] / out['num_items_ordered'].clip(lower=1)
    out['cancel_rate_50d'] = out['num_cancelled_orders_last_50days'] / (count + 1)
    out['refund_rate_50d'] = out['num_refund_orders_last_50days'] / (count + 1)
    out['unverified_account_velocity'] = (1 - out['mobile_verified']) * out['orders_per_account_day']
    out['weekend_order'] = out['order_day_of_week'].isin([5, 6]).astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def build_pipeline(frame, **parameters):
    numeric = [col for col in frame.columns if col not in CATEGORICAL]
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ('prep', preprocessing),
        ('clf', ExtraTreesClassifier(random_state=42, n_jobs=-1, **parameters)),
    ])


def evaluate(model, features, labels):
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'roc_auc': roc_auc_score(labels, probabilities),
        'average_precision': average_precision_score(labels, probabilities),
        'f1_score': f1_score(labels, predictions, zero_division=0),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
    }
    return metrics, confusion_matrix(labels, predictions, labels=[0, 1]).tolist()


def main(df=None, batch_tag=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true', help='Log to team MLflow server')
    parser.add_argument('--csv', type=Path, help='Read local cleaned CSV instead of RDS')
    args = parser.parse_args() if df is None else parser.parse_args([])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'results' / f'jaime_{stamp}'
    output.mkdir(parents=True, exist_ok=True)

    if df is None:
        if args.csv:
            df = pd.read_csv(args.csv, low_memory=False)
        else:
            sys.path.insert(0, str(ROOT / 'database'))
            from db_config import get_db_engine
            engine = get_db_engine()
            print('Loading cleaned_fraud_data from RDS...', flush=True)
            try:
                with engine.connect() as connection:
                    df = pd.read_sql('SELECT * FROM cleaned_fraud_data', connection)
            finally:
                engine.dispose()
    print(f'Loaded {len(df):,} rows.', flush=True)
    y = pd.to_numeric(df['is_fraud'], errors='raise')
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('is_fraud must contain both 0 and 1, with no missing labels.')
    y = y.astype(int)
    X = df[BASE_FEATURES].copy()
    X['mobile_verified'] = X['mobile_verified'].replace({
        True: 1, False: 0, 'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0,
    })
    for col in X.columns:
        if col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors='raise')
    X = X.replace([np.inf, -np.inf], np.nan)
    train, test = train_test_split(np.arange(len(X)), test_size=0.2,
                                   stratify=y, random_state=42)
    y_train, y_test = y.iloc[train], y.iloc[test]
    engineered = engineer_features(X)
    batch_tag = batch_tag or os.getenv('DATA_BATCH_TAG', 'base')
    results = []

    # A separate local store keeps recording available while the shared server is down.
    uri = os.getenv('MLFLOW_TRACKING_URI') if args.remote else f'sqlite:///{(ROOT / "mlflow_jaime.db").as_posix()}'
    if not uri:
        raise ValueError('Set MLFLOW_TRACKING_URI in .env before using --remote.')
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('SP-Buy-Fraud-Detection-Jaime')

    for version, features in [('v1_baseline', X), ('v2_optimized', engineered)]:
        print(f'\nTraining {version}...', flush=True)
        model = build_pipeline(features, n_estimators=100, max_depth=10,
                               min_samples_leaf=20, class_weight=None)
        with mlflow.start_run(run_name=f'{version}_extratrees_jaime_{batch_tag}_{stamp}'):
            mlflow.set_tags({'author': 'jaime', 'model_version': version,
                             'data_batch': batch_tag, 'recording_mode': 'remote' if args.remote else 'local'})
            mlflow.log_params({'model_type': 'ExtraTreesClassifier', 'random_state': 42,
                               'test_size': 0.2, 'threshold': 0.5,
                               'feature_count': features.shape[1],
                               'current_order_refunds_excluded': True})
            mlflow.log_dict({'columns': features.columns.tolist()}, 'feature_columns.json')
            if version == 'v2_optimized':
                model.set_params(clf__class_weight='balanced')
                search = RandomizedSearchCV(
                    model,
                    {'clf__n_estimators': [120, 180], 'clf__max_depth': [12, 20, None],
                     'clf__min_samples_leaf': [5, 15, 30],
                     'clf__max_features': ['sqrt', 0.7]},
                    n_iter=4, scoring='average_precision',
                    cv=StratifiedKFold(3, shuffle=True, random_state=42),
                    random_state=42, n_jobs=1, verbose=1,
                )
                search.fit(features.iloc[train], y_train)
                model = search.best_estimator_
                mlflow.log_metric('cv_average_precision', search.best_score_)
            else:
                model.fit(features.iloc[train], y_train)
            mlflow.log_params({key: value for key, value in model.get_params().items()
                               if key.startswith('clf__')})
            metrics, matrix = evaluate(model, features.iloc[test], y_test)
            results.append({'version': version, **metrics})
            # Save results before remote artifact uploads so metrics remain available.
            pd.DataFrame(results).to_csv(output / 'model_comparison.csv', index=False)
            (output / f'{version}_confusion_matrix.json').write_text(json.dumps(
                {'labels': [0, 1], 'rows': 'actual', 'columns': 'predicted', 'matrix': matrix}, indent=2))
            mlflow.log_metrics(metrics)
            mlflow.log_dict({'labels': [0, 1], 'matrix': matrix}, 'confusion_matrix.json')
            mlflow.sklearn.log_model(model, artifact_path='model', serialization_format='cloudpickle')
            print({name: round(value, 4) for name, value in metrics.items()}, flush=True)
    print('\nComparison:\n', pd.DataFrame(results).to_string(index=False))
    print(f'\nResults saved in: {output}')
    if not args.remote:
        print('Runs recorded locally. They are NOT yet on the shared server.')
        print('When the server works, rerun with --remote to record team-visible runs.')


if __name__ == '__main__':
    main()
