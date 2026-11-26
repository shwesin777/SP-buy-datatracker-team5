from uuid import uuid4

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri(
    "http://spbuy-mlflow-v2-env.eba-vg28gb2y.us-east-1.elasticbeanstalk.com"
)
mlflow.set_experiment(f"pbi2-versioning-smoke-{uuid4().hex[:8]}")

features = [[0], [1], [2], [3]]
labels = [0, 0, 1, 1]

model = LogisticRegression(random_state=42)
model.fit(features, labels)

with mlflow.start_run(run_name="after-enabling-s3-versioning") as run:
    run_id = run.info.run_id
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("smoke_test_accuracy", model.score(features, labels))
    model_info = mlflow.sklearn.log_model(model, name="smoke-model")

saved_run = mlflow.get_run(run_id)
assert saved_run.info.status == "FINISHED"
assert saved_run.data.params["model_type"] == "logistic_regression"
assert saved_run.data.metrics["smoke_test_accuracy"] == 1.0

loaded_model = mlflow.sklearn.load_model(model_info.model_uri)
predictions = loaded_model.predict([[0], [3]]).tolist()
assert predictions == [0, 1]

print("Run ID:", run_id)
print("Model URI:", model_info.model_uri)
print("Predictions:", predictions)