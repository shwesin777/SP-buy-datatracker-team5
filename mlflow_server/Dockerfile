FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.18.0 pymysql boto3 cryptography

EXPOSE 5000

CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri \"$MLFLOW_BACKEND_URI\" --artifacts-destination s3://spbuy-mlflow-artifacts-team5 --serve-artifacts"]