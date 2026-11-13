FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow

WORKDIR /mlflow

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
    "--artifacts-destination", "/mlflow/artifacts", \
    "--allowed-hosts", "*", \
    "--cors-allowed-origins", "*"]