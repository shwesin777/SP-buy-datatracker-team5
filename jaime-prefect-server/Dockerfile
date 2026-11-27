FROM python:3.12-slim

RUN pip install --no-cache-dir prefect==3.9.4

EXPOSE 5000

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "5000"]