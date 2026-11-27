FROM prefecthq/prefect:3-latest

RUN pip install --no-cache-dir asyncpg psycopg2-binary

EXPOSE 4200

ENV PREFECT_SERVER_API_HOST=0.0.0.0
ENV PREFECT_SERVER_API_PORT=4200
ENV PREFECT_SERVER_ANALYTICS_ENABLED=false

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
