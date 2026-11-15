import re
import zipfile
from pathlib import Path

SRC = Path("mlflow_server")
OUT = Path("mlflow_bundle_v8.zip")
ARTIFACT_ROOT = "s3://spbuy-mlflow-artifacts-team5"

old = (SRC / "Dockerfile").read_text(encoding="utf-8", errors="ignore") if (SRC / "Dockerfile").exists() else ""
m = re.search(r"mlflow(?:\[[^\]]*\])?\s*([=<>~!]=[^\s\"']+)", old)
pin = f"mlflow{m.group(1)}" if m else "mlflow"
print("MLflow requirement:", pin)
if not m:
    print("WARNING: no version pin found. Edit `pin` above to match the version you initialised the DB with.")

dockerfile = f"""FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "{pin}" pymysql cryptography boto3
EXPOSE 5000
CMD mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri "${{MLFLOW_BACKEND_URI}}" --default-artifact-root {ARTIFACT_ROOT} --allowed-hosts '*' --cors-allowed-origins 'http://spbuy-mlflow-v2-env.eba-vg28gb2y.us-east-1.elasticbeanstalk.com,https://spbuy-mlflow-v2-env.eba-vg28gb2y.us-east-1.elasticbeanstalk.com'
"""

SKIP_DIRS = {"__pycache__", ".git", ".venv"}
TEXT_EXT = {".ebextensions", ".config", ".conf", ".txt", ".json", ".yml", ".yaml", ".sh", ".py", ".md"}

if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("Dockerfile", dockerfile.replace("\r\n", "\n"))
    for p in sorted(SRC.rglob("*")):
        rel = p.relative_to(SRC)
        if p.is_dir() or p.name == "Dockerfile" or p.suffix == ".zip":
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        data = p.read_bytes()
        if p.suffix in TEXT_EXT or ".ebextensions" in rel.parts:
            data = data.replace(b"\r\n", b"\n")
        z.writestr(rel.as_posix(), data)

with zipfile.ZipFile(OUT) as z:
    for n in z.namelist():
        print(" ", n)
print("Built", OUT)