# Pinned to backend/.python-version (3.10.14) to kill the local 3.12-vs-3.10 skew.
FROM python:3.10.14-slim

# Runtime system libs: libgomp1 provides OpenMP for LightGBM / XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Dependency layer first so it caches unless requirements.txt changes.
COPY config/requirements.txt config/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r config/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    FLASK_APP=web.app

EXPOSE 5001

# Code, data, and model artifacts are bind-mounted by docker-compose at runtime,
# so the image stays small and edits hot-reload.
# We use `flask run` (not `python web/app.py`) to bind 0.0.0.0 without editing
# app.py, which hardcodes host=127.0.0.1 (backlog 10.1).
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001", "--debug"]
