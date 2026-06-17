# Basketball betting / NBA predictions

- **[PLANNING.md](PLANNING.md)** — data layout, what’s done, commands, next steps (edit here as the project evolves).
- **`backend/`** — Features, Kaggle → parquet, training, Flask API. Quick start: [`backend/README.md`](backend/README.md).
- **`lovable/`** — React frontend (Vite + TypeScript).

## Local development

Hybrid setup: Docker runs the backing services (Redis cache + MinIO, a local
S3-compatible stand-in for Cloudflare R2) and the Flask backend (pinned to Python
3.10.14). The Vite frontend runs natively for fast HMR.

**Prerequisites:** Docker Desktop (running), Node 18+, `make`.

```bash
cp .env.example .env     # placeholders are fine for local dev
make setup               # build image, start redis + minio + backend, health-check
make seed-r2             # one-time: push local data + model artifacts into MinIO
make cache               # warm Redis baselines (replaces the old "Redis not configured")
make frontend            # Vite on http://localhost:8080  (API on :5001)
```

| Service | URL |
| --- | --- |
| Flask API | http://localhost:5001/api |
| Frontend (Vite) | http://localhost:8080 |
| MinIO console | http://localhost:9001 |
| Redis | localhost:6379 |

Run `make help` for all targets. **Local vs prod cache:** local dev uses the Docker
Redis (`REDIS_URL`); production/CI uses Upstash (`UPSTASH_REDIS_URL`) — `app.py` reads
either. Do not put real secrets in `.env`; rotate any that were in `connection.txt`.
