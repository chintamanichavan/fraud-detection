# fraud-detection

Real-time transaction fraud scoring. Streaming ingestion, gradient-boosted model scoring, SHAP-based explanations, and observability built in. Uses a feature store so online and offline features stay consistent.

## Stack

- **Streaming** — `faust-streaming` for Kafka consumers
- **Models** — XGBoost + LightGBM, ensembled
- **Explanations** — SHAP for per-prediction attribution
- **Feature store** — Feast (offline: Parquet, online: Redis)
- **Data validation** — great-expectations checks on incoming events
- **Serving** — FastAPI + Uvicorn
- **Observability** — Prometheus client + OpenTelemetry tracing
- **Backing services** — Redis (features + cache), Postgres (events + outcomes)

## Setup

Requires Python ≥ 3.12.

```sh
uv venv
uv pip install -e ".[dev]"
cp .env.example .env       # configure Redis/Postgres/Kafka endpoints
```

## Run

```sh
# scoring API
uvicorn src.fraud_detection.api:app --reload

# streaming worker
faust -A src.fraud_detection.worker worker -l info
```

## Develop

```sh
ruff check
mypy src
pytest
```

## Status

Reference architecture for an online fraud pipeline. The model artifacts are not committed (see `.gitignore`); train your own from labeled data.
