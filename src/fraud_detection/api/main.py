from __future__ import annotations
import os
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from ..models.transaction import AuditRecord, ScoredTransaction, Transaction
from ..pipeline.features import engineer
from ..pipeline.scorer import FraudScorer
from ..audit.trail import AuditTrail

load_dotenv()

_scorer: FraudScorer
_redis:  aioredis.Redis
_audit:  AuditTrail


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _scorer, _redis, _audit
    _scorer = FraudScorer(os.environ["MODEL_PATH"])
    _redis  = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    _audit  = AuditTrail(os.environ["POSTGRES_DSN"])
    await _audit.setup()
    yield
    await _redis.aclose()


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://chintamani.dev"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class ScoreResponse(ScoredTransaction):
    audit_id: str
    explanation: str


@app.post("/score", response_model=ScoreResponse)
async def score(txn: Transaction) -> ScoreResponse:
    features = await engineer(txn, _redis)
    scored   = _scorer.score(features)
    record   = await _audit.write(scored, features)

    return ScoreResponse(
        **scored.model_dump(),
        audit_id=record.audit_id,
        explanation=record.explanation,
    )


@app.get("/audit/{txn_id}", response_model=list[AuditRecord])
async def get_audit(txn_id: str) -> list[AuditRecord]:
    import psycopg
    async with await psycopg.AsyncConnection.connect(os.environ["POSTGRES_DSN"]) as conn:
        rows = await conn.execute(
            "SELECT * FROM audit_trail WHERE txn_id = %s ORDER BY created_at DESC",
            (txn_id,),
        )
        results = await rows.fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="No audit records found.")
    cols = [d.name for d in rows.description]
    return [AuditRecord(**dict(zip(cols, r))) for r in results]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
