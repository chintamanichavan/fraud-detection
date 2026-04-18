from __future__ import annotations
import uuid
import json
from datetime import datetime
import psycopg
from ..models.transaction import AuditRecord, FeatureVector, ScoredTransaction

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_id       TEXT PRIMARY KEY,
    txn_id         TEXT NOT NULL,
    decision       TEXT NOT NULL,
    risk_score     REAL NOT NULL,
    shap_values    JSONB NOT NULL,
    features       JSONB NOT NULL,
    model_version  TEXT NOT NULL,
    explanation    TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_txn ON audit_trail (txn_id);
"""

_INSERT = """
INSERT INTO audit_trail
    (audit_id, txn_id, decision, risk_score, shap_values, features, model_version, explanation, created_at)
VALUES
    (%(audit_id)s, %(txn_id)s, %(decision)s, %(risk_score)s,
     %(shap_values)s::jsonb, %(features)s::jsonb, %(model_version)s, %(explanation)s, %(created_at)s)
ON CONFLICT (audit_id) DO NOTHING;
"""


def _build_explanation(scored: ScoredTransaction, features: FeatureVector) -> str:
    top = sorted(scored.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    drivers = ", ".join(f"{k} ({v:+.3f})" for k, v in top)
    return (
        f"Decision: {scored.decision.value} (score={scored.risk_score:.4f}). "
        f"Top SHAP drivers: {drivers}. "
        f"Model: {scored.model_version}."
    )


class AuditTrail:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def setup(self) -> None:
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            await conn.execute(_CREATE_TABLE)

    async def write(self, scored: ScoredTransaction, features: FeatureVector) -> AuditRecord:
        record = AuditRecord(
            audit_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            txn_id=scored.txn_id,
            decision=scored.decision,
            risk_score=scored.risk_score,
            shap_values=scored.shap_values,
            model_version=scored.model_version,
            features=features.model_dump(),
            explanation=_build_explanation(scored, features),
            created_at=datetime.utcnow(),
        )

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            await conn.execute(_INSERT, {
                **record.model_dump(),
                "decision":    record.decision.value,
                "shap_values": json.dumps(record.shap_values),
                "features":    json.dumps(record.features),
            })

        return record
