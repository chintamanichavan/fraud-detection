from __future__ import annotations
import os
from datetime import datetime
import numpy as np
import xgboost as xgb
import shap
from ..models.transaction import Decision, FeatureVector, ScoredTransaction

_THRESHOLD_FLAG   = float(os.getenv("RISK_THRESHOLD_FLAG",   "0.55"))
_THRESHOLD_REJECT = float(os.getenv("RISK_THRESHOLD_REJECT", "0.85"))
_MODEL_VERSION    = os.getenv("MODEL_VERSION", "xgb_v1")

_FEATURE_ORDER = [
    "velocity_1h", "velocity_24h", "geo_delta_km",
    "merchant_risk_score", "amount_zscore",
    "hour_of_day", "days_since_last_txn", "is_new_merchant",
]


class FraudScorer:
    def __init__(self, model_path: str) -> None:
        self._model = xgb.Booster()
        self._model.load_model(model_path)
        self._explainer = shap.TreeExplainer(self._model)

    def score(self, features: FeatureVector) -> ScoredTransaction:
        row = np.array([[
            features.velocity_1h,
            features.velocity_24h,
            features.geo_delta_km,
            features.merchant_risk_score,
            features.amount_zscore,
            features.hour_of_day,
            features.days_since_last_txn,
            int(features.is_new_merchant),
        ]], dtype=np.float32)

        dmatrix = xgb.DMatrix(row, feature_names=_FEATURE_ORDER)
        risk_score = float(self._model.predict(dmatrix)[0])

        shap_vals = self._explainer.shap_values(dmatrix)[0]
        shap_map  = {k: round(float(v), 5) for k, v in zip(_FEATURE_ORDER, shap_vals)}

        if risk_score >= _THRESHOLD_REJECT:
            decision = Decision.REJECTED
        elif risk_score >= _THRESHOLD_FLAG:
            decision = Decision.FLAGGED
        else:
            decision = Decision.APPROVED

        return ScoredTransaction(
            txn_id=features.txn_id,
            risk_score=round(risk_score, 4),
            decision=decision,
            shap_values=shap_map,
            model_version=_MODEL_VERSION,
            scored_at=datetime.utcnow(),
        )
