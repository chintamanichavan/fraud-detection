from __future__ import annotations
import math
from datetime import datetime, timezone
import redis.asyncio as aioredis
from ..models.transaction import Transaction, FeatureVector

_EARTH_R_KM = 6371.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_R_KM * math.asin(math.sqrt(a))


async def engineer(txn: Transaction, redis: aioredis.Redis) -> FeatureVector:
    card = txn.card_hash
    now  = txn.timestamp.timestamp()

    # velocity windows — stored as sorted sets of timestamps
    pipe = redis.pipeline(transaction=False)
    pipe.zcount(f"vel:{card}", now - 3600, now)           # 1-hour window
    pipe.zcount(f"vel:{card}", now - 86400, now)          # 24-hour window
    pipe.get(f"home_lat:{card}")
    pipe.get(f"home_lon:{card}")
    pipe.get(f"last_txn_ts:{card}")
    pipe.zscore(f"merchants:{card}", txn.merchant_id)     # None if new merchant
    results = await pipe.execute()

    vel_1h, vel_24h, home_lat_raw, home_lon_raw, last_ts_raw, merchant_score = results

    home_lat = float(home_lat_raw) if home_lat_raw else txn.latitude
    home_lon = float(home_lon_raw) if home_lon_raw else txn.longitude
    geo_delta = _haversine(txn.latitude, txn.longitude, home_lat, home_lon)

    last_ts = float(last_ts_raw) if last_ts_raw else now
    days_since = (now - last_ts) / 86400

    # persist new event to velocity set (TTL: 25h to cover the 24h window with buffer)
    await redis.zadd(f"vel:{card}", {str(now): now})
    await redis.expire(f"vel:{card}", 90000)
    await redis.set(f"last_txn_ts:{card}", now, ex=90000)

    # amount z-score from rolling stats (simplified — use a proper feature store in prod)
    mean_raw = await redis.get(f"amount_mean:{card}")
    std_raw  = await redis.get(f"amount_std:{card}")
    mean = float(mean_raw) if mean_raw else txn.amount_usd
    std  = max(float(std_raw) if std_raw else 1.0, 1.0)
    amount_zscore = (txn.amount_usd - mean) / std

    return FeatureVector(
        txn_id=txn.id,
        velocity_1h=int(vel_1h or 0),
        velocity_24h=int(vel_24h or 0),
        geo_delta_km=round(geo_delta, 2),
        merchant_risk_score=float(merchant_score) if merchant_score else 0.5,
        amount_zscore=round(amount_zscore, 4),
        hour_of_day=txn.timestamp.astimezone(timezone.utc).hour,
        days_since_last_txn=round(days_since, 4),
        is_new_merchant=merchant_score is None,
    )
