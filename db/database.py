import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool

from constants import MACHINES

load_dotenv()

SQL_DIR = Path(__file__).parent.parent / "sql"

_SQL_INSERT  = (SQL_DIR / "insert_prediction.sql").read_text()
_SQL_HISTORY = (SQL_DIR / "get_history.sql").read_text()
_SQL_STATUS  = (SQL_DIR / "get_machine_status.sql").read_text()

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 10, dsn=os.environ["SUPABASE_URL"])
    return _pool


@contextmanager
def _conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _serialize(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def save_prediction(
    machine_id: str,
    filename: str,
    features: dict,
    predicted_label: str,
    probability_faulty: float,
    probability_healthy: float,
) -> dict:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SQL_INSERT, {
                "machine_id": machine_id,
                "filename": filename,
                "features": json.dumps(features),
                "predicted_label": predicted_label,
                "probability_faulty": probability_faulty,
                "probability_healthy": probability_healthy,
            })
            row = cur.fetchone()
    return _serialize(dict(row)) if row else {}


def get_machine_history(machine_id: str, limit: int = 100) -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SQL_HISTORY, {"machine_id": machine_id, "limit": limit})
            rows = cur.fetchall()
    return [_serialize(dict(r)) for r in rows]


def _consecutive_bad(predictions: list[str]) -> int:
    count = 0
    for p in predictions:
        if p == "f":
            count += 1
        else:
            break
    return count


def _status_label(n: int) -> str:
    if n >= 5:
        return "red"
    if n >= 3:
        return "yellow"
    return "green"


def get_all_machines_status() -> dict:
    status = {}
    with _conn() as conn:
        for machine_id in MACHINES:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(_SQL_STATUS, {"machine_id": machine_id})
                rows = cur.fetchall()
            preds = [r["predicted_label"] for r in rows]
            consecutive = _consecutive_bad(preds)
            status[machine_id] = {
                "status": _status_label(consecutive),
                "consecutive_bad": consecutive,
                "total_checked": len(preds),
                "last_prediction": preds[0] if preds else None,
            }
    return status
