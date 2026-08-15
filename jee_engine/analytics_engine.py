"""
analytics_engine.py
=====================
Optional local student-performance analytics, built on scikit-learn.

This module is deliberately conservative: it only produces a result
when there is enough data to support one, and never fabricates a
prediction. If there isn't enough data, every function here returns
"Insufficient data for prediction." instead of guessing.

Kept independent from the rest of the engine so it can be swapped out
or improved (better features, a real model, persistence) later without
touching math/physics/units/graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .safety import MAX_ANALYTICS_RECORDS, SafetyError

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when scikit-learn is absent
    np = None
    LinearRegression = None
    SKLEARN_AVAILABLE = False

MIN_POINTS_FOR_TREND = 3
INSUFFICIENT_DATA_MSG = "Insufficient data for prediction."


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _require_sklearn() -> Optional[dict]:
    if not SKLEARN_AVAILABLE:
        return _fail("scikit-learn is not installed. Run: pip install -r requirements.txt")
    return None


def _validate_records(records: List[Dict[str, Any]]) -> None:
    if not isinstance(records, list):
        raise SafetyError("records must be a list.")
    if len(records) > MAX_ANALYTICS_RECORDS:
        raise SafetyError(f"Too many records (max {MAX_ANALYTICS_RECORDS}).")


def topic_performance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """records: [{"topic": str, "correct": bool, ...}, ...]
    Returns a simple, honest accuracy-based weak-topic score per topic -
    no model, just arithmetic, so it never needs 'insufficient data'."""
    try:
        _validate_records(records)
        if not records:
            return _ok(topics={}, note=INSUFFICIENT_DATA_MSG)

        per_topic: Dict[str, List[bool]] = {}
        for r in records:
            topic = str(r.get("topic", "")).strip()
            if not topic:
                continue
            per_topic.setdefault(topic, []).append(bool(r.get("correct")))

        result = {}
        for topic, outcomes in per_topic.items():
            attempts = len(outcomes)
            accuracy = sum(outcomes) / attempts if attempts else 0.0
            result[topic] = {
                "attempts": attempts,
                "accuracy": round(accuracy, 3),
                "weak_topic_score": round(1 - accuracy, 3),  # higher = weaker
            }
        return _ok(topics=result)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely analyze this data. ({e})")


def improvement_trend(scores_over_time: List[float]) -> Dict[str, Any]:
    """A basic linear-regression trend over a sequence of scores
    (e.g. weekly test percentages). Requires at least
    MIN_POINTS_FOR_TREND data points or it honestly reports that
    there isn't enough data - it does not invent a trend."""
    if (err := _require_sklearn()) is not None:
        return err
    try:
        if not isinstance(scores_over_time, list):
            raise SafetyError("scores_over_time must be a list of numbers.")
        if len(scores_over_time) > MAX_ANALYTICS_RECORDS:
            raise SafetyError(f"Too many data points (max {MAX_ANALYTICS_RECORDS}).")

        scores = [float(s) for s in scores_over_time]
        if len(scores) < MIN_POINTS_FOR_TREND:
            return _ok(trend=None, slope=None, note=INSUFFICIENT_DATA_MSG)

        X = np.arange(len(scores)).reshape(-1, 1)
        y = np.array(scores)
        model = LinearRegression().fit(X, y)
        slope = float(model.coef_[0])

        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        return _ok(trend=trend, slope=round(slope, 4), points_used=len(scores))
    except SafetyError as e:
        return _fail(str(e))
    except (TypeError, ValueError) as e:
        return _fail(f"Invalid numeric data. ({e})")
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely compute a trend. ({e})")


def revision_priority(topics_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """topics_data: {topic: {"accuracy": float, "attempts": int,
    "days_since_last_revision": float}, ...}
    Ranks topics by a simple, transparent weighted score - not a
    black-box model - so results stay explainable for students."""
    try:
        if not isinstance(topics_data, dict):
            raise SafetyError("topics_data must be a dict of topic -> stats.")
        if len(topics_data) > MAX_ANALYTICS_RECORDS:
            raise SafetyError(f"Too many topics (max {MAX_ANALYTICS_RECORDS}).")
        if not topics_data:
            return _ok(priority=[], note=INSUFFICIENT_DATA_MSG)

        scored = []
        for topic, stats in topics_data.items():
            attempts = int(stats.get("attempts", 0))
            if attempts < 1:
                continue
            accuracy = float(stats.get("accuracy", 0.0))
            days_since = float(stats.get("days_since_last_revision", 0.0))
            # Higher priority = weaker accuracy + longer since last revision.
            priority_score = round((1 - accuracy) * 0.7 + min(days_since / 30, 1.0) * 0.3, 3)
            scored.append({"topic": topic, "priority_score": priority_score, "accuracy": accuracy})

        if not scored:
            return _ok(priority=[], note=INSUFFICIENT_DATA_MSG)

        scored.sort(key=lambda r: r["priority_score"], reverse=True)
        return _ok(priority=scored)
    except SafetyError as e:
        return _fail(str(e))
    except (TypeError, ValueError) as e:
        return _fail(f"Invalid numeric data. ({e})")
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely compute revision priority. ({e})")
