"""
router.py
=========
Classifies a raw natural-language style question into one of:
    MATH | NUMERICAL | PHYSICS | UNIT | GRAPH | UNKNOWN

This is a lightweight, deterministic, keyword/pattern based classifier -
NOT a machine-learning model and NOT an LLM call. It never contacts
Gemini or any network service.
"""

from __future__ import annotations

import re
from typing import Dict

CATEGORY_MATH = "MATH"
CATEGORY_NUMERICAL = "NUMERICAL"
CATEGORY_PHYSICS = "PHYSICS"
CATEGORY_UNIT = "UNIT"
CATEGORY_GRAPH = "GRAPH"
CATEGORY_UNKNOWN = "UNKNOWN"

# Ordered: first matching category wins.
_GRAPH_PATTERNS = [
    r"\bplot\b", r"\bgraph\b", r"\bdraw\b", r"\btrajectory\b",
    r"\bposition[- ]time\b", r"\bvelocity[- ]time\b", r"\bacceleration[- ]time\b",
]

_UNIT_PATTERNS = [
    r"\bconvert\b", r"\bto\s+(m/s|km/h|kg|joule|newton|watt|ohm|volt|m|cm|mm|s|hr|hour)\b",
    r"->|→", r"\bin\s+(si|cgs)\s+units\b",
]

_PHYSICS_KEYWORDS = [
    "force", "velocity", "acceleration", "displacement", "projectile",
    "work done", "power", "kinetic energy", "potential energy", "momentum",
    "ohm", "resistance", "resistor", "voltage", "current", "charge",
    "frequency", "period", "angular frequency", "shm", "simple harmonic",
    "gravitational", "escape velocity", "newton's second law",
]

_NUMERICAL_KEYWORDS = [
    "dot product", "cross product", "vector", "matrix", "numerically",
    "root of", "interpolate", "interpolation", "optimi", "numerical integration",
    "using scipy", "using numpy",
]

_MATH_KEYWORDS = [
    "solve", "differentiate", "derivative", "integrate", "integral",
    "limit", "factor", "expand", "simplify", "determinant", "inverse of",
    "d/dx", "matrix",
]


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify(question: str) -> Dict[str, str]:
    """Classify a raw question string. Always returns a dict with at
    least {'type': ..., 'raw': question}. Never raises."""
    if not isinstance(question, str) or not question.strip():
        return {"type": CATEGORY_UNKNOWN, "raw": question, "reason": "Empty input."}

    text = question.strip()
    lower = text.lower()

    # 1. Explicit unit-conversion phrasing takes priority (e.g. "72 km/h to m/s")
    if _matches_any(_UNIT_PATTERNS, lower) or re.search(r"\d+\s*[a-zA-Z/]+\s*(to|→|->)\s*[a-zA-Z/]+", lower):
        return {"type": CATEGORY_UNIT, "raw": text}

    # 2. Graph / plot requests
    if _matches_any(_GRAPH_PATTERNS, lower):
        return {"type": CATEGORY_GRAPH, "raw": text}

    # 3. Physics formula requests (explicit physics vocabulary + a numeric value)
    if any(kw in lower for kw in _PHYSICS_KEYWORDS):
        return {"type": CATEGORY_PHYSICS, "raw": text}

    # 4. Numerical (NumPy/SciPy) requests: vectors, numeric root finding, etc.
    if any(kw in lower for kw in _NUMERICAL_KEYWORDS):
        return {"type": CATEGORY_NUMERICAL, "raw": text}

    # 5. Symbolic math (SymPy): solve / differentiate / integrate / etc.
    if any(kw in lower for kw in _MATH_KEYWORDS):
        return {"type": CATEGORY_MATH, "raw": text}

    # 6. Bare equation heuristic: contains '=' and algebraic symbols -> MATH
    if "=" in text and re.search(r"[a-zA-Z]\s*[\^\*]|\d[a-zA-Z]", text):
        return {"type": CATEGORY_MATH, "raw": text}

    return {
        "type": CATEGORY_UNKNOWN,
        "raw": text,
        "reason": "Could not classify this question into a supported category.",
    }
