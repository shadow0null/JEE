"""
topic_matcher.py
=================
Fuzzy JEE chapter/topic matching using RapidFuzz.

Used to map messy, misspelled or loosely-phrased input (from students,
OCR'd PDFs, or the planner) onto the canonical JEE syllabus topic
list, e.g.:

    "electro statics"   -> "Electrostatics"
    "thermo dynamics"   -> "Thermodynamics"
    "probablity"        -> "Probability"
    "sequence series"   -> "Sequence and Series"

This is intentionally NOT an LLM call - it is a deterministic,
offline string-similarity lookup, used specifically so simple topic
matching never touches Gemini quota.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .safety import validate_input, MAX_TOPIC_QUERY_LENGTH, SafetyError

try:
    from rapidfuzz import fuzz, process

    RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when RapidFuzz is absent
    fuzz = None
    process = None
    RAPIDFUZZ_AVAILABLE = False


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


# Small, easily-editable canonical JEE syllabus topic list.
# Extend by simply appending to these lists - no other code changes needed.
JEE_TOPICS: Dict[str, List[str]] = {
    "Physics": [
        "Basic Mathematics", "Units and Dimensions", "Vectors", "Kinematics",
        "Newton's Laws of Motion", "Friction", "Work Energy Power",
        "Centre of Mass", "Rotational Motion", "Gravitation",
        "Mechanical Properties of Solids", "Mechanical Properties of Fluids",
        "Thermal Properties of Matter", "Thermodynamics",
        "Kinetic Theory of Gases", "Oscillations", "Waves", "Electrostatics",
        "Capacitance", "Current Electricity", "Magnetic Effects of Current",
        "Magnetism and Matter", "Electromagnetic Induction",
        "Alternating Current", "Electromagnetic Waves", "Ray Optics",
        "Wave Optics", "Dual Nature of Matter and Radiation", "Atoms",
        "Nuclei", "Semiconductor Electronics",
    ],
    "Chemistry": [
        "Some Basic Concepts of Chemistry", "Atomic Structure",
        "Chemical Bonding", "States of Matter", "Thermodynamics",
        "Equilibrium", "Redox Reactions", "Hydrogen",
        "s-Block Elements", "p-Block Elements", "d and f Block Elements",
        "Coordination Compounds", "Organic Chemistry Basics",
        "Hydrocarbons", "Haloalkanes and Haloarenes", "Alcohols Phenols Ethers",
        "Aldehydes Ketones Carboxylic Acids", "Amines", "Biomolecules",
        "Polymers", "Chemistry in Everyday Life", "Electrochemistry",
        "Chemical Kinetics", "Surface Chemistry", "Solutions",
    ],
    "Mathematics": [
        "Sets Relations and Functions", "Complex Numbers",
        "Quadratic Equations", "Sequence and Series", "Permutations and Combinations",
        "Binomial Theorem", "Matrices", "Determinants", "Probability",
        "Trigonometry", "Straight Lines", "Circles", "Conic Sections",
        "Three Dimensional Geometry", "Vector Algebra", "Limits and Continuity",
        "Differentiability", "Application of Derivatives", "Indefinite Integrals",
        "Definite Integrals", "Differential Equations", "Statistics",
    ],
}


def all_topics() -> List[str]:
    seen: List[str] = []
    for topics in JEE_TOPICS.values():
        for t in topics:
            if t not in seen:
                seen.append(t)
    return seen


def match_topic(query: str, top_n: int = 3, score_cutoff: float = 60.0) -> Dict[str, Any]:
    """Return the best-matching canonical topic(s) for a fuzzy query
    string, e.g. 'electro statics' -> 'Electrostatics'."""
    if not RAPIDFUZZ_AVAILABLE:
        return _fail("RapidFuzz is not installed. Run: pip install -r requirements.txt")
    try:
        if not isinstance(query, str) or not query.strip():
            raise SafetyError("Empty topic query.")
        if len(query) > MAX_TOPIC_QUERY_LENGTH:
            raise SafetyError(f"Topic query too long (max {MAX_TOPIC_QUERY_LENGTH} characters).")
        top_n = max(1, min(int(top_n), 10))
        score_cutoff = max(0.0, min(float(score_cutoff), 100.0))

        candidates = all_topics()
        results = process.extract(
            query, candidates, scorer=fuzz.WRatio, limit=top_n, score_cutoff=score_cutoff,
        )

        matches = [{"topic": name, "score": round(score, 1)} for name, score, _ in results]
        best = matches[0]["topic"] if matches else None

        return _ok(query=query, best_match=best, matches=matches)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely match this topic. ({e})")


def classify_subject(topic_name: str) -> Optional[str]:
    """Given a canonical (or close) topic name, return which subject
    (Physics/Chemistry/Mathematics) it belongs to, or None."""
    for subject, topics in JEE_TOPICS.items():
        if topic_name in topics:
            return subject
    match = match_topic(topic_name, top_n=1, score_cutoff=80.0)
    if match.get("success") and match.get("best_match"):
        for subject, topics in JEE_TOPICS.items():
            if match["best_match"] in topics:
                return subject
    return None
