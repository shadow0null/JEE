"""
topic_graph.py
===============
Optional JEE topic dependency graph, built on NetworkX.

Example chain:

    Basic Mathematics -> Vectors -> Kinematics -> Newton's Laws of Motion
        -> Work Energy Power -> Centre of Mass

The graph is a small, deliberately incomplete, easily-editable seed
dataset (see _PREREQUISITE_EDGES below) - NOT hundreds of hard-coded
relationships. Extend it by adding (prerequisite, topic) tuples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when NetworkX is absent
    nx = None
    NETWORKX_AVAILABLE = False


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


# Seed data: (prerequisite, topic). Small and intentionally editable -
# add rows here to grow the graph rather than hard-coding hundreds of
# questionable relationships up front.
_PREREQUISITE_EDGES: List[tuple] = [
    ("Basic Mathematics", "Vectors"),
    ("Vectors", "Kinematics"),
    ("Kinematics", "Newton's Laws of Motion"),
    ("Newton's Laws of Motion", "Friction"),
    ("Newton's Laws of Motion", "Work Energy Power"),
    ("Work Energy Power", "Centre of Mass"),
    ("Centre of Mass", "Rotational Motion"),
    ("Newton's Laws of Motion", "Gravitation"),
    ("Basic Mathematics", "Trigonometry"),
    ("Trigonometry", "Oscillations"),
    ("Oscillations", "Waves"),
    ("Basic Mathematics", "Limits and Continuity"),
    ("Limits and Continuity", "Differentiability"),
    ("Differentiability", "Application of Derivatives"),
    ("Application of Derivatives", "Indefinite Integrals"),
    ("Indefinite Integrals", "Definite Integrals"),
    ("Definite Integrals", "Differential Equations"),
    ("Quadratic Equations", "Complex Numbers"),
    ("Sets Relations and Functions", "Limits and Continuity"),
    ("Vectors", "Electrostatics"),
    ("Electrostatics", "Capacitance"),
    ("Electrostatics", "Current Electricity"),
    ("Current Electricity", "Magnetic Effects of Current"),
    ("Magnetic Effects of Current", "Electromagnetic Induction"),
    ("Electromagnetic Induction", "Alternating Current"),
    ("Atomic Structure", "Chemical Bonding"),
    ("Chemical Bonding", "Coordination Compounds"),
    ("Some Basic Concepts of Chemistry", "Redox Reactions"),
    ("Redox Reactions", "Electrochemistry"),
    ("States of Matter", "Thermodynamics"),
    ("Thermodynamics", "Equilibrium"),
    ("Equilibrium", "Chemical Kinetics"),
]


def _build_graph() -> "nx.DiGraph":
    g = nx.DiGraph()
    g.add_edges_from(_PREREQUISITE_EDGES)
    return g


_GRAPH = _build_graph() if NETWORKX_AVAILABLE else None


def _require_networkx() -> Optional[dict]:
    if not NETWORKX_AVAILABLE:
        return _fail("NetworkX is not installed. Run: pip install -r requirements.txt")
    return None


def list_topics() -> Dict[str, Any]:
    if (err := _require_networkx()) is not None:
        return err
    return _ok(topics=sorted(_GRAPH.nodes))


def add_topic(name: str) -> Dict[str, Any]:
    if (err := _require_networkx()) is not None:
        return err
    if not isinstance(name, str) or not name.strip():
        return _fail("Topic name must be a non-empty string.")
    _GRAPH.add_node(name.strip())
    return _ok(topic=name.strip())


def add_prerequisite(prerequisite: str, topic: str) -> Dict[str, Any]:
    if (err := _require_networkx()) is not None:
        return err
    if not prerequisite or not topic:
        return _fail("Both 'prerequisite' and 'topic' are required.")
    _GRAPH.add_edge(prerequisite.strip(), topic.strip())
    if not nx.is_directed_acyclic_graph(_GRAPH):
        _GRAPH.remove_edge(prerequisite.strip(), topic.strip())
        return _fail("Adding this relationship would create a cycle; rejected.")
    return _ok(prerequisite=prerequisite.strip(), topic=topic.strip())


def get_prerequisites(topic: str) -> Dict[str, Any]:
    """All topics that must (transitively) come before `topic`."""
    if (err := _require_networkx()) is not None:
        return err
    if topic not in _GRAPH:
        return _fail(f"Unknown topic '{topic}'.")
    ancestors = nx.ancestors(_GRAPH, topic)
    return _ok(topic=topic, prerequisites=sorted(ancestors))


def get_dependents(topic: str) -> Dict[str, Any]:
    """All topics that (transitively) depend on `topic`."""
    if (err := _require_networkx()) is not None:
        return err
    if topic not in _GRAPH:
        return _fail(f"Unknown topic '{topic}'.")
    descendants = nx.descendants(_GRAPH, topic)
    return _ok(topic=topic, dependents=sorted(descendants))


def study_path(topic: str) -> Dict[str, Any]:
    """A suggested study order (topological) covering `topic` and every
    one of its prerequisites."""
    if (err := _require_networkx()) is not None:
        return err
    if topic not in _GRAPH:
        return _fail(f"Unknown topic '{topic}'.")
    ancestors = nx.ancestors(_GRAPH, topic) | {topic}
    subgraph = _GRAPH.subgraph(ancestors)
    try:
        path = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        return _fail("Topic graph contains a cycle; cannot compute a study path.")
    return _ok(topic=topic, study_path=path)
