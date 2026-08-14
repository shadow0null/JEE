#!/usr/bin/env python3
"""
StudyDesk Local JEE Engine - standalone entry point.

Usage:
    python main.py                 interactive CLI
    python main.py --json '{...}'  single JSON request/response, then exit
    python main.py --self-test     run a handful of built-in sanity checks

This file makes ZERO network calls and ZERO calls to Gemini or any AI
provider. It only imports the local jee_engine package.
"""

from __future__ import annotations

import sys
import json
import re

from jee_engine import router, math_engine, numerical_engine, physics_engine, units_engine, graph_engine


def _extract_numbers(text: str):
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", text)]


_LEADING_COMMAND_WORDS = re.compile(
    r"^\s*(?:please\s+)?(?:solve|simplify|factor(?:ize)?|expand|differentiate|integrate|find)\s+",
    re.IGNORECASE,
)


def _extract_equation(text: str) -> str:
    """Pull the equation/expression substring out of a natural language
    question such as 'solve x^2 - 5x + 6 = 0' -> 'x^2 - 5x + 6 = 0'."""
    text = _LEADING_COMMAND_WORDS.sub("", text.strip())
    match = re.search(r"([\d\w\.\+\-\*\/\^\(\)\s]+=\s*[\d\w\.\+\-\*\/\^\(\)\s]+)", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def handle_math(raw: str) -> dict:
    lower = raw.lower()
    expr = _extract_equation(raw)

    if "differentiate" in lower or "d/dx" in lower or "derivative" in lower:
        expr = re.sub(r"d/dx|differentiate", "", expr, flags=re.IGNORECASE).strip()
        expr = re.sub(r"^\(|\)$", "", expr).strip()
        return {"operation": "differentiate", **math_engine.differentiate(expr)}

    if "integrate" in lower or "integral" in lower or "∫" in raw:
        expr = re.sub(r"integrate|integral of|∫|dx", "", expr, flags=re.IGNORECASE).strip()
        return {"operation": "integrate", **math_engine.integrate_indefinite(expr)}

    if "factor" in lower:
        return {"operation": "factor", **math_engine.factorize(expr)}

    if "expand" in lower:
        return {"operation": "expand", **math_engine.expand_expression(expr)}

    if "simplify" in lower:
        return {"operation": "simplify", **math_engine.simplify_expression(expr)}

    if "limit" in lower:
        return {"operation": "limit", "error": "Please use JSON mode for limits (needs var/point)."}

    # default: treat as an equation to solve
    return {"operation": "solve", **math_engine.solve_equation(expr)}


def handle_unit(raw: str) -> dict:
    match = re.search(r"(.+?)\s*(?:to|→|->)\s*([a-zA-Z/°\^\d\s]+)$", raw.strip())
    if match:
        quantity, target = match.group(1).strip(), match.group(2).strip()
        return units_engine.convert(quantity, target)
    return units_engine.evaluate_dimensional_expression(raw)


def handle_physics(raw: str) -> dict:
    lower = raw.lower()
    nums = _extract_numbers(raw)

    if "force" in lower and len(nums) >= 2:
        return {"operation": "force", **physics_engine.force(mass=nums[0], accel=nums[1])}
    if "ohm" in lower or ("voltage" in lower and "current" in lower):
        return {"operation": "ohms_law", "error": "Please use JSON mode for Ohm's law (needs named fields)."}

    return {
        "success": False,
        "error": "Recognised as PHYSICS but could not extract enough structured "
                 "data from free text. Use JSON mode with an explicit 'operation'.",
    }


def handle_graph(raw: str) -> dict:
    match = re.search(r"y\s*=\s*(.+)", raw, re.IGNORECASE)
    expr = match.group(1).strip() if match else raw
    return graph_engine.plot_function(expr)


def process_text(raw: str) -> dict:
    classification = router.classify(raw)
    kind = classification["type"]

    if kind == router.CATEGORY_MATH:
        result = handle_math(raw)
    elif kind == router.CATEGORY_UNIT:
        result = handle_unit(raw)
    elif kind == router.CATEGORY_PHYSICS:
        result = handle_physics(raw)
    elif kind == router.CATEGORY_GRAPH:
        result = handle_graph(raw)
    elif kind == router.CATEGORY_NUMERICAL:
        result = {
            "success": False,
            "error": "Recognised as NUMERICAL. Use JSON mode with an explicit "
                     "'operation' (dot, cross, root, integrate, interpolate, minimize).",
        }
    else:
        result = {"success": False, "error": "Unable to solve this expression safely."}

    return {"type": kind, "query": raw, **result}


# --------------------------------------------------------------------------- #
# JSON request/response mode
# --------------------------------------------------------------------------- #

_MATH_OPS = {
    "solve": lambda p: math_engine.solve_equation(p["expression"], p.get("var", "x")),
    "solve_system": lambda p: math_engine.solve_system(p["equations"], p["variables"]),
    "factor": lambda p: math_engine.factorize(p["expression"]),
    "expand": lambda p: math_engine.expand_expression(p["expression"]),
    "simplify": lambda p: math_engine.simplify_expression(p["expression"]),
    "differentiate": lambda p: math_engine.differentiate(p["expression"], p.get("var", "x"), p.get("order", 1)),
    "integrate": lambda p: math_engine.integrate_indefinite(p["expression"], p.get("var", "x")),
    "integrate_definite": lambda p: math_engine.integrate_definite(
        p["expression"], p.get("var", "x"), p["lower"], p["upper"]),
    "limit": lambda p: math_engine.compute_limit(
        p["expression"], p.get("var", "x"), p["point"], p.get("direction")),
    "trig_simplify": lambda p: math_engine.trig_simplify(p["expression"]),
    "solve_trig": lambda p: math_engine.solve_trig_equation(p["expression"], p.get("var", "x")),
    "determinant": lambda p: math_engine.matrix_determinant(p["matrix"]),
    "inverse": lambda p: math_engine.matrix_inverse(p["matrix"]),
    "matrix_multiply": lambda p: math_engine.matrix_multiply(p["matrix_a"], p["matrix_b"]),
}

_NUMERICAL_OPS = {
    "vector_add": lambda p: numerical_engine.vector_add(p["a"], p["b"]),
    "dot": lambda p: numerical_engine.vector_dot(p["a"], p["b"]),
    "cross": lambda p: numerical_engine.vector_cross(p["a"], p["b"]),
    "magnitude": lambda p: numerical_engine.vector_magnitude(p["a"]),
    "matrix_multiply": lambda p: numerical_engine.numeric_matrix_multiply(p["matrix_a"], p["matrix_b"]),
    "determinant": lambda p: numerical_engine.numeric_matrix_determinant(p["matrix"]),
    "root": lambda p: numerical_engine.find_root(
        p["expression"], p.get("var", "x"), p.get("bracket"), p.get("initial_guess")),
    "integrate": lambda p: numerical_engine.numerical_integrate(
        p["expression"], p.get("var", "x"), p["lower"], p["upper"]),
    "interpolate": lambda p: numerical_engine.interpolate_points(
        p["x_points"], p["y_points"], p["query_x"], p.get("kind", "linear")),
    "minimize": lambda p: numerical_engine.minimize_expression(
        p["expression"], p.get("var", "x"), p.get("initial_guess", 0.0)),
}

_UNIT_OPS = {
    "convert": lambda p: units_engine.convert(p["quantity"], p["target_unit"]),
    "evaluate": lambda p: units_engine.evaluate_dimensional_expression(p["expression"]),
    "check_addable": lambda p: units_engine.check_addable(p["a"], p["b"]),
    "identify_derived_unit": lambda p: units_engine.identify_derived_unit(p["expression"]),
}

_PHYSICS_OPS = {
    "velocity": lambda p: physics_engine.velocity(**p),
    "acceleration": lambda p: physics_engine.acceleration(**p),
    "displacement": lambda p: physics_engine.displacement(**p),
    "projectile_motion": lambda p: physics_engine.projectile_motion(**p),
    "force": lambda p: physics_engine.force(**p),
    "work_done": lambda p: physics_engine.work_done(**p),
    "power": lambda p: physics_engine.power(**p),
    "kinetic_energy": lambda p: physics_engine.kinetic_energy(**p),
    "potential_energy": lambda p: physics_engine.potential_energy(**p),
    "momentum": lambda p: physics_engine.momentum(**p),
    "ohms_law": lambda p: physics_engine.ohms_law(**p),
    "electrical_power": lambda p: physics_engine.electrical_power(**p),
    "series_resistance": lambda p: physics_engine.series_resistance(**p),
    "parallel_resistance": lambda p: physics_engine.parallel_resistance(**p),
    "charge": lambda p: physics_engine.charge(**p),
    "frequency_from_period": lambda p: physics_engine.frequency_from_period(**p),
    "period_from_frequency": lambda p: physics_engine.period_from_frequency(**p),
    "angular_frequency": lambda p: physics_engine.angular_frequency(**p),
    "shm_displacement": lambda p: physics_engine.shm_displacement(**p),
    "shm_velocity": lambda p: physics_engine.shm_velocity(**p),
    "shm_max_acceleration": lambda p: physics_engine.shm_max_acceleration(**p),
    "gravitational_force": lambda p: physics_engine.gravitational_force(**p),
    "gravitational_potential_energy": lambda p: physics_engine.gravitational_potential_energy(**p),
    "escape_velocity": lambda p: physics_engine.escape_velocity(**p),
}

_GRAPH_OPS = {
    "plot": lambda p: graph_engine.plot_function(
        p["expression"], p.get("var", "x"), p.get("x_min", -10), p.get("x_max", 10),
        p.get("points", 500), p.get("title")),
    "projectile": lambda p: graph_engine.plot_projectile_trajectory(
        p["speed"], p["angle_deg"], p.get("gravity", 9.8)),
    "kinematics": lambda p: graph_engine.plot_kinematics(
        p["kind"], p.get("initial_v", 0.0), p.get("accel", 0.0), p.get("t_max", 10.0)),
    "shm": lambda p: graph_engine.plot_shm(
        p["amplitude"], p["omega"], p.get("phase", 0.0), p.get("cycles", 2.0)),
}

_DISPATCH = {
    "math": _MATH_OPS,
    "numerical": _NUMERICAL_OPS,
    "unit": _UNIT_OPS,
    "physics": _PHYSICS_OPS,
    "graph": _GRAPH_OPS,
}


def process_json(payload: dict) -> dict:
    """Process a structured JSON request of the shape:
        {"type": "math", "operation": "solve", "expression": "x^2-4=0"}
    Returns a structured JSON-serialisable dict. Never raises - all
    errors are caught and returned as {"success": False, "error": ...}.
    """
    try:
        req_type = str(payload.get("type", "")).lower()
        operation = payload.get("operation")

        ops_table = _DISPATCH.get(req_type)
        if ops_table is None:
            return {"success": False, "error": f"Unknown type '{req_type}'. "
                                                f"Expected one of: {list(_DISPATCH)}."}

        handler = ops_table.get(operation)
        if handler is None:
            return {"success": False, "error": f"Unknown operation '{operation}' for type '{req_type}'. "
                                                f"Supported: {list(ops_table)}."}

        params = {k: v for k, v in payload.items() if k not in ("type", "operation")}
        return handler(params)
    except KeyError as e:
        return {"success": False, "error": f"Missing required field: {e}"}
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments: {e}"}
    except Exception as e:  # noqa: BLE001 - never crash on user input
        return {"success": False, "error": f"Unable to process this request safely. ({e})"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def run_repl() -> None:
    print("=" * 60)
    print("  StudyDesk Local JEE Engine  (standalone, offline, no AI)")
    print("=" * 60)
    print("Type a math/physics/unit/graph question, or 'exit' to quit.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        response = process_text(raw)
        print(json.dumps(response, indent=2, default=str))
        print()


def main() -> None:
    args = sys.argv[1:]

    if not args:
        run_repl()
        return

    if args[0] == "--json":
        if len(args) < 2:
            print(json.dumps({"success": False, "error": "Provide a JSON string after --json."}))
            sys.exit(1)
        try:
            payload = json.loads(args[1])
        except json.JSONDecodeError as e:
            print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
            sys.exit(1)
        print(json.dumps(process_json(payload), indent=2, default=str))
        return

    if args[0] == "--self-test":
        _run_self_test()
        return

    # Anything else: treat the remaining args as a plain-text question.
    raw = " ".join(args)
    print(json.dumps(process_text(raw), indent=2, default=str))


def _run_self_test() -> None:
    checks = [
        ("solve 2x^2 - 5x - 3 = 0", process_text("solve 2x^2 - 5x - 3 = 0")),
        ("differentiate x^3", process_text("differentiate x^3")),
        ("72 km/h to m/s", process_text("72 km/h to m/s")),
        ("force with m=2 a=5", process_text("force m=2 a=5")),
    ]
    ok = True
    for label, result in checks:
        status = "PASS" if result.get("success", True) is not False else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {label} -> {json.dumps(result, default=str)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
