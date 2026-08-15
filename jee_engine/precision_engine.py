"""
precision_engine.py
====================
High-precision numerical calculations using mpmath.

This module is used only when standard double-precision floats are
insufficient - e.g. many correct digits of an irrational constant, or
a root-finding problem that is numerically sensitive. It is NOT a
replacement for math_engine/numerical_engine and should not be used
for everyday arithmetic.

Every expression string is routed through the shared safety module
before it is ever touched by mpmath, exactly like every other engine
in this project. No eval()/exec(), no dunder access, no filesystem or
network access.
"""

from __future__ import annotations

from typing import Any, Dict

import mpmath
import sympy

from .safety import safe_parse_expr, validate_input, SafetyError, time_limit

MAX_PRECISION_DIGITS = 500  # hard ceiling to prevent memory/CPU abuse


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _clamp_precision(precision: int) -> int:
    try:
        precision = int(precision)
    except (TypeError, ValueError):
        return 50
    return max(5, min(precision, MAX_PRECISION_DIGITS))


def evaluate_high_precision(expr_str: str, precision: int = 50) -> Dict[str, Any]:
    """Evaluate a numeric expression (no free variables) to `precision`
    significant decimal digits using mpmath."""
    try:
        precision = _clamp_precision(precision)
        expr = safe_parse_expr(expr_str)

        free_symbols = getattr(expr, "free_symbols", set())
        if free_symbols:
            return _fail(
                f"Expression must be fully numeric for high-precision evaluation "
                f"(found free symbol(s): {', '.join(sorted(str(s) for s in free_symbols))})."
            )

        with mpmath.workdps(precision):
            with time_limit():
                value = mpmath.mpf(sympy.N(expr, precision))

        return _ok(result=mpmath.nstr(value, precision), precision=precision)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely evaluate this expression at high precision. ({e})")


def high_precision_constant(name: str, precision: int = 50) -> Dict[str, Any]:
    """Return a well-known mathematical constant to `precision` digits."""
    constants = {
        "pi": mpmath.pi,
        "e": mpmath.e,
        "phi": mpmath.phi,
        "sqrt2": lambda: mpmath.sqrt(2),
        "euler": mpmath.euler,
    }
    key = str(name).strip().lower()
    if key not in constants:
        return _fail(f"Unknown constant '{name}'. Supported: {list(constants)}.")

    try:
        precision = _clamp_precision(precision)
        with mpmath.workdps(precision):
            value = constants[key]() if callable(constants[key]) else constants[key]
            value = mpmath.mpf(value)
        return _ok(constant=key, result=mpmath.nstr(value, precision), precision=precision)
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute this constant safely. ({e})")


def high_precision_root(expr_str: str, var: str = "x", initial_guess: float = 1.0,
                         precision: int = 50) -> Dict[str, Any]:
    """Find a root of a single-variable expression using mpmath's
    high-precision Newton/secant solver."""
    try:
        precision = _clamp_precision(precision)
        validate_input(expr_str)
        v = sympy.Symbol(var)
        expr = safe_parse_expr(expr_str, {var: v})

        if v not in expr.free_symbols:
            return _fail(f"Expression does not contain variable '{var}'.")

        f = sympy.lambdify(v, expr, modules=["mpmath"])

        with mpmath.workdps(precision):
            with time_limit():
                root = mpmath.findroot(f, mpmath.mpf(initial_guess))

        return _ok(result=mpmath.nstr(root, precision), precision=precision)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely find a root at high precision. ({e})")
