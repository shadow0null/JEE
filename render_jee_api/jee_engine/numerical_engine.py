"""
numerical_engine.py
====================
Numerical computation using NumPy and SciPy only.

Handles vectors, numeric matrices, root finding, numerical integration,
interpolation and optimization. All expression strings still go through
the shared safety module before being turned into callables.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import sympy
from scipy import integrate as sp_integrate
from scipy import optimize as sp_optimize
from scipy import interpolate as sp_interpolate

from .safety import safe_parse_expr, validate_input, SafetyError, time_limit, MAX_MATRIX_DIM

X = sympy.Symbol("x")


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _to_lambda(expr_str: str, var: str = "x"):
    v = sympy.Symbol(var)
    expr = safe_parse_expr(expr_str, {var: v})
    return sympy.lambdify(v, expr, modules=["numpy"])


# --------------------------------------------------------------------------- #
# Vectors (NumPy)
# --------------------------------------------------------------------------- #

def _validate_vector(vec: Sequence[float], name: str = "vector") -> np.ndarray:
    if not isinstance(vec, (list, tuple)) or len(vec) == 0:
        raise SafetyError(f"{name} must be a non-empty list of numbers.")
    if len(vec) > 10:
        raise SafetyError(f"{name} too large (max 10 components).")
    for val in vec:
        if not isinstance(val, (int, float)):
            raise SafetyError(f"{name} entries must be numbers.")
    return np.array(vec, dtype=float)


def vector_add(a: Sequence[float], b: Sequence[float]) -> dict:
    try:
        va, vb = _validate_vector(a, "A"), _validate_vector(b, "B")
        if va.shape != vb.shape:
            raise SafetyError("Vectors must be the same length to add.")
        result = va + vb
        return _ok(result=result.tolist())
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to add vectors safely. ({e})")


def vector_dot(a: Sequence[float], b: Sequence[float]) -> dict:
    try:
        va, vb = _validate_vector(a, "A"), _validate_vector(b, "B")
        if va.shape != vb.shape:
            raise SafetyError("Vectors must be the same length for a dot product.")
        result = float(np.dot(va, vb))
        return _ok(result=result)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute dot product safely. ({e})")


def vector_cross(a: Sequence[float], b: Sequence[float]) -> dict:
    try:
        va, vb = _validate_vector(a, "A"), _validate_vector(b, "B")
        if va.shape[0] not in (2, 3) or vb.shape[0] not in (2, 3):
            raise SafetyError("Cross product requires 2D or 3D vectors.")
        result = np.cross(va, vb)
        return _ok(result=result.tolist())
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute cross product safely. ({e})")


def vector_magnitude(a: Sequence[float]) -> dict:
    try:
        va = _validate_vector(a, "A")
        return _ok(result=float(np.linalg.norm(va)))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute magnitude safely. ({e})")


# --------------------------------------------------------------------------- #
# Numeric matrices (NumPy)
# --------------------------------------------------------------------------- #

def _validate_np_matrix(rows: List[List[float]]) -> np.ndarray:
    if not rows or not all(isinstance(r, list) for r in rows):
        raise SafetyError("Matrix must be a non-empty list of rows.")
    n_cols = len(rows[0])
    if len(rows) > MAX_MATRIX_DIM or n_cols > MAX_MATRIX_DIM:
        raise SafetyError(f"Matrix too large (max {MAX_MATRIX_DIM}x{MAX_MATRIX_DIM}).")
    if any(len(r) != n_cols for r in rows):
        raise SafetyError("All matrix rows must have the same length.")
    return np.array(rows, dtype=float)


def numeric_matrix_multiply(rows_a: List[List[float]], rows_b: List[List[float]]) -> dict:
    try:
        a, b = _validate_np_matrix(rows_a), _validate_np_matrix(rows_b)
        if a.shape[1] != b.shape[0]:
            raise SafetyError("Matrix dimensions are not compatible for multiplication.")
        result = a @ b
        return _ok(result=result.tolist())
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to multiply matrices safely. ({e})")


def numeric_matrix_determinant(rows: List[List[float]]) -> dict:
    try:
        a = _validate_np_matrix(rows)
        if a.shape[0] != a.shape[1]:
            raise SafetyError("Determinant requires a square matrix.")
        return _ok(result=float(np.linalg.det(a)))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute determinant safely. ({e})")


# --------------------------------------------------------------------------- #
# SciPy: root finding
# --------------------------------------------------------------------------- #

def find_root(expr_str: str, var: str = "x", bracket: Sequence[float] | None = None,
               initial_guess: float | None = None) -> dict:
    """Find a numerical root of expr(var) = 0 using SciPy.
    Prefers a bracketing bisection (brentq) if a [low, high] bracket is
    given, otherwise falls back to Newton's method from initial_guess."""
    try:
        f = _to_lambda(expr_str, var)
        with time_limit():
            if bracket is not None:
                lo, hi = float(bracket[0]), float(bracket[1])
                root = sp_optimize.brentq(f, lo, hi)
            else:
                x0 = float(initial_guess) if initial_guess is not None else 1.0
                root = sp_optimize.newton(f, x0)
        return _ok(result=float(root))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to find a root safely. ({e})")


# --------------------------------------------------------------------------- #
# SciPy: numerical integration
# --------------------------------------------------------------------------- #

def numerical_integrate(expr_str: str, var: str, lower: float, upper: float) -> dict:
    try:
        lower, upper = float(lower), float(upper)
        if abs(lower) > 1e6 or abs(upper) > 1e6:
            raise SafetyError("Integration bounds exceed the allowed limit.")
        f = _to_lambda(expr_str, var)
        with time_limit():
            value, err_estimate = sp_integrate.quad(f, lower, upper)
        return _ok(result=float(value), error_estimate=float(err_estimate))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to integrate numerically safely. ({e})")


# --------------------------------------------------------------------------- #
# SciPy: interpolation
# --------------------------------------------------------------------------- #

def interpolate_points(x_points: Sequence[float], y_points: Sequence[float],
                        query_x: float, kind: str = "linear") -> dict:
    try:
        if kind not in ("linear", "quadratic", "cubic", "nearest"):
            raise SafetyError("Unsupported interpolation kind.")
        xs = _validate_vector(x_points, "x_points")
        ys = _validate_vector(y_points, "y_points")
        if xs.shape != ys.shape:
            raise SafetyError("x_points and y_points must be the same length.")
        f = sp_interpolate.interp1d(xs, ys, kind=kind, fill_value="extrapolate")
        with time_limit():
            result = float(f(float(query_x)))
        return _ok(result=result)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to interpolate safely. ({e})")


# --------------------------------------------------------------------------- #
# SciPy: 1-D optimization (minimize a scalar expression)
# --------------------------------------------------------------------------- #

def minimize_expression(expr_str: str, var: str = "x", initial_guess: float = 0.0) -> dict:
    try:
        f = _to_lambda(expr_str, var)
        with time_limit():
            res = sp_optimize.minimize_scalar(f)
        if not res.success:
            return _fail("Optimizer did not converge.")
        return _ok(result_x=float(res.x), result_value=float(res.fun))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to optimize safely. ({e})")
