"""
math_engine.py
===============
Deterministic symbolic mathematics using SymPy only.

Every public function returns a plain dict of the shape:
    {"success": True,  "result": ..., "result_str": "...", ...}
    {"success": False, "error": "human readable message"}

so the router / main.py / tests never need to know SymPy internals.
"""

from __future__ import annotations

from typing import List, Sequence

import sympy
from sympy import Symbol, Matrix

from .safety import safe_parse_expr, validate_input, SafetyError, time_limit, MAX_MATRIX_DIM

X = Symbol("x")


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _var(name: str) -> Symbol:
    validate_input(name)
    if not name.isidentifier() or len(name) > 3:
        raise SafetyError("Invalid variable name.")
    return Symbol(name)


# --------------------------------------------------------------------------- #
# Equation solving (linear, quadratic, polynomial, simultaneous)
# --------------------------------------------------------------------------- #

def solve_equation(expr_str: str, var: str = "x") -> dict:
    """Solve a single equation such as '2*x**2 - 5*x - 3 = 0' or an
    expression implicitly set to zero such as 'x**2 - 4'."""
    try:
        v = _var(var)
        expr_str = validate_input(expr_str)
        if "=" in expr_str and not any(op in expr_str for op in ("==", "<=", ">=")):
            lhs_str, rhs_str = expr_str.split("=", 1)
            lhs = safe_parse_expr(lhs_str, {var: v})
            rhs = safe_parse_expr(rhs_str, {var: v})
            equation = sympy.Eq(lhs, rhs)
        else:
            lhs = safe_parse_expr(expr_str, {var: v})
            equation = sympy.Eq(lhs, 0)

        with time_limit():
            solutions = sympy.solve(equation, v)

        return _ok(
            result=[str(s) for s in solutions],
            result_str=", ".join(f"{var} = {s}" for s in solutions) or "No solution",
            latex=[sympy.latex(s) for s in solutions],
        )
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to solve this expression safely. ({e})")


def solve_system(equations: Sequence[str], variables: Sequence[str]) -> dict:
    """Solve a system of simultaneous equations, e.g.
    equations=['x + y = 5', 'x - y = 1'], variables=['x', 'y']."""
    try:
        if len(equations) > 6 or len(variables) > 6:
            raise SafetyError("Too many equations/variables (max 6).")

        syms = [_var(v) for v in variables]
        sym_map = dict(zip(variables, syms))

        eqs = []
        for eq_str in equations:
            eq_str = validate_input(eq_str)
            if "=" in eq_str:
                lhs_str, rhs_str = eq_str.split("=", 1)
                lhs = safe_parse_expr(lhs_str, sym_map)
                rhs = safe_parse_expr(rhs_str, sym_map)
                eqs.append(sympy.Eq(lhs, rhs))
            else:
                eqs.append(sympy.Eq(safe_parse_expr(eq_str, sym_map), 0))

        with time_limit():
            solution = sympy.solve(eqs, syms)

        if isinstance(solution, dict):
            result_str = ", ".join(f"{k} = {v}" for k, v in solution.items())
            result = {str(k): str(v) for k, v in solution.items()}
        else:
            result_str = str(solution)
            result = str(solution)

        return _ok(result=result, result_str=result_str)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to solve this system safely. ({e})")


# --------------------------------------------------------------------------- #
# Algebraic manipulation
# --------------------------------------------------------------------------- #

def factorize(expr_str: str) -> dict:
    try:
        expr = safe_parse_expr(expr_str)
        with time_limit():
            result = sympy.factor(expr)
        return _ok(result=str(result), latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to factor this expression safely. ({e})")


def expand_expression(expr_str: str) -> dict:
    try:
        expr = safe_parse_expr(expr_str)
        with time_limit():
            result = sympy.expand(expr)
        return _ok(result=str(result), latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to expand this expression safely. ({e})")


def simplify_expression(expr_str: str) -> dict:
    try:
        expr = safe_parse_expr(expr_str)
        with time_limit():
            result = sympy.simplify(expr)
        return _ok(result=str(result), latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to simplify this expression safely. ({e})")


# --------------------------------------------------------------------------- #
# Calculus
# --------------------------------------------------------------------------- #

def differentiate(expr_str: str, var: str = "x", order: int = 1) -> dict:
    try:
        if not (1 <= int(order) <= 6):
            raise SafetyError("Derivative order must be between 1 and 6.")
        v = _var(var)
        expr = safe_parse_expr(expr_str, {var: v})
        with time_limit():
            result = sympy.diff(expr, v, int(order))
        return _ok(result=str(result), latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to differentiate this expression safely. ({e})")


def integrate_indefinite(expr_str: str, var: str = "x") -> dict:
    try:
        v = _var(var)
        expr = safe_parse_expr(expr_str, {var: v})
        with time_limit():
            result = sympy.integrate(expr, v)
        return _ok(result=f"{result} + C", latex=f"{sympy.latex(result)} + C")
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to integrate this expression safely. ({e})")


def integrate_definite(expr_str: str, var: str, lower: str, upper: str) -> dict:
    try:
        v = _var(var)
        expr = safe_parse_expr(expr_str, {var: v})
        lo = safe_parse_expr(str(lower))
        hi = safe_parse_expr(str(upper))
        with time_limit():
            result = sympy.integrate(expr, (v, lo, hi))
        return _ok(result=str(result), numeric=float(result) if result.is_number else None,
                    latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to evaluate this definite integral safely. ({e})")


def compute_limit(expr_str: str, var: str, point: str, direction: str | None = None) -> dict:
    try:
        v = _var(var)
        expr = safe_parse_expr(expr_str, {var: v})
        pt = safe_parse_expr(str(point))
        kwargs = {}
        if direction in ("+", "-"):
            kwargs["dir"] = direction
        with time_limit():
            result = sympy.limit(expr, v, pt, **kwargs)
        return _ok(result=str(result), latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to evaluate this limit safely. ({e})")


# --------------------------------------------------------------------------- #
# Trigonometry
# --------------------------------------------------------------------------- #

def trig_simplify(expr_str: str) -> dict:
    try:
        expr = safe_parse_expr(expr_str)
        with time_limit():
            result = sympy.trigsimp(expr)
        return _ok(result=str(result), latex=sympy.latex(result))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to simplify this trig expression safely. ({e})")


def solve_trig_equation(expr_str: str, var: str = "x") -> dict:
    try:
        v = _var(var)
        expr_str = validate_input(expr_str)
        if "=" in expr_str:
            lhs_str, rhs_str = expr_str.split("=", 1)
            lhs = safe_parse_expr(lhs_str, {var: v})
            rhs = safe_parse_expr(rhs_str, {var: v})
            equation = sympy.Eq(lhs, rhs)
        else:
            equation = sympy.Eq(safe_parse_expr(expr_str, {var: v}), 0)
        with time_limit():
            solutions = sympy.solveset(equation, v, domain=sympy.S.Reals)
        return _ok(result=str(solutions))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to solve this trig equation safely. ({e})")


# --------------------------------------------------------------------------- #
# Matrices / determinants (symbolic, via SymPy Matrix)
# --------------------------------------------------------------------------- #

def _validate_matrix(rows: List[List[float]]) -> Matrix:
    if not rows or not all(isinstance(r, list) for r in rows):
        raise SafetyError("Matrix must be a non-empty list of rows.")
    n_rows = len(rows)
    n_cols = len(rows[0])
    if n_rows > MAX_MATRIX_DIM or n_cols > MAX_MATRIX_DIM:
        raise SafetyError(f"Matrix too large (max {MAX_MATRIX_DIM}x{MAX_MATRIX_DIM}).")
    if any(len(r) != n_cols for r in rows):
        raise SafetyError("All matrix rows must have the same length.")
    for row in rows:
        for val in row:
            if not isinstance(val, (int, float)):
                raise SafetyError("Matrix entries must be numbers.")
    return Matrix(rows)


def matrix_determinant(rows: List[List[float]]) -> dict:
    try:
        m = _validate_matrix(rows)
        if m.rows != m.cols:
            raise SafetyError("Determinant requires a square matrix.")
        with time_limit():
            det = m.det()
        return _ok(result=str(det), numeric=float(det))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute determinant safely. ({e})")


def matrix_inverse(rows: List[List[float]]) -> dict:
    try:
        m = _validate_matrix(rows)
        if m.rows != m.cols:
            raise SafetyError("Inverse requires a square matrix.")
        with time_limit():
            if m.det() == 0:
                return _fail("Matrix is singular; inverse does not exist.")
            inv = m.inv()
        return _ok(result=[[str(v) for v in row] for row in inv.tolist()])
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to compute inverse safely. ({e})")


def matrix_multiply(rows_a: List[List[float]], rows_b: List[List[float]]) -> dict:
    try:
        a = _validate_matrix(rows_a)
        b = _validate_matrix(rows_b)
        if a.cols != b.rows:
            raise SafetyError("Matrix dimensions are not compatible for multiplication.")
        with time_limit():
            result = a * b
        return _ok(result=[[str(v) for v in row] for row in result.tolist()])
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to multiply matrices safely. ({e})")
