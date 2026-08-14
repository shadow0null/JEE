"""
units_engine.py
================
Unit conversion and dimensional analysis, built on Pint.

Pint is an optional-at-import-time dependency here: if it has not been
installed yet (see requirements.txt), every function below returns a
structured error instead of crashing the whole engine, so the rest of
the Local JEE Engine keeps working even before `pip install -r
requirements.txt` has been run.
"""

from __future__ import annotations

import re
from typing import Optional

from .safety import validate_input, SafetyError, MAX_INPUT_LENGTH

try:
    import pint

    _UREG = pint.UnitRegistry()
    _UREG.default_format = "~P"  # short, pretty unit formatting e.g. 'm/s'
    PINT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when pint is absent
    _UREG = None
    PINT_AVAILABLE = False


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _require_pint() -> Optional[dict]:
    if not PINT_AVAILABLE:
        return _fail(
            "Pint is not installed. Run 'pip install -r requirements.txt' "
            "to enable unit conversion and dimensional analysis."
        )
    return None


# A conservative whitelist of characters allowed in a quantity/unit string,
# e.g. "72 km/h", "5 eV", "2 kg * 3 m/s**2". Kept intentionally narrow -
# no double underscores, no parentheses-heavy expressions that could be
# used to smuggle attribute access into Pint's parser.
_UNIT_EXPR_RE = re.compile(r"^[0-9a-zA-Z\s\.\+\-\*\/\^\%°]*$")


def _validate_unit_string(s: str) -> str:
    s = validate_input(s)
    if not _UNIT_EXPR_RE.match(s):
        raise SafetyError("Unit expression contains disallowed characters.")
    return s.replace("^", "**")


def convert(quantity_str: str, target_unit: str) -> dict:
    """Convert a quantity string like '72 km/h' to a target unit like 'm/s'."""
    err = _require_pint()
    if err:
        return err
    try:
        quantity_str = _validate_unit_string(quantity_str)
        target_unit = _validate_unit_string(target_unit)
        qty = _UREG.Quantity(quantity_str)
        converted = qty.to(target_unit)
        return _ok(
            result=round(float(converted.magnitude), 10),
            unit=f"{converted.units:~P}",
            result_str=f"{converted:~P.6g}",
        )
    except SafetyError as e:
        return _fail(str(e))
    except pint.errors.DimensionalityError as e:  # noqa: F821
        return _fail(f"Invalid conversion: incompatible dimensions. ({e})")
    except pint.errors.UndefinedUnitError as e:  # noqa: F821
        return _fail(f"Unknown unit. ({e})")
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to convert this quantity safely. ({e})")


def evaluate_dimensional_expression(expr_str: str) -> dict:
    """Evaluate an expression combining quantities with units, e.g.
    '2 kg * 3 m/s**2' -> 6 N-equivalent (kg*m/s**2), and reject invalid
    combinations like '2 kg + 5 seconds'."""
    err = _require_pint()
    if err:
        return err
    try:
        expr_str = _validate_unit_string(expr_str)
        result = _UREG.parse_expression(expr_str)
        reduced = result.to_base_units()
        return _ok(
            result=round(float(reduced.magnitude), 10),
            unit=f"{result.units:~P}",
            base_unit=f"{reduced.units:~P}",
            result_str=f"{result:~P.6g}",
        )
    except SafetyError as e:
        return _fail(str(e))
    except pint.errors.DimensionalityError as e:  # noqa: F821
        return _fail(f"Invalid dimensional operation: incompatible units. ({e})")
    except pint.errors.UndefinedUnitError as e:  # noqa: F821
        return _fail(f"Unknown unit. ({e})")
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to evaluate this expression safely. ({e})")


def check_addable(quantity_a: str, quantity_b: str) -> dict:
    """Explicitly check whether two quantities can be added/subtracted,
    e.g. rejecting '2 kg' + '5 seconds'."""
    err = _require_pint()
    if err:
        return err
    try:
        a_str = _validate_unit_string(quantity_a)
        b_str = _validate_unit_string(quantity_b)
        qa = _UREG.Quantity(a_str)
        qb = _UREG.Quantity(b_str)
        total = qa + qb  # raises DimensionalityError if incompatible
        return _ok(result=round(float(total.magnitude), 10), unit=f"{total.units:~P}")
    except SafetyError as e:
        return _fail(str(e))
    except pint.errors.DimensionalityError as e:  # noqa: F821
        return _fail(f"Rejected: '{quantity_a}' and '{quantity_b}' have incompatible dimensions.")
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to evaluate this expression safely. ({e})")


def identify_derived_unit(expr_str: str) -> dict:
    """Given a dimensional expression, try to identify a well-known
    derived SI unit it matches (e.g. kg*m/s**2 -> newton)."""
    err = _require_pint()
    if err:
        return err
    try:
        expr_str = _validate_unit_string(expr_str)
        qty = _UREG.parse_expression(expr_str)
        base = qty.to_base_units()

        candidates = ["newton", "joule", "watt", "pascal", "volt", "ohm",
                      "coulomb", "farad", "tesla", "weber", "henry", "hertz"]
        matched = None
        for name in candidates:
            try:
                if base.dimensionality == _UREG.Quantity(1, name).dimensionality:
                    matched = name
                    break
            except Exception:  # noqa: BLE001
                continue

        return _ok(
            base_unit=f"{base.units:~P}",
            matched_derived_unit=matched,
            magnitude=round(float(base.magnitude), 10),
        )
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to identify derived unit safely. ({e})")
