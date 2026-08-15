"""
chemistry_engine.py
====================
A controlled, whitelisted layer of JEE/NEET chemistry calculations:
mole concepts, solution stoichiometry, gas laws, acid-base (pH/pOH),
equation balancing, electrochemistry (Nernst), calorimetry and basic
nuclear/radioactive-decay chemistry.

Same contract as physics_engine.py: every function takes plain
arguments, validates them, and returns either
    {"success": True, "result": ..., ...}
    {"success": False, "error": "..."}
never raises, and makes ZERO network calls.

Molar-mass / percent-composition use a hardcoded atomic-weight table
(same philosophy as physics_engine's hardcoded physical constants)
instead of a chemistry-database dependency, so those two functions
work even without any optional library installed. Equation balancing
uses `chempy` (optional dependency, same guarded-import pattern as
pint in units_engine.py) since correctly balancing an arbitrary
reaction is impractical to hand-roll safely.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from .safety import SafetyError

try:
    from chempy import balance_stoichiometry as _chempy_balance

    CHEMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when chempy is absent
    _chempy_balance = None
    CHEMPY_AVAILABLE = False


GAS_CONSTANT_J = 8.314462618       # J / (mol*K)
GAS_CONSTANT_L_ATM = 0.0820573660  # L*atm / (mol*K)
FARADAY_CONST = 96485.33212        # C / mol
AVOGADRO = 6.02214076e23           # / mol

# Atomic weights (g/mol) for elements that actually appear in the JEE/NEET
# syllabus. Extend this table rather than adding a dependency if a missing
# element is ever needed.
ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.868, "I": 126.904,
    "Ba": 137.327, "Au": 196.967, "Hg": 200.592, "Pb": 207.2, "U": 238.029,
    "Sn": 118.710, "Ti": 47.867,
}


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _require(**kwargs):
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required value(s): {', '.join(missing)}")


def _require_chempy() -> Optional[dict]:
    if not CHEMPY_AVAILABLE:
        return _fail(
            "chempy is not installed. Run 'pip install -r requirements.txt' "
            "to enable chemical equation balancing."
        )
    return None


# --------------------------------------------------------------------------- #
# Formula parsing (small self-contained parser - no eval, whitelist only)
# --------------------------------------------------------------------------- #

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_ALLOWED_RE = re.compile(r"^[A-Za-z0-9\(\)\.]+$")


def _parse_formula(formula: str) -> Dict[str, float]:
    """Parse a simple chemical formula (supports one level of parentheses,
    e.g. 'Ca(OH)2', 'Al2(SO4)3') into {element: count}. Raises ValueError
    on anything unrecognised - deliberately does not support hydrates,
    charges or isotopes."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Formula is required.")
    formula = formula.strip()
    if len(formula) > 60:
        raise ValueError("Formula is too long.")
    if not _FORMULA_ALLOWED_RE.match(formula):
        raise ValueError("Formula contains disallowed characters.")

    # Expand one level of parentheses: X(YZ)n -> repeat YZ's atoms n times.
    def _expand_group(match: "re.Match[str]") -> str:
        inner, mult = match.group(1), match.group(2) or "1"
        counts = _parse_formula(inner)
        expanded = ""
        for el, n in counts.items():
            n_total = n * int(mult)
            expanded += el + (str(int(n_total)) if n_total != 1 else "")
        return expanded

    if "(" in formula:
        formula = re.sub(r"\(([A-Za-z0-9]+)\)(\d*)", _expand_group, formula)
        if "(" in formula:
            raise ValueError("Only one level of parentheses is supported.")

    counts: Dict[str, float] = {}
    pos = 0
    for m in _FORMULA_TOKEN_RE.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"Could not parse formula near '{formula[pos:]}'.")
        el, num = m.group(1), m.group(2)
        if el not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown element symbol '{el}'.")
        counts[el] = counts.get(el, 0) + (int(num) if num else 1)
        pos = m.end()
    if pos != len(formula):
        raise ValueError(f"Could not parse formula near '{formula[pos:]}'.")
    if not counts:
        raise ValueError("Could not parse any elements from the formula.")
    return counts


def molar_mass(formula: Optional[str] = None) -> dict:
    try:
        _require(formula=formula)
        counts = _parse_formula(formula)
        total = sum(ATOMIC_WEIGHTS[el] * n for el, n in counts.items())
        return _ok(result=round(total, 4), unit="g/mol",
                    composition={el: n for el, n in counts.items()},
                    formula=formula)
    except ValueError as e:
        return _fail(str(e))


def percent_composition(formula: Optional[str] = None) -> dict:
    try:
        _require(formula=formula)
        counts = _parse_formula(formula)
        total = sum(ATOMIC_WEIGHTS[el] * n for el, n in counts.items())
        if total == 0:
            raise ValueError("Molar mass evaluated to zero.")
        breakdown = {
            el: round(ATOMIC_WEIGHTS[el] * n / total * 100, 4)
            for el, n in counts.items()
        }
        return _ok(result=breakdown, unit="% by mass",
                    molar_mass_g_per_mol=round(total, 4), formula=formula)
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Mole concept / solutions
# --------------------------------------------------------------------------- #

def moles_from_mass(mass_g: Optional[float] = None, molar_mass_g: Optional[float] = None) -> dict:
    try:
        _require(mass_g=mass_g, molar_mass_g=molar_mass_g)
        if molar_mass_g <= 0:
            raise ValueError("Molar mass must be positive.")
        return _ok(result=round(mass_g / molar_mass_g, 8), unit="mol",
                    formula="n = mass / M")
    except ValueError as e:
        return _fail(str(e))


def mass_from_moles(moles: Optional[float] = None, molar_mass_g: Optional[float] = None) -> dict:
    try:
        _require(moles=moles, molar_mass_g=molar_mass_g)
        return _ok(result=round(moles * molar_mass_g, 8), unit="g",
                    formula="mass = n * M")
    except ValueError as e:
        return _fail(str(e))


def molarity(moles: Optional[float] = None, volume_l: Optional[float] = None) -> dict:
    try:
        _require(moles=moles, volume_l=volume_l)
        if volume_l <= 0:
            raise ValueError("Volume must be positive.")
        return _ok(result=round(moles / volume_l, 8), unit="mol/L",
                    formula="M = n / V")
    except ValueError as e:
        return _fail(str(e))


def dilution(conc1: Optional[float] = None, vol1: Optional[float] = None,
             conc2: Optional[float] = None, vol2: Optional[float] = None) -> dict:
    """Solve M1V1 = M2V2 for whichever single quantity is missing."""
    try:
        provided = [x for x in (conc1, vol1, conc2, vol2) if x is not None]
        if len(provided) != 3:
            raise ValueError("Provide exactly three of: conc1, vol1, conc2, vol2.")
        if conc1 is None:
            if vol1 == 0:
                raise ValueError("vol1 cannot be zero.")
            return _ok(result=round(conc2 * vol2 / vol1, 8), unit="mol/L (conc1)",
                        formula="M1V1 = M2V2")
        if vol1 is None:
            if conc1 == 0:
                raise ValueError("conc1 cannot be zero.")
            return _ok(result=round(conc2 * vol2 / conc1, 8), unit="L (vol1)",
                        formula="M1V1 = M2V2")
        if conc2 is None:
            if vol2 == 0:
                raise ValueError("vol2 cannot be zero.")
            return _ok(result=round(conc1 * vol1 / vol2, 8), unit="mol/L (conc2)",
                        formula="M1V1 = M2V2")
        if vol2 == 0:
            raise ValueError("vol2 cannot be zero.")
        return _ok(result=round(conc1 * vol1 / conc2, 8), unit="L (vol2)",
                    formula="M1V1 = M2V2")
    except ValueError as e:
        return _fail(str(e))


def normality(molarity_val: Optional[float] = None, n_factor: Optional[float] = None) -> dict:
    try:
        _require(molarity_val=molarity_val, n_factor=n_factor)
        return _ok(result=round(molarity_val * n_factor, 8), unit="eq/L",
                    formula="N = M * n_factor")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Gas laws
# --------------------------------------------------------------------------- #

def ideal_gas_law(pressure_atm: Optional[float] = None, volume_l: Optional[float] = None,
                   moles: Optional[float] = None, temp_k: Optional[float] = None) -> dict:
    """Solve PV = nRT for whichever single quantity is missing (R in L*atm)."""
    try:
        provided = [x for x in (pressure_atm, volume_l, moles, temp_k) if x is not None]
        if len(provided) != 3:
            raise ValueError("Provide exactly three of: pressure_atm, volume_l, moles, temp_k.")
        R = GAS_CONSTANT_L_ATM
        if pressure_atm is None:
            if volume_l == 0:
                raise ValueError("volume_l cannot be zero.")
            return _ok(result=round(moles * R * temp_k / volume_l, 8), unit="atm",
                        formula="PV = nRT")
        if volume_l is None:
            if pressure_atm == 0:
                raise ValueError("pressure_atm cannot be zero.")
            return _ok(result=round(moles * R * temp_k / pressure_atm, 8), unit="L",
                        formula="PV = nRT")
        if moles is None:
            return _ok(result=round(pressure_atm * volume_l / (R * temp_k), 8), unit="mol",
                        formula="PV = nRT")
        if pressure_atm * volume_l == 0:
            raise ValueError("pressure_atm * volume_l cannot be zero.")
        return _ok(result=round(pressure_atm * volume_l / (moles * R), 8), unit="K",
                    formula="PV = nRT")
    except ValueError as e:
        return _fail(str(e))


def boyles_law(p1: Optional[float] = None, v1: Optional[float] = None,
                p2: Optional[float] = None, v2: Optional[float] = None) -> dict:
    try:
        provided = [x for x in (p1, v1, p2, v2) if x is not None]
        if len(provided) != 3:
            raise ValueError("Provide exactly three of: p1, v1, p2, v2.")
        if p1 is None:
            return _ok(result=round(p2 * v2 / v1, 8), unit="(pressure) p1", formula="P1V1 = P2V2")
        if v1 is None:
            return _ok(result=round(p2 * v2 / p1, 8), unit="(volume) v1", formula="P1V1 = P2V2")
        if p2 is None:
            return _ok(result=round(p1 * v1 / v2, 8), unit="(pressure) p2", formula="P1V1 = P2V2")
        return _ok(result=round(p1 * v1 / p2, 8), unit="(volume) v2", formula="P1V1 = P2V2")
    except ValueError as e:
        return _fail(str(e))


def charles_law(v1: Optional[float] = None, t1: Optional[float] = None,
                 v2: Optional[float] = None, t2: Optional[float] = None) -> dict:
    """Volume/temperature relationship at constant pressure. Temperatures in Kelvin."""
    try:
        provided = [x for x in (v1, t1, v2, t2) if x is not None]
        if len(provided) != 3:
            raise ValueError("Provide exactly three of: v1, t1, v2, t2 (Kelvin).")
        if v1 is None:
            return _ok(result=round(v2 * t1 / t2, 8), unit="(volume) v1", formula="V1/T1 = V2/T2")
        if t1 is None:
            return _ok(result=round(v1 * t2 / v2, 8), unit="K (t1)", formula="V1/T1 = V2/T2")
        if v2 is None:
            return _ok(result=round(v1 * t2 / t1, 8), unit="(volume) v2", formula="V1/T1 = V2/T2")
        return _ok(result=round(v2 * t1 / v1, 8), unit="K (t2)", formula="V1/T1 = V2/T2")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Acid-base
# --------------------------------------------------------------------------- #

def ph_from_h_concentration(h_conc: Optional[float] = None) -> dict:
    try:
        _require(h_conc=h_conc)
        if h_conc <= 0:
            raise ValueError("[H+] must be positive.")
        return _ok(result=round(-math.log10(h_conc), 6), unit="pH", formula="pH = -log10[H+]")
    except ValueError as e:
        return _fail(str(e))


def h_concentration_from_ph(ph: Optional[float] = None) -> dict:
    try:
        _require(ph=ph)
        return _ok(result=10 ** (-ph), unit="mol/L", formula="[H+] = 10^(-pH)")
    except ValueError as e:
        return _fail(str(e))


def poh_from_oh_concentration(oh_conc: Optional[float] = None) -> dict:
    try:
        _require(oh_conc=oh_conc)
        if oh_conc <= 0:
            raise ValueError("[OH-] must be positive.")
        return _ok(result=round(-math.log10(oh_conc), 6), unit="pOH", formula="pOH = -log10[OH-]")
    except ValueError as e:
        return _fail(str(e))


def ph_poh_relation(ph: Optional[float] = None, poh: Optional[float] = None) -> dict:
    """At 25 degC, pH + pOH = 14."""
    try:
        if ph is None and poh is None:
            raise ValueError("Provide either ph or poh.")
        if ph is None:
            return _ok(result=round(14 - poh, 6), unit="pH", formula="pH + pOH = 14 (at 25\u00b0C)")
        return _ok(result=round(14 - ph, 6), unit="pOH", formula="pH + pOH = 14 (at 25\u00b0C)")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Electrochemistry
# --------------------------------------------------------------------------- #

def nernst_equation(e_standard: Optional[float] = None, n_electrons: Optional[float] = None,
                     reaction_quotient: Optional[float] = None, temp_k: float = 298.0) -> dict:
    try:
        _require(e_standard=e_standard, n_electrons=n_electrons, reaction_quotient=reaction_quotient)
        if n_electrons <= 0:
            raise ValueError("n_electrons must be positive.")
        if reaction_quotient <= 0:
            raise ValueError("reaction_quotient (Q) must be positive.")
        e_cell = e_standard - (GAS_CONSTANT_J * temp_k) / (n_electrons * FARADAY_CONST) * math.log(reaction_quotient)
        return _ok(result=round(e_cell, 8), unit="V",
                    formula="E = E\u00b0 - (RT/nF) ln Q")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Thermochemistry
# --------------------------------------------------------------------------- #

def heat_energy(mass_g: Optional[float] = None, specific_heat: Optional[float] = None,
                 delta_t: Optional[float] = None) -> dict:
    """q = m*c*deltaT. specific_heat in J/(g*K) unless the caller passes
    a different consistent unit."""
    try:
        _require(mass_g=mass_g, specific_heat=specific_heat, delta_t=delta_t)
        q = mass_g * specific_heat * delta_t
        return _ok(result=round(q, 6), unit="J (or unit consistent with specific_heat)",
                    formula="q = m*c*\u0394T")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Nuclear / radioactive decay (first-order kinetics - shared with Biology
# dating problems, kept here since it's fundamentally a chemistry/physics
# topic in both syllabi)
# --------------------------------------------------------------------------- #

def half_life_to_decay_constant(half_life: Optional[float] = None) -> dict:
    try:
        _require(half_life=half_life)
        if half_life <= 0:
            raise ValueError("Half-life must be positive.")
        return _ok(result=round(math.log(2) / half_life, 10), unit="1/time",
                    formula="\u03bb = ln(2) / t_half")
    except ValueError as e:
        return _fail(str(e))


def remaining_quantity(initial: Optional[float] = None, half_life: Optional[float] = None,
                        elapsed_time: Optional[float] = None) -> dict:
    try:
        _require(initial=initial, half_life=half_life, elapsed_time=elapsed_time)
        if half_life <= 0:
            raise ValueError("Half-life must be positive.")
        remaining = initial * (0.5 ** (elapsed_time / half_life))
        return _ok(result=round(remaining, 10), unit="(same unit as initial)",
                    formula="N = N0 * (1/2)^(t / t_half)")
    except ValueError as e:
        return _fail(str(e))


def elapsed_time_from_decay(initial: Optional[float] = None, remaining: Optional[float] = None,
                             half_life: Optional[float] = None) -> dict:
    try:
        _require(initial=initial, remaining=remaining, half_life=half_life)
        if initial <= 0 or remaining <= 0:
            raise ValueError("initial and remaining must be positive.")
        if remaining > initial:
            raise ValueError("remaining cannot exceed initial.")
        t = half_life * math.log2(initial / remaining)
        return _ok(result=round(t, 8), unit="(same unit as half_life)",
                    formula="t = t_half * log2(N0 / N)")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Equation balancing (requires chempy)
# --------------------------------------------------------------------------- #

def balance_equation(reactants: Optional[List[str]] = None, products: Optional[List[str]] = None) -> dict:
    """Balance a chemical equation given lists of reactant/product formulas,
    e.g. reactants=['H2','O2'], products=['H2O']."""
    err = _require_chempy()
    if err:
        return err
    try:
        _require(reactants=reactants, products=products)
        if not reactants or not products:
            raise ValueError("Provide non-empty reactants and products lists.")
        if len(reactants) > 10 or len(products) > 10:
            raise ValueError("Too many species (max 10 each side).")
        for f in list(reactants) + list(products):
            if not isinstance(f, str) or len(f) > 60:
                raise ValueError("Each formula must be a short string.")
        reac_coeffs, prod_coeffs = _chempy_balance(set(reactants), set(products))
        equation = (
            " + ".join(f"{v} {k}" for k, v in reac_coeffs.items())
            + " -> "
            + " + ".join(f"{v} {k}" for k, v in prod_coeffs.items())
        )
        return _ok(
            result=equation,
            reactant_coefficients={str(k): v for k, v in reac_coeffs.items()},
            product_coefficients={str(k): v for k, v in prod_coeffs.items()},
        )
    except ValueError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001 - chempy raises its own error types
        return _fail(f"Could not balance this equation. ({e})")
