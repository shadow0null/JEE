"""
physics_engine.py
==================
A controlled, whitelisted layer of JEE physics formulas.

Every function takes plain numeric keyword arguments (already extracted
by the router / caller), validates them, computes a result with
NumPy where useful, and returns:
    {"success": True, "result": <float>, "unit": "<SI unit>", "formula": "..."}
    {"success": False, "error": "..."}

No expression parsing happens here (that's math_engine / units_engine's
job) - this module is a fixed set of known-safe formulas, which is what
keeps it deterministic and trivially safe.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

G_CONST = 6.674e-11       # gravitational constant, N m^2/kg^2
G_EARTH = 9.8             # standard gravity, m/s^2
ELEMENTARY_CHARGE = 1.602176634e-19  # coulombs


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(result, unit: str, formula: str) -> dict:
    return {"success": True, "result": round(float(result), 10), "unit": unit, "formula": formula}


def _require(**kwargs):
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required value(s): {', '.join(missing)}")


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #

def velocity(distance: Optional[float] = None, time: Optional[float] = None) -> dict:
    try:
        _require(distance=distance, time=time)
        if time == 0:
            raise ValueError("Time cannot be zero.")
        return _ok(distance / time, "m/s", "v = d / t")
    except ValueError as e:
        return _fail(str(e))


def acceleration(delta_v: Optional[float] = None, time: Optional[float] = None,
                  final_v: Optional[float] = None, initial_v: Optional[float] = None) -> dict:
    try:
        if delta_v is None and final_v is not None and initial_v is not None:
            delta_v = final_v - initial_v
        _require(delta_v=delta_v, time=time)
        if time == 0:
            raise ValueError("Time cannot be zero.")
        return _ok(delta_v / time, "m/s^2", "a = Δv / t")
    except ValueError as e:
        return _fail(str(e))


def displacement(initial_v: Optional[float] = None, time: Optional[float] = None,
                  accel: Optional[float] = None) -> dict:
    try:
        _require(initial_v=initial_v, time=time, accel=accel)
        s = initial_v * time + 0.5 * accel * time ** 2
        return _ok(s, "m", "s = u*t + 1/2*a*t^2")
    except ValueError as e:
        return _fail(str(e))


def projectile_motion(speed: Optional[float] = None, angle_deg: Optional[float] = None,
                       gravity: float = G_EARTH) -> dict:
    """Returns range, max height and time of flight for projectile launched
    on level ground."""
    try:
        _require(speed=speed, angle_deg=angle_deg)
        theta = math.radians(angle_deg)
        v = speed
        time_of_flight = (2 * v * math.sin(theta)) / gravity
        max_height = (v ** 2) * (math.sin(theta) ** 2) / (2 * gravity)
        horiz_range = (v ** 2) * math.sin(2 * theta) / gravity
        return {
            "success": True,
            "result": {
                "range_m": round(horiz_range, 6),
                "max_height_m": round(max_height, 6),
                "time_of_flight_s": round(time_of_flight, 6),
            },
            "unit": "m / s",
            "formula": "R = v^2*sin(2θ)/g, H = v^2*sin^2(θ)/(2g), T = 2v*sin(θ)/g",
        }
    except ValueError as e:
        return _fail(str(e))


def force(mass: Optional[float] = None, accel: Optional[float] = None) -> dict:
    try:
        _require(mass=mass, accel=accel)
        return _ok(mass * accel, "N", "F = m * a")
    except ValueError as e:
        return _fail(str(e))


def work_done(force_n: Optional[float] = None, distance: Optional[float] = None,
              angle_deg: float = 0.0) -> dict:
    try:
        _require(force_n=force_n, distance=distance)
        w = force_n * distance * math.cos(math.radians(angle_deg))
        return _ok(w, "J", "W = F * d * cos(θ)")
    except ValueError as e:
        return _fail(str(e))


def power(work_j: Optional[float] = None, time: Optional[float] = None) -> dict:
    try:
        _require(work_j=work_j, time=time)
        if time == 0:
            raise ValueError("Time cannot be zero.")
        return _ok(work_j / time, "W", "P = W / t")
    except ValueError as e:
        return _fail(str(e))


def kinetic_energy(mass: Optional[float] = None, speed: Optional[float] = None) -> dict:
    try:
        _require(mass=mass, speed=speed)
        return _ok(0.5 * mass * speed ** 2, "J", "KE = 1/2 * m * v^2")
    except ValueError as e:
        return _fail(str(e))


def potential_energy(mass: Optional[float] = None, height: Optional[float] = None,
                      gravity: float = G_EARTH) -> dict:
    try:
        _require(mass=mass, height=height)
        return _ok(mass * gravity * height, "J", "PE = m * g * h")
    except ValueError as e:
        return _fail(str(e))


def momentum(mass: Optional[float] = None, speed: Optional[float] = None) -> dict:
    try:
        _require(mass=mass, speed=speed)
        return _ok(mass * speed, "kg*m/s", "p = m * v")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Electricity
# --------------------------------------------------------------------------- #

def ohms_law(voltage: Optional[float] = None, current: Optional[float] = None,
             resistance: Optional[float] = None) -> dict:
    try:
        provided = [x for x in (voltage, current, resistance) if x is not None]
        if len(provided) != 2:
            raise ValueError("Provide exactly two of: voltage, current, resistance.")
        if voltage is None:
            return _ok(current * resistance, "V", "V = I * R")
        if current is None:
            if resistance == 0:
                raise ValueError("Resistance cannot be zero.")
            return _ok(voltage / resistance, "A", "I = V / R")
        if current == 0:
            raise ValueError("Current cannot be zero.")
        return _ok(voltage / current, "Ω", "R = V / I")
    except ValueError as e:
        return _fail(str(e))


def electrical_power(voltage: Optional[float] = None, current: Optional[float] = None,
                      resistance: Optional[float] = None) -> dict:
    try:
        if voltage is not None and current is not None:
            return _ok(voltage * current, "W", "P = V * I")
        if current is not None and resistance is not None:
            return _ok(current ** 2 * resistance, "W", "P = I^2 * R")
        if voltage is not None and resistance is not None:
            if resistance == 0:
                raise ValueError("Resistance cannot be zero.")
            return _ok(voltage ** 2 / resistance, "W", "P = V^2 / R")
        raise ValueError("Provide any two of: voltage, current, resistance.")
    except ValueError as e:
        return _fail(str(e))


def series_resistance(resistances: Optional[list] = None) -> dict:
    try:
        if not resistances:
            raise ValueError("Provide a non-empty list of resistances.")
        if len(resistances) > 20:
            raise ValueError("Too many resistors (max 20).")
        total = float(np.sum(np.array(resistances, dtype=float)))
        return _ok(total, "Ω", "R_total = R1 + R2 + ... + Rn")
    except ValueError as e:
        return _fail(str(e))


def parallel_resistance(resistances: Optional[list] = None) -> dict:
    try:
        if not resistances:
            raise ValueError("Provide a non-empty list of resistances.")
        if len(resistances) > 20:
            raise ValueError("Too many resistors (max 20).")
        arr = np.array(resistances, dtype=float)
        if np.any(arr == 0):
            raise ValueError("Resistance values cannot be zero.")
        total = float(1.0 / np.sum(1.0 / arr))
        return _ok(total, "Ω", "1/R_total = 1/R1 + 1/R2 + ... + 1/Rn")
    except ValueError as e:
        return _fail(str(e))


def charge(current: Optional[float] = None, time: Optional[float] = None) -> dict:
    try:
        _require(current=current, time=time)
        return _ok(current * time, "C", "Q = I * t")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# SHM / Waves
# --------------------------------------------------------------------------- #

def frequency_from_period(period: Optional[float] = None) -> dict:
    try:
        _require(period=period)
        if period == 0:
            raise ValueError("Period cannot be zero.")
        return _ok(1.0 / period, "Hz", "f = 1 / T")
    except ValueError as e:
        return _fail(str(e))


def period_from_frequency(freq: Optional[float] = None) -> dict:
    try:
        _require(freq=freq)
        if freq == 0:
            raise ValueError("Frequency cannot be zero.")
        return _ok(1.0 / freq, "s", "T = 1 / f")
    except ValueError as e:
        return _fail(str(e))


def angular_frequency(freq: Optional[float] = None, period: Optional[float] = None) -> dict:
    try:
        if freq is not None:
            return _ok(2 * math.pi * freq, "rad/s", "ω = 2πf")
        if period is not None:
            if period == 0:
                raise ValueError("Period cannot be zero.")
            return _ok(2 * math.pi / period, "rad/s", "ω = 2π / T")
        raise ValueError("Provide either freq or period.")
    except ValueError as e:
        return _fail(str(e))


def shm_displacement(amplitude: Optional[float] = None, omega: Optional[float] = None,
                      time: Optional[float] = None, phase: float = 0.0) -> dict:
    try:
        _require(amplitude=amplitude, omega=omega, time=time)
        x = amplitude * math.cos(omega * time + phase)
        return _ok(x, "m", "x = A*cos(ωt + φ)")
    except ValueError as e:
        return _fail(str(e))


def shm_velocity(amplitude: Optional[float] = None, omega: Optional[float] = None,
                  time: Optional[float] = None, phase: float = 0.0) -> dict:
    try:
        _require(amplitude=amplitude, omega=omega, time=time)
        v = -amplitude * omega * math.sin(omega * time + phase)
        return _ok(v, "m/s", "v = -Aω*sin(ωt + φ)")
    except ValueError as e:
        return _fail(str(e))


def shm_max_acceleration(amplitude: Optional[float] = None, omega: Optional[float] = None) -> dict:
    try:
        _require(amplitude=amplitude, omega=omega)
        return _ok(amplitude * omega ** 2, "m/s^2", "a_max = A*ω^2")
    except ValueError as e:
        return _fail(str(e))


# --------------------------------------------------------------------------- #
# Gravitation
# --------------------------------------------------------------------------- #

def gravitational_force(m1: Optional[float] = None, m2: Optional[float] = None,
                         distance: Optional[float] = None) -> dict:
    try:
        _require(m1=m1, m2=m2, distance=distance)
        if distance == 0:
            raise ValueError("Distance cannot be zero.")
        f = G_CONST * m1 * m2 / distance ** 2
        return _ok(f, "N", "F = G*m1*m2 / r^2")
    except ValueError as e:
        return _fail(str(e))


def gravitational_potential_energy(m1: Optional[float] = None, m2: Optional[float] = None,
                                    distance: Optional[float] = None) -> dict:
    try:
        _require(m1=m1, m2=m2, distance=distance)
        if distance == 0:
            raise ValueError("Distance cannot be zero.")
        u = -G_CONST * m1 * m2 / distance
        return _ok(u, "J", "U = -G*m1*m2 / r")
    except ValueError as e:
        return _fail(str(e))


def escape_velocity(mass: Optional[float] = None, radius: Optional[float] = None) -> dict:
    try:
        _require(mass=mass, radius=radius)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        v = math.sqrt(2 * G_CONST * mass / radius)
        return _ok(v, "m/s", "v_e = sqrt(2*G*M / R)")
    except ValueError as e:
        return _fail(str(e))
