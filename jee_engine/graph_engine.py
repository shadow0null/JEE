"""
graph_engine.py
================
Safe graph generation using Matplotlib.

Every expression string is parsed through the shared safety module
(SymPy safe_parse_expr) before being lambdified into a NumPy-callable
function. Graph ranges and point counts are clamped to the limits
defined in safety.py to bound output size, memory and execution time.
"""

from __future__ import annotations

import os
import time
import uuid
import math
from typing import Sequence

import numpy as np
import sympy

import matplotlib
matplotlib.use("Agg")  # headless, no GUI backend - safe for server/CLI use
import matplotlib.pyplot as plt

from .safety import (
    safe_parse_expr,
    SafetyError,
    check_graph_range,
    clamp_graph_points,
    time_limit,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _new_output_path(prefix: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    return os.path.join(OUTPUT_DIR, filename)


def plot_function(expr_str: str, var: str = "x", x_min: float = -10, x_max: float = 10,
                   points: int = 500, title: str | None = None) -> dict:
    """Plot y = f(x) for an arbitrary safely-parsed expression."""
    try:
        check_graph_range(x_min, x_max)
        points = clamp_graph_points(points)

        v = sympy.Symbol(var)
        expr = safe_parse_expr(expr_str, {var: v})
        f = sympy.lambdify(v, expr, modules=["numpy"])

        xs = np.linspace(x_min, x_max, points)
        with time_limit():
            ys = f(xs)
            ys = np.asarray(ys, dtype=float)
            ys = np.where(np.isfinite(ys), ys, np.nan)

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110)
        ax.plot(xs, ys, color="#3b6fd6", linewidth=2)
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.set_xlabel(var)
        ax.set_ylabel(f"f({var})")
        ax.set_title(title or f"y = {expr_str}")
        ax.grid(True, linestyle="--", alpha=0.4)

        path = _new_output_path("plot")
        fig.savefig(path)
        plt.close(fig)

        return _ok(file_path=path, expression=str(expr))
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to generate this graph safely. ({e})")


def plot_projectile_trajectory(speed: float, angle_deg: float, gravity: float = 9.8) -> dict:
    try:
        if speed <= 0 or speed > 1e6:
            raise SafetyError("Speed out of allowed range.")
        if not (0 < angle_deg < 90):
            raise SafetyError("Angle must be between 0 and 90 degrees.")

        theta = math.radians(angle_deg)
        t_flight = 2 * speed * math.sin(theta) / gravity
        check_graph_range(0, t_flight if t_flight > 0 else 1)

        ts = np.linspace(0, t_flight, clamp_graph_points(500))
        xs = speed * math.cos(theta) * ts
        ys = speed * math.sin(theta) * ts - 0.5 * gravity * ts ** 2
        ys = np.clip(ys, 0, None)

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110)
        ax.plot(xs, ys, color="#d6633b", linewidth=2)
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title(f"Projectile trajectory (v0={speed} m/s, θ={angle_deg}°)")
        ax.grid(True, linestyle="--", alpha=0.4)

        path = _new_output_path("projectile")
        fig.savefig(path)
        plt.close(fig)
        return _ok(file_path=path)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to generate this graph safely. ({e})")


def plot_kinematics(kind: str, initial_v: float = 0.0, accel: float = 0.0,
                     t_max: float = 10.0) -> dict:
    """kind: 'position' | 'velocity' | 'acceleration' vs time, for
    constant-acceleration 1D motion."""
    try:
        if kind not in ("position", "velocity", "acceleration"):
            raise SafetyError("kind must be 'position', 'velocity' or 'acceleration'.")
        check_graph_range(0, t_max)
        ts = np.linspace(0, t_max, clamp_graph_points(500))

        if kind == "position":
            ys = initial_v * ts + 0.5 * accel * ts ** 2
            ylabel = "Position (m)"
        elif kind == "velocity":
            ys = initial_v + accel * ts
            ylabel = "Velocity (m/s)"
        else:
            ys = np.full_like(ts, accel)
            ylabel = "Acceleration (m/s^2)"

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110)
        ax.plot(ts, ys, color="#2f9e44", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{kind.capitalize()}-time graph")
        ax.grid(True, linestyle="--", alpha=0.4)

        path = _new_output_path(f"{kind}_time")
        fig.savefig(path)
        plt.close(fig)
        return _ok(file_path=path)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to generate this graph safely. ({e})")


def plot_shm(amplitude: float, omega: float, phase: float = 0.0, cycles: float = 2.0) -> dict:
    try:
        if amplitude <= 0 or amplitude > 1e6:
            raise SafetyError("Amplitude out of allowed range.")
        if omega <= 0 or omega > 1e4:
            raise SafetyError("Angular frequency out of allowed range.")
        period = 2 * math.pi / omega
        t_max = period * min(max(cycles, 0.5), 20)
        check_graph_range(0, t_max)

        ts = np.linspace(0, t_max, clamp_graph_points(800))
        xs = amplitude * np.cos(omega * ts + phase)

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110)
        ax.plot(ts, xs, color="#9c36b5", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title("SHM: displacement vs time")
        ax.grid(True, linestyle="--", alpha=0.4)

        path = _new_output_path("shm")
        fig.savefig(path)
        plt.close(fig)
        return _ok(file_path=path)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to generate this graph safely. ({e})")
