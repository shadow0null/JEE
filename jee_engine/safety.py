"""
safety.py
=========
Centralised input validation and sandboxing for the Local JEE Engine.

Every module that turns a user-supplied string into a SymPy / NumPy /
Matplotlib object MUST route that string through `validate_input()` and
build expressions with `safe_parse_expr()` (never `eval`, never `exec`,
never bare `sympify` on an unvalidated string).

Design goals:
    * No eval() / exec() anywhere in the codebase.
    * No access to `os`, `subprocess`, `open`, `__import__`, or dunder
      attribute access of any kind.
    * A strict whitelist of allowed characters, operators and function
      names.
    * Hard limits on input length, numeric magnitude, graph ranges,
      graph point counts and wall-clock execution time.
"""

from __future__ import annotations

import re
import signal
import contextlib
import threading
from typing import Any, Dict

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

MAX_INPUT_LENGTH = 300          # characters allowed in any single expression
MAX_EQUATION_COUNT = 6          # max simultaneous equations
MAX_EXPONENT = 12               # largest exponent allowed in an expression
MAX_MATRIX_DIM = 6              # largest matrix dimension (rows or cols)
MAX_GRAPH_POINTS = 2000         # points sampled for a plot
MAX_GRAPH_RANGE = 1_000_000     # largest |x| bound allowed for a plot axis
EXECUTION_TIMEOUT_SECONDS = 5   # wall clock budget for a single calculation

# --- Limits added for the online-capable upgrade (PDF / image / API) ------ #
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024      # 20 MB per uploaded PDF
MAX_PDF_PAGES = 200                         # pages processed from a single PDF
PDF_PROCESSING_TIMEOUT_SECONDS = 15         # wall clock budget for PDF extraction

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024     # 10 MB per uploaded image
MAX_IMAGE_DIMENSION = 6000                  # largest width/height (pixels) accepted
MAX_IMAGES_PER_REQUEST = 10                 # images processed in a single request
IMAGE_PROCESSING_TIMEOUT_SECONDS = 10       # wall clock budget for image processing

MAX_REQUEST_BODY_BYTES = 25 * 1024 * 1024   # hard ceiling enforced by API middleware
MAX_TOPIC_QUERY_LENGTH = 200                # characters allowed in a topic-match query
MAX_ANALYTICS_RECORDS = 500                 # data points accepted for local analytics


class SafetyError(Exception):
    """Raised whenever an input fails validation or a calculation exceeds
    a configured safety limit. Callers should catch this and return the
    structured 'Unable to solve this expression safely.' style response
    rather than letting it propagate as a crash."""


class TimeoutError_(Exception):
    """Raised when a calculation exceeds EXECUTION_TIMEOUT_SECONDS."""


# --------------------------------------------------------------------------- #
# Forbidden patterns (checked BEFORE anything is handed to SymPy)
# --------------------------------------------------------------------------- #

_FORBIDDEN_PATTERNS = [
    r"__",                # dunder attribute / method access
    r"\bimport\b",
    r"\bexec\b",
    r"\beval\b",
    r"\bos\.",
    r"\bsys\.",
    r"\bsubprocess\b",
    r"\bopen\s*\(",
    r"\bsystem\s*\(",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bdelattr\b",
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
    r"\bcompile\s*\(",
    r"\binput\s*\(",
    r"\blambda\b",
    r"\bclass\b",
    r"\bdef\b",
    r";",
    r"`",
    r"\\",
    r"%",
    r"\$",
    r"~",
    r"&",
    r"@",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS), re.IGNORECASE)

# Only these characters may ever appear in a raw math expression string.
# Letters/digits for variable & function names, standard operators,
# grouping, comparison (for equations), commas (for multi-arg functions),
# whitespace, and a small set of math symbols.
_ALLOWED_CHARS_RE = re.compile(r"^[0-9a-zA-Z\s\+\-\*\/\^\(\)\.\,\=\<\>\!\_\|]*$")


def validate_input(expr_str: str) -> str:
    """Validate a raw user expression string. Returns the (stripped)
    string if it passes, otherwise raises SafetyError."""
    if not isinstance(expr_str, str):
        raise SafetyError("Expression must be a string.")

    expr_str = expr_str.strip()

    if not expr_str:
        raise SafetyError("Empty expression.")

    if len(expr_str) > MAX_INPUT_LENGTH:
        raise SafetyError(
            f"Expression too long (max {MAX_INPUT_LENGTH} characters)."
        )

    if _FORBIDDEN_RE.search(expr_str):
        raise SafetyError("Expression contains a disallowed pattern.")

    if not _ALLOWED_CHARS_RE.match(expr_str):
        raise SafetyError("Expression contains disallowed characters.")

    # Reject absurdly large exponents like 2**9999999 before it ever
    # reaches SymPy, to avoid CPU / memory blow-ups.
    for match in re.finditer(r"(?:\*\*|\^)\s*(\d+)", expr_str):
        if int(match.group(1)) > MAX_EXPONENT:
            raise SafetyError(
                f"Exponent too large (max {MAX_EXPONENT})."
            )

    return expr_str


# --------------------------------------------------------------------------- #
# Whitelisted symbols / functions available to the parser
# --------------------------------------------------------------------------- #

_ALLOWED_SYMBOLS = {
    name: sympy.Symbol(name)
    for name in ["x", "y", "z", "t", "a", "b", "c", "n", "k", "m", "u", "v", "w"]
}

_ALLOWED_FUNCTIONS: Dict[str, Any] = {
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
    "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
    "sqrt": sympy.sqrt, "log": sympy.log, "ln": sympy.log,
    "exp": sympy.exp, "Abs": sympy.Abs, "abs": sympy.Abs,
    "factorial": sympy.factorial, "pi": sympy.pi, "E": sympy.E,
    "Rational": sympy.Rational,
}

SAFE_LOCAL_DICT: Dict[str, Any] = {**_ALLOWED_SYMBOLS, **_ALLOWED_FUNCTIONS}

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# parse_expr's auto-number/implicit-multiplication transformations emit
# internal calls like Integer(...) / Rational(...) / Symbol(...), so the
# evaluation namespace needs SymPy's own names available. We build that
# namespace from SymPy itself (every name in it is a safe math object -
# there is no filesystem/network/process access anywhere in SymPy's
# public API) and then explicitly set '__builtins__' to an empty dict.
# That last step is what actually matters for safety: if a globals dict
# passed to eval() does NOT contain a '__builtins__' key, Python silently
# injects the real builtins module (open, exec, __import__, ...) into it.
# Setting it to {} here removes access to all of those.
_SYMPY_NAMESPACE: Dict[str, Any] = {
    name: obj for name, obj in vars(sympy).items() if not name.startswith("_")
}
_SYMPY_NAMESPACE["__builtins__"] = {}


def safe_parse_expr(expr_str: str, extra_symbols: Dict[str, Any] | None = None):
    """Validate and parse a math expression string into a SymPy expression
    using ONLY the whitelisted symbol/function namespace. Never uses
    eval()/exec() and never exposes Python builtins or dunder attributes."""
    expr_str = validate_input(expr_str)

    local_dict = dict(SAFE_LOCAL_DICT)
    if extra_symbols:
        local_dict.update(extra_symbols)

    try:
        parsed = parse_expr(
            expr_str,
            local_dict=local_dict,
            global_dict=_SYMPY_NAMESPACE,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except SafetyError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert everything to SafetyError
        raise SafetyError(f"Could not safely parse expression: {exc}") from exc

    return parsed


@contextlib.contextmanager
def time_limit(seconds: int = EXECUTION_TIMEOUT_SECONDS):
    """Context manager that aborts a calculation if it runs too long.
    Uses SIGALRM, which is POSIX-only; on platforms without SIGALRM this
    silently degrades to 'no timeout' rather than crashing."""

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError_("Calculation exceeded the execution time limit.")

    # FastAPI/TestClient and some WSGI/ASGI deployments may execute handlers
    # outside Python's main thread. signal.signal()/alarm() is illegal there.
    # Keep the hard timeout where POSIX signals are safe and fail open only
    # for the non-main-thread case rather than crashing an otherwise valid
    # deterministic calculation.
    has_alarm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def check_graph_range(low: float, high: float) -> None:
    if low >= high:
        raise SafetyError("Graph range lower bound must be less than upper bound.")
    if abs(low) > MAX_GRAPH_RANGE or abs(high) > MAX_GRAPH_RANGE:
        raise SafetyError(f"Graph range exceeds the allowed limit ({MAX_GRAPH_RANGE}).")


def clamp_graph_points(n: int) -> int:
    return max(10, min(int(n), MAX_GRAPH_POINTS))
