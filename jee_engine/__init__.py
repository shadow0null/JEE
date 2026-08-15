"""
StudyDesk Local JEE Engine
==========================

A completely standalone, offline, deterministic calculation engine for
JEE Mathematics/Physics and NEET Chemistry/Biology. Contains ZERO
network calls and ZERO Gemini / AI integration of any kind.

Modules:
    safety           - input validation / sandboxing for all expression parsing
    router            - classifies a raw question into MATH / NUMERICAL /
                        PHYSICS / CHEMISTRY / BIOLOGY / UNIT / GRAPH / UNKNOWN
    math_engine       - SymPy based symbolic mathematics (algebra, calculus,
                        trigonometry, matrices)
    numerical_engine  - NumPy / SciPy based numerical computation (vectors,
                        root finding, numerical integration, interpolation)
    physics_engine    - JEE physics formula layer (mechanics, electricity,
                        SHM/waves, gravitation)
    chemistry_engine  - JEE/NEET chemistry formula layer (mole concept, gas
                        laws, acid-base, electrochemistry, equation balancing,
                        nuclear decay)
    biology_engine    - NEET biology formula layer (transcription/translation,
                        Mendelian genetics, Hardy-Weinberg, population ecology)
    units_engine      - Pint based unit conversion & dimensional analysis
    graph_engine      - Matplotlib based safe graph generation
    precision_engine  - mpmath based high-precision numerical calculations
    pdf_engine        - PyMuPDF based local JEE PDF text/question extraction
    image_engine      - Pillow based safe image processing
    topic_matcher     - RapidFuzz based JEE chapter/topic fuzzy matching
    topic_graph       - NetworkX based JEE topic prerequisite graph
    analytics_engine  - scikit-learn based optional local performance analytics

This package still contains ZERO network calls and ZERO Gemini / AI
integration of any kind - every new module above is as offline and
deterministic as the original five.
"""

__version__ = "1.2.0"
__all__ = [
    "safety",
    "router",
    "math_engine",
    "numerical_engine",
    "physics_engine",
    "chemistry_engine",
    "biology_engine",
    "units_engine",
    "graph_engine",
    "precision_engine",
    "pdf_engine",
    "image_engine",
    "topic_matcher",
    "topic_graph",
    "analytics_engine",
]
