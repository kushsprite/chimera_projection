"""Chimera: IPL fantasy points prediction, team optimization, and RAG explanations.

This must be the first thing imported so the OpenMP workaround is set before
lightgbm or torch load. On macOS both libraries bundle their own copy of libomp,
and loading both into one process kills the Python kernel without this flag.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.2.0"
