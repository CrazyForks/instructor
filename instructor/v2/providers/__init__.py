from __future__ import annotations

__all__: list[str] = []

try:
    from . import genai as genai  # noqa: F401

    __all__.append("genai")
except ModuleNotFoundError:
    pass

