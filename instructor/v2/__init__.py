from __future__ import annotations

from .core.patch import patch_v2

__all__: list[str] = ["patch_v2"]

try:
    from .providers.genai.client import from_genai
    __all__.append("from_genai")
except ModuleNotFoundError:  # google-genai not installed
    from_genai = None  # type: ignore


