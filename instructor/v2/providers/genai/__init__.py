from __future__ import annotations

from . import handlers  # noqa: F401 - Import to trigger handler registration
from .client import from_genai

__all__ = ["from_genai"]
