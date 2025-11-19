from __future__ import annotations

from .client import from_genai
from . import handlers  # noqa: F401 - Import to trigger handler registration

__all__ = ["from_genai"]

