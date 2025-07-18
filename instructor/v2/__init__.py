"""Instructor v2 - Adapter-based implementation"""

from .adapters.registry import registry
from .adapters.openai import OpenAIAdapter

# Register adapters
registry.register(OpenAIAdapter())

__all__ = ["registry"]
