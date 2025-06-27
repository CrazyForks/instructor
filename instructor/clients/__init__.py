"""Client factory functions for various LLM providers.

This module provides factory functions to create Instructor instances
from different LLM provider clients. Each function handles the specific
requirements and configurations needed for that provider.
"""

from __future__ import annotations
import importlib.util

# Always available - from main instructor module
from ..client import from_openai

__all__ = ["from_openai"]

# Conditional imports based on available packages
if importlib.util.find_spec("anthropic") is not None:
    from .client_anthropic import from_anthropic

    __all__ += ["from_anthropic"]

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    from .client_gemini import from_gemini

    __all__ += ["from_gemini"]

if importlib.util.find_spec("fireworks") is not None:
    from .client_fireworks import from_fireworks

    __all__ += ["from_fireworks"]

if importlib.util.find_spec("cerebras") is not None:
    from .client_cerebras import from_cerebras

    __all__ += ["from_cerebras"]

if importlib.util.find_spec("groq") is not None:
    from .client_groq import from_groq

    __all__ += ["from_groq"]

if importlib.util.find_spec("mistralai") is not None:
    from .client_mistral import from_mistral

    __all__ += ["from_mistral"]

if importlib.util.find_spec("cohere") is not None:
    from .client_cohere import from_cohere

    __all__ += ["from_cohere"]

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    from .client_vertexai import from_vertexai

    __all__ += ["from_vertexai"]

if importlib.util.find_spec("boto3") is not None:
    from .client_bedrock import from_bedrock

    __all__ += ["from_bedrock"]

if importlib.util.find_spec("writerai") is not None:
    from .client_writer import from_writer

    __all__ += ["from_writer"]

if importlib.util.find_spec("openai") is not None:
    from .client_perplexity import from_perplexity

    __all__ += ["from_perplexity"]

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.genai") is not None
):
    from .client_genai import from_genai

    __all__ += ["from_genai"]
