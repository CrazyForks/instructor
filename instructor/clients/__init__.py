"""Client factory functions for various LLM providers.

This module provides factory functions to create Instructor instances
from different LLM provider clients. Each function handles the specific
requirements and configurations needed for that provider.
"""

from __future__ import annotations

# The actual imports are handled conditionally in instructor/__init__.py
# to avoid import errors when optional dependencies are not installed.
# This file serves as the package marker for the clients submodule.

__all__ = [
    "from_anthropic",
    "from_bedrock",
    "from_cerebras",
    "from_cohere",
    "from_fireworks",
    "from_gemini",
    "from_genai",
    "from_groq",
    "from_mistral",
    "from_perplexity",
    "from_vertexai",
    "from_writer",
]
