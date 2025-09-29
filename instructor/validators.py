"""Backwards compatibility module for instructor.validators.

This module provides lazy imports to maintain backwards compatibility.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for validators imports."""
    warnings.warn(
        f"Importing from 'instructor.validators' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use the new location:\n"
        "  from instructor.validation import llm_validator, openai_moderation",
        DeprecationWarning,
        stacklevel=2,
    )

    from . import validation
    from .processing import validation_utils

    # Try validation module first
    if hasattr(validation, name):
        return getattr(validation, name)

    # Then try processing.validation_utils
    if hasattr(validation_utils, name):
        return getattr(validation_utils, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
