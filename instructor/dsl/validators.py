"""Backwards compatibility module for instructor.dsl.validators.

This module provides lazy imports to avoid circular import issues.
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    from ..processing import validation_utils
    from .. import validation

    # Try processing.validation_utils first
    if hasattr(validation_utils, name):
        return getattr(validation_utils, name)

    # Then try validation module
    if hasattr(validation, name):
        return getattr(validation, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
