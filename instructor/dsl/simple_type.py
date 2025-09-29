"""Backwards compatibility module for instructor.dsl.simple_type.

This module has been renamed to primitives.py for clarity.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for simple_type imports."""
    warnings.warn(
        f"Importing from 'instructor.dsl.simple_type' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use 'instructor.dsl.primitives.{name}' instead:\n"
        "  from instructor.dsl.primitives import is_simple_type, ModelAdapter",
        DeprecationWarning,
        stacklevel=2,
    )

    from . import primitives

    # Try to get the attribute from the primitives module
    if hasattr(primitives, name):
        return getattr(primitives, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")