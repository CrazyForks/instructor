"""Backwards compatibility module for instructor._types.

This module has been renamed to types (without underscore prefix) for clarity.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for _types imports."""
    warnings.warn(
        f"Importing from 'instructor._types' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use 'instructor.types' instead:\n"
        "  from instructor.types import ...",
        DeprecationWarning,
        stacklevel=2,
    )

    from .. import types

    # Try to get the attribute from the types module
    if hasattr(types, name):
        return getattr(types, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")