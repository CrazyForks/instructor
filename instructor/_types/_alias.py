"""Backwards compatibility module for instructor._types._alias.

This module has been renamed to types._alias for clarity.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for _types._alias imports."""
    warnings.warn(
        f"Importing from 'instructor._types._alias' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use 'instructor.types._alias.{name}' instead",
        DeprecationWarning,
        stacklevel=2,
    )

    from ..types import _alias

    # Try to get the attribute from the types._alias module
    if hasattr(_alias, name):
        return getattr(_alias, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")