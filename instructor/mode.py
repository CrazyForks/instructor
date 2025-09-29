"""Backwards compatibility module for instructor.mode.

This module has been moved to instructor.core.mode.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for mode imports."""
    warnings.warn(
        f"Importing from 'instructor.mode' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use 'instructor.core.mode.{name}' instead:\n"
        "  from instructor.core.mode import Mode",
        DeprecationWarning,
        stacklevel=2,
    )

    from .core import mode

    # Try to get the attribute from the core.mode module
    if hasattr(mode, name):
        return getattr(mode, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")