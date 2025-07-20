"""
Response handler registry with lazy loading for optional dependencies.

This module provides a registry system that only loads handlers when their
corresponding provider libraries are available in the environment.
"""

from .registry import handler_registry, HandlerRegistry
from .base import ResponseHandler, ResponseHandlerBase

__all__ = [
    "handler_registry",
    "HandlerRegistry",
    "ResponseHandler",
    "ResponseHandlerBase",
]
