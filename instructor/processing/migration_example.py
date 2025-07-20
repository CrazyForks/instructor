"""
Example showing how to migrate from the old dictionary-based system to the new handler registry.

This demonstrates how the existing code can be gradually migrated while maintaining
backward compatibility.
"""

from typing import Any
from ..mode import Mode
from .handlers import handler_registry


def handle_response_model_with_fallback(
    response_model: type[Any] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Enhanced version that tries the new handler registry first, then falls back to old system.

    This allows gradual migration while ensuring nothing breaks.
    """

    # Try the new handler registry first
    if handler_registry.has_handler(mode):
        try:
            handler = handler_registry.get_handler(mode)
            return handler.prepare_request(response_model, kwargs)
        except Exception as e:
            # Log the error and fall back to old system
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Handler failed for mode {mode}, falling back to old system: {e}"
            )

    # Fall back to the old dictionary-based system
    from .response import handle_response_model as old_handle_response_model

    return old_handle_response_model(response_model, mode, **kwargs)


# Example of how to register a custom handler
def register_custom_handler():
    """Example of registering a custom handler for a specific use case."""

    from .handlers.base import ResponseHandlerBase

    class CustomHandler(ResponseHandlerBase):
        def prepare_request(
            self, response_model: type[Any] | None, kwargs: dict[str, Any]
        ) -> tuple[type[Any] | None, dict[str, Any]]:
            # Custom preparation logic
            kwargs["custom_param"] = "custom_value"
            return response_model, kwargs

        def format_retry_request(
            self, kwargs: dict[str, Any], response: Any, exception: Exception
        ) -> dict[str, Any]:
            # Custom retry logic
            # response is intentionally unused in this example
            _ = response
            kwargs["messages"].append(
                {"role": "system", "content": f"Custom retry handler: {exception}"}
            )
            return kwargs

    # Register the custom handler for a specific mode
    handler_registry.register(Mode.TOOLS, CustomHandler())


# Example of conditional registration based on environment
def register_experimental_handlers():
    """Register experimental handlers only if enabled."""

    import os

    if os.getenv("INSTRUCTOR_EXPERIMENTAL_HANDLERS") == "true":
        # Register experimental handlers
        handler_registry.register_lazy(
            Mode.TOOLS,
            "experimental",
            "ExperimentalToolsHandler",
            required_spec=None,  # No special package required
        )
