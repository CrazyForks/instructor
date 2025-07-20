"""Anthropic response handlers."""

from typing import Any

import importlib.util

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None

from ..base import ResponseHandlerBase


class AnthropicToolsHandler(ResponseHandlerBase):
    """Handler for Anthropic tools mode."""

    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for AnthropicToolsHandler")

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Anthropic tools."""
        from ...providers.anthropic import utils

        return utils.handle_anthropic_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Anthropic tool error messages."""
        from ...providers.anthropic import utils

        return utils.reask_anthropic_tools(kwargs, response, exception)


class AnthropicReasoningToolsHandler(ResponseHandlerBase):
    """Handler for Anthropic reasoning tools mode."""

    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for AnthropicReasoningToolsHandler"
            )

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Anthropic reasoning tools."""
        from ...providers.anthropic import utils

        return utils.handle_anthropic_reasoning_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Anthropic tool error messages."""
        from ...providers.anthropic import utils

        return utils.reask_anthropic_tools(kwargs, response, exception)


class AnthropicJSONHandler(ResponseHandlerBase):
    """Handler for Anthropic JSON mode."""

    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for AnthropicJSONHandler")

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Anthropic JSON mode."""
        from ...providers.anthropic import utils

        return utils.handle_anthropic_json(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Anthropic JSON formatting."""
        from ...providers.anthropic import utils

        return utils.reask_anthropic_json(kwargs, response, exception)


class AnthropicParallelToolsHandler(ResponseHandlerBase):
    """Handler for Anthropic parallel tools mode."""

    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for AnthropicParallelToolsHandler"
            )

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Anthropic parallel tools."""
        from ...providers.anthropic import utils

        return utils.handle_anthropic_parallel_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Anthropic tool error messages."""
        from ...providers.anthropic import utils

        return utils.reask_anthropic_tools(kwargs, response, exception)
