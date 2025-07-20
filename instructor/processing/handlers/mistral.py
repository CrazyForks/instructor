"""Mistral response handlers."""

from typing import Any

import importlib.util

MISTRAL_AVAILABLE = importlib.util.find_spec("mistralai") is not None

from ..base import ResponseHandlerBase


class MistralToolsHandler(ResponseHandlerBase):
    """Handler for Mistral tools mode."""

    def __init__(self):
        if not MISTRAL_AVAILABLE:
            raise ImportError("mistralai package is required for MistralToolsHandler")

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Mistral tools."""
        from ...providers.mistral import utils

        return utils.handle_mistral_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Mistral tool error messages."""
        from ...providers.mistral import utils

        return utils.reask_mistral_tools(kwargs, response, exception)


class MistralStructuredOutputsHandler(ResponseHandlerBase):
    """Handler for Mistral structured outputs mode."""

    def __init__(self):
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "mistralai package is required for MistralStructuredOutputsHandler"
            )

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Mistral structured outputs."""
        from ...providers.mistral import utils

        return utils.handle_mistral_structured_outputs(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Mistral structured outputs formatting."""
        from ...providers.mistral import utils

        return utils.reask_mistral_structured_outputs(kwargs, response, exception)
