"""Gemini/Google response handlers."""

from typing import Any

import importlib.util

GEMINI_AVAILABLE = importlib.util.find_spec("google.generativeai") is not None

from ..base import ResponseHandlerBase


class GeminiJSONHandler(ResponseHandlerBase):
    """Handler for Gemini JSON mode."""

    def __init__(self):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required for GeminiJSONHandler"
            )

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Gemini JSON mode."""
        from ...providers.gemini import utils

        return utils.handle_gemini_json(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Gemini JSON formatting."""
        from ...providers.gemini import utils

        return utils.reask_gemini_json(kwargs, response, exception)


class GeminiToolsHandler(ResponseHandlerBase):
    """Handler for Gemini tools mode."""

    def __init__(self):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required for GeminiToolsHandler"
            )

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with Gemini tools."""
        from ...providers.gemini import utils

        return utils.handle_gemini_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with Gemini tool error messages."""
        from ...providers.gemini import utils

        return utils.reask_gemini_tools(kwargs, response, exception)
