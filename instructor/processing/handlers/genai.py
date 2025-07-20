"""Google GenAI response handlers."""

from typing import Any

import importlib.util

GENAI_AVAILABLE = importlib.util.find_spec("google.genai") is not None

from ..base import ResponseHandlerBase


class GenAIToolsHandler(ResponseHandlerBase):
    """Handler for GenAI tools mode."""

    def __init__(self):
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package is required for GenAIToolsHandler")
        self.autodetect_images = False

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with GenAI tools."""
        from ...providers.gemini import utils

        # Extract autodetect_images from kwargs
        self.autodetect_images = kwargs.pop("autodetect_images", False)
        return utils.handle_genai_tools(response_model, kwargs, self.autodetect_images)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with GenAI tool error messages."""
        from ...providers.gemini import utils

        return utils.reask_genai_tools(kwargs, response, exception)


class GenAIStructuredOutputsHandler(ResponseHandlerBase):
    """Handler for GenAI structured outputs mode."""

    def __init__(self):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package is required for GenAIStructuredOutputsHandler"
            )
        self.autodetect_images = False

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with GenAI structured outputs."""
        from ...providers.gemini import utils

        # Extract autodetect_images from kwargs
        self.autodetect_images = kwargs.pop("autodetect_images", False)
        return utils.handle_genai_structured_outputs(
            response_model, kwargs, self.autodetect_images
        )

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with GenAI structured outputs formatting."""
        from ...providers.gemini import utils

        return utils.reask_genai_structured_outputs(kwargs, response, exception)
