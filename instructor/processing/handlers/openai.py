"""OpenAI response handlers."""

from typing import Any

from ..base import ResponseHandlerBase
from ...providers.openai import utils


class DefaultHandler(ResponseHandlerBase):
    """Default handler for basic response processing."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Default preparation - no modifications."""
        return response_model, kwargs

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Use default retry formatting."""
        return utils.reask_default(kwargs, response, exception)


class ToolsHandler(ResponseHandlerBase):
    """Handler for OpenAI tools mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with tools."""
        return utils.handle_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with tool error messages."""
        return utils.reask_tools(kwargs, response, exception)


class ToolsStrictHandler(ResponseHandlerBase):
    """Handler for OpenAI strict tools mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with strict tools."""
        return utils.handle_tools_strict(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with tool error messages."""
        return utils.reask_tools(kwargs, response, exception)


class FunctionsHandler(ResponseHandlerBase):
    """Handler for OpenAI functions mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with functions."""
        return utils.handle_functions(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with default formatting."""
        return utils.reask_default(kwargs, response, exception)


class JSONHandler(ResponseHandlerBase):
    """Handler for OpenAI JSON mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with JSON mode."""
        from ...mode import Mode

        return utils.handle_json_modes(response_model, kwargs, Mode.JSON)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with markdown JSON formatting."""
        return utils.reask_md_json(kwargs, response, exception)


class MDJSONHandler(ResponseHandlerBase):
    """Handler for OpenAI Markdown JSON mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with MD_JSON mode."""
        from ...mode import Mode

        return utils.handle_json_modes(response_model, kwargs, Mode.MD_JSON)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with markdown JSON formatting."""
        return utils.reask_md_json(kwargs, response, exception)


class JSONSchemaHandler(ResponseHandlerBase):
    """Handler for OpenAI JSON Schema mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with JSON_SCHEMA mode."""
        from ...mode import Mode

        return utils.handle_json_modes(response_model, kwargs, Mode.JSON_SCHEMA)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with markdown JSON formatting."""
        return utils.reask_md_json(kwargs, response, exception)


class JSONO1Handler(ResponseHandlerBase):
    """Handler for OpenAI O1 JSON mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request for O1 JSON mode."""
        return utils.handle_json_o1(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with default formatting."""
        return utils.reask_default(kwargs, response, exception)


class ParallelToolsHandler(ResponseHandlerBase):
    """Handler for OpenAI parallel tools mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with parallel tools."""
        return utils.handle_parallel_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with tool error messages."""
        return utils.reask_tools(kwargs, response, exception)


class ResponsesToolsHandler(ResponseHandlerBase):
    """Handler for OpenAI responses tools mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with responses tools."""
        return utils.handle_responses_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with responses tools formatting."""
        return utils.reask_responses_tools(kwargs, response, exception)


class ResponsesToolsWithInbuiltHandler(ResponseHandlerBase):
    """Handler for OpenAI responses tools with inbuilt tools mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request with responses tools with inbuilt."""
        return utils.handle_responses_tools_with_inbuilt_tools(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with responses tools formatting."""
        return utils.reask_responses_tools(kwargs, response, exception)


class OpenRouterHandler(ResponseHandlerBase):
    """Handler for OpenRouter structured outputs mode."""

    def prepare_request(
        self, response_model: type[Any] | None, kwargs: dict[str, Any]
    ) -> tuple[type[Any] | None, dict[str, Any]]:
        """Prepare request for OpenRouter."""
        return utils.handle_openrouter_structured_outputs(response_model, kwargs)

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format retry with default formatting."""
        return utils.reask_default(kwargs, response, exception)
