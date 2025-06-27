"""OpenAI provider implementation."""

from __future__ import annotations

import json
from functools import partial
from textwrap import dedent
from typing import Any, Callable, TypeVar

from instructor.providers.base import BaseProvider, ProviderRegistry
from instructor.mode import Mode
from instructor.utils import merge_consecutive_messages

T = TypeVar("T")


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation.

    Supports models from OpenAI and OpenAI-compatible providers like:
    - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
    - Anyscale
    - Together
    - Databricks
    - OpenRouter (with some modes)
    """

    @property
    def name(self) -> str:
        return "openai"

    @property
    def package_name(self) -> str:
        return "openai"

    def create_client(
        self, _model_name: str, async_client: bool = False, **kwargs
    ) -> Any:
        """Create OpenAI client instance."""
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI package is not installed. Install it with: pip install openai"
            ) from e

        if async_client:
            return openai.AsyncOpenAI(**kwargs)
        return openai.OpenAI(**kwargs)

    def from_client(self, client: Any, **kwargs) -> Any:
        """Create Instructor instance from OpenAI client.

        This wraps the OpenAI client with Instructor's patching to enable
        structured outputs via function calling, tools, or JSON modes.
        """
        import openai
        from instructor.client import (
            Instructor,
            AsyncInstructor,
            Provider,
            get_provider,
        )
        from instructor.patch import patch

        # Determine provider based on base URL
        if hasattr(client, "base_url"):
            provider = get_provider(str(client.base_url))
        else:
            provider = Provider.OPENAI

        mode = kwargs.get("mode", Mode.TOOLS)

        # Validate mode based on provider
        if provider in {Provider.OPENROUTER}:
            assert mode in {
                Mode.TOOLS,
                Mode.OPENROUTER_STRUCTURED_OUTPUTS,
                Mode.JSON,
            }, (
                f"OpenRouter only supports TOOLS, OPENROUTER_STRUCTURED_OUTPUTS, and JSON modes, got {mode}"
            )

        if provider in {Provider.ANYSCALE, Provider.TOGETHER}:
            assert mode in {
                Mode.TOOLS,
                Mode.JSON,
                Mode.JSON_SCHEMA,
                Mode.MD_JSON,
            }, (
                f"Anyscale/Together only support TOOLS, JSON, JSON_SCHEMA, and MD_JSON modes, got {mode}"
            )

        if provider in {Provider.OPENAI, Provider.DATABRICKS}:
            assert mode in {
                Mode.TOOLS,
                Mode.JSON,
                Mode.FUNCTIONS,
                Mode.PARALLEL_TOOLS,
                Mode.MD_JSON,
                Mode.TOOLS_STRICT,
                Mode.JSON_O1,
                Mode.RESPONSES_TOOLS,
                Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            }, f"OpenAI/Databricks provider does not support mode {mode}"

        # Special handling for RESPONSES modes
        if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:

            def map_chat_completion_to_response(messages, client, *args, **kwargs):
                return client.responses.create(*args, input=messages, **kwargs)

            async def async_map_chat_completion_to_response(
                messages, client, *args, **kwargs
            ):
                return await client.responses.create(*args, input=messages, **kwargs)

            if isinstance(client, openai.AsyncOpenAI):
                create_fn = partial(
                    async_map_chat_completion_to_response, client=client
                )
            else:
                create_fn = partial(map_chat_completion_to_response, client=client)
        else:
            create_fn = client.chat.completions.create

        # Patch and return appropriate Instructor type
        patched_create = patch(create=create_fn, mode=mode)

        if isinstance(client, openai.AsyncOpenAI):
            return AsyncInstructor(
                client=client,
                create=patched_create,
                mode=mode,
                provider=provider,
                **kwargs,
            )
        else:
            return Instructor(
                client=client,
                create=patched_create,
                mode=mode,
                provider=provider,
                **kwargs,
            )

    def get_mode_handlers(
        self,
    ) -> dict[
        Mode, Callable[[type[T], dict[str, Any]], tuple[type[T], dict[str, Any]]]
    ]:
        """Get all mode handlers for OpenAI."""
        return {
            Mode.TOOLS: self.handle_tools,
            Mode.TOOLS_STRICT: self.handle_tools_strict,
            Mode.FUNCTIONS: self.handle_functions,
            Mode.JSON: self.handle_json,
            Mode.JSON_SCHEMA: self.handle_json_schema,
            Mode.MD_JSON: self.handle_md_json,
            Mode.JSON_O1: self.handle_json_o1,
            Mode.PARALLEL_TOOLS: self.handle_parallel_tools,
            Mode.OPENROUTER_STRUCTURED_OUTPUTS: self.handle_openrouter_structured_outputs,
            Mode.RESPONSES_TOOLS: self.handle_responses_tools,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: self.handle_responses_tools_with_inbuilt_tools,
        }

    def handle_tools(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle TOOLS mode."""
        new_kwargs["tools"] = [
            {
                "type": "function",
                "function": response_model.openai_schema,
            }
        ]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": response_model.openai_schema["name"]},
        }
        return response_model, new_kwargs

    def handle_tools_strict(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle TOOLS_STRICT mode."""
        from openai import pydantic_function_tool

        response_model_schema = pydantic_function_tool(response_model)
        response_model_schema["function"]["strict"] = True
        new_kwargs["tools"] = [response_model_schema]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": response_model_schema["function"]["name"]},
        }
        return response_model, new_kwargs

    def handle_functions(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle FUNCTIONS mode (deprecated)."""
        Mode.warn_mode_functions_deprecation()
        new_kwargs["functions"] = [response_model.openai_schema]
        new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}
        return response_model, new_kwargs

    def handle_json(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle JSON mode."""
        return self._handle_json_modes(response_model, new_kwargs, Mode.JSON)

    def handle_json_schema(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle JSON_SCHEMA mode."""
        return self._handle_json_modes(response_model, new_kwargs, Mode.JSON_SCHEMA)

    def handle_md_json(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle MD_JSON mode."""
        return self._handle_json_modes(response_model, new_kwargs, Mode.MD_JSON)

    def _handle_json_modes(
        self, response_model: type[T], new_kwargs: dict[str, Any], mode: Mode
    ) -> tuple[type[T], dict[str, Any]]:
        """Common handler for JSON-based modes."""
        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        if mode == Mode.JSON:
            new_kwargs["response_format"] = {"type": "json_object"}
        elif mode == Mode.JSON_SCHEMA:
            new_kwargs["response_format"] = {
                "type": "json_object",
                "schema": response_model.model_json_schema(),
            }
        elif mode == Mode.MD_JSON:
            new_kwargs["messages"].append(
                {
                    "role": "user",
                    "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
                },
            )
            new_kwargs["messages"] = merge_consecutive_messages(new_kwargs["messages"])

        # Add or update system message
        if new_kwargs["messages"][0]["role"] != "system":
            new_kwargs["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": message,
                },
            )
        elif isinstance(new_kwargs["messages"][0]["content"], str):
            new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        elif isinstance(new_kwargs["messages"][0]["content"], list):
            new_kwargs["messages"][0]["content"][0]["text"] += f"\n\n{message}"
        else:
            raise ValueError(
                "Invalid message format, must be a string or a list of messages"
            )

        return response_model, new_kwargs

    def handle_json_o1(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle JSON_O1 mode for O1 models."""
        roles = [message["role"] for message in new_kwargs.get("messages", [])]
        if "system" in roles:
            raise ValueError("System messages are not supported for the O1 models")

        message = dedent(
            f"""
            Understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        new_kwargs["messages"].append(
            {
                "role": "user",
                "content": message,
            },
        )
        return response_model, new_kwargs

    def handle_parallel_tools(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle PARALLEL_TOOLS mode."""
        from instructor.dsl.parallel import ParallelModel, handle_parallel_model

        if new_kwargs.get("stream", False):
            from instructor.exceptions import ConfigurationError

            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )

        new_kwargs["tools"] = handle_parallel_model(response_model)
        new_kwargs["tool_choice"] = "auto"
        return ParallelModel(typehint=response_model), new_kwargs

    def handle_openrouter_structured_outputs(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OPENROUTER_STRUCTURED_OUTPUTS mode."""
        schema = response_model.model_json_schema()
        schema["additionalProperties"] = False
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        return response_model, new_kwargs

    def handle_responses_tools(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle RESPONSES_TOOLS mode."""
        from openai import pydantic_function_tool

        schema = pydantic_function_tool(response_model)
        del schema["function"]["strict"]

        tool_definition = {
            "type": "function",
            "name": schema["function"]["name"],
            "parameters": schema["function"]["parameters"],
        }

        if "description" in schema["function"]:
            tool_definition["description"] = schema["function"]["description"]
        else:
            tool_definition["description"] = (
                f"Correctly extracted `{response_model.__name__}` with all "
                f"the required parameters with correct types"
            )

        new_kwargs["tools"] = [
            {
                "type": "function",
                "name": schema["function"]["name"],
                "parameters": schema["function"]["parameters"],
            }
        ]

        new_kwargs["tool_choice"] = {
            "type": "function",
            "name": response_model.openai_schema["name"],
        }

        if new_kwargs.get("max_tokens") is not None:
            new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

        return response_model, new_kwargs

    def handle_responses_tools_with_inbuilt_tools(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle RESPONSES_TOOLS_WITH_INBUILT_TOOLS mode."""
        from openai import pydantic_function_tool

        schema = pydantic_function_tool(response_model)
        del schema["function"]["strict"]

        tool_definition = {
            "type": "function",
            "name": schema["function"]["name"],
            "parameters": schema["function"]["parameters"],
        }

        if "description" in schema["function"]:
            tool_definition["description"] = schema["function"]["description"]
        else:
            tool_definition["description"] = (
                f"Correctly extracted `{response_model.__name__}` with all "
                f"the required parameters with correct types"
            )

        if not new_kwargs.get("tools"):
            new_kwargs["tools"] = [tool_definition]
            new_kwargs["tool_choice"] = {
                "type": "function",
                "name": response_model.openai_schema["name"],
            }
        else:
            new_kwargs["tools"].append(tool_definition)

        if new_kwargs.get("max_tokens") is not None:
            new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

        return response_model, new_kwargs


# Register the provider
ProviderRegistry.register(OpenAIProvider())
