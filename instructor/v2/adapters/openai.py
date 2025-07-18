"""OpenAI provider adapter"""

import json
from typing import Any, Optional
from pydantic import BaseModel
from instructor.mode import Mode
from instructor.function_calls import (
    _handle_incomplete_output,
    _extract_text_content,
    _validate_model_from_json,
)
from instructor.utils import extract_json_from_codeblock, dump_message
from instructor.hooks import Hooks
from .base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API"""

    id = "openai"
    modes = {
        Mode.TOOLS,
        Mode.TOOLS_STRICT,
        Mode.JSON,
        Mode.JSON_SCHEMA,
        Mode.PARALLEL_TOOLS,
    }

    def build_request(
        self,
        response_model: type[BaseModel],
        mode: Mode,
        messages: list[dict[str, Any]],
        hooks: Hooks | None,
        **kwargs: Any,
    ) -> dict:
        """Build request for OpenAI API"""
        # If no response model, just pass through the messages
        if response_model is None:
            return {**kwargs, "messages": messages}

        if mode == Mode.TOOLS or mode == Mode.TOOLS_STRICT:
            schema = self.transform_schema(response_model.model_json_schema())
            return {
                **kwargs,
                "messages": messages,
                "tools": [{"type": "function", "function": schema}],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": schema["name"]},
                },
            }

        elif mode == Mode.JSON or mode == Mode.JSON_SCHEMA:
            return {
                **kwargs,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }

        return {**kwargs, "messages": messages}

    def handle_response_model(
        self, response_model: type[BaseModel], user_kwargs: dict
    ) -> tuple[type[BaseModel], dict]:
        """Handle response model for OpenAI"""
        return response_model, user_kwargs

    def handle_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        """Handle context for OpenAI"""
        return context or {}

    def initialize_usage(self, mode: Mode) -> Any:
        """Initialize usage tracking object for OpenAI"""
        from ..adapters.base import Usage

        return Usage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
            thinking_tokens=0,
        )

    def extract_usage(self, response: Any) -> Any:
        """Extract usage information from OpenAI response"""
        return getattr(response, "usage", None)

    def update_total_usage(self, response: Any, total_usage: Any) -> Any:
        """Update total usage with usage from current OpenAI response"""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                total_usage.prompt_tokens += usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                total_usage.completion_tokens += usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                total_usage.total_tokens += usage.total_tokens
            # OpenAI doesn't have thinking_tokens, so we leave it as is
        return response

    def process_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        mode: Mode,
        strict: bool | None,
        context: dict[str, Any] | None,
        hooks: Hooks | None,
    ) -> BaseModel:
        """Process OpenAI response"""
        # If no response model, return the raw response
        if response_model is None:
            return response

        if mode == Mode.TOOLS or mode == Mode.TOOLS_STRICT:
            return self._parse_tools(response_model, response, context, strict)
        elif mode == Mode.JSON or mode == Mode.JSON_SCHEMA:
            return self._parse_json(response_model, response, context, strict)

        raise ValueError(f"Unsupported mode: {mode}")

    def parse_response(
        self,
        response_model: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
        hooks: Hooks | None = None,
    ) -> BaseModel:
        """Parse OpenAI completion response"""
        mode = getattr(completion, "_mode", Mode.TOOLS)

        if mode == Mode.TOOLS or mode == Mode.TOOLS_STRICT:
            return self._parse_tools(
                response_model, completion, validation_context, strict
            )
        elif mode == Mode.JSON or mode == Mode.JSON_SCHEMA:
            return self._parse_json(
                response_model, completion, validation_context, strict
            )

        raise ValueError(f"Unsupported mode: {mode}")

    def build_reask_request(
        self, original_kwargs: dict, error: Exception, completion: Any, mode: Mode
    ) -> dict:
        """Build re-ask request for OpenAI after validation error"""
        kwargs = original_kwargs.copy()

        if mode in {Mode.TOOLS, Mode.TOOLS_STRICT}:
            # For tools mode, use tool response format
            reask_msgs = [self._format_assistant_message(completion)]

            # Add tool response messages for each tool call
            if hasattr(completion, "choices") and completion.choices:
                message = completion.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        reask_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": (
                                    f"Validation Error found:\n{error}\n"
                                    "Recall the function correctly, fix the errors"
                                ),
                            }
                        )

        else:
            # For JSON mode, use simple user message format
            reask_msgs = [self._format_assistant_message(completion)]
            reask_msgs.append(self._create_error_message(error, mode))

        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def _format_assistant_message(self, completion: Any) -> dict[str, Any]:
        """Format assistant message from OpenAI completion for re-asking"""
        if hasattr(completion, "choices") and completion.choices:
            return dump_message(completion.choices[0].message)
        else:
            # Fallback for unexpected completion format
            return {"role": "assistant", "content": str(completion)}

    def _parse_tools(
        self,
        response_model: type[BaseModel],
        response: Any,
        context: Optional[dict[str, Any]],
        strict: Optional[bool],
    ) -> BaseModel:
        """Parse tools response"""
        message = response.choices[0].message

        if hasattr(message, "refusal"):
            assert message.refusal is None, (
                f"Unable to generate a response due to {message.refusal}"
            )

        assert len(message.tool_calls or []) == 1, (
            "Instructor does not support multiple tool calls"
        )

        tool_call = message.tool_calls[0]
        assert (
            tool_call.function.name
            == self.transform_schema(response_model.model_json_schema())["name"]
        ), "Tool name does not match"

        return response_model.model_validate_json(
            tool_call.function.arguments, context=context, strict=strict
        )

    def _parse_json(
        self,
        response_model: type[BaseModel],
        response: Any,
        context: Optional[dict[str, Any]],
        strict: Optional[bool],
    ) -> BaseModel:
        """Parse JSON response"""
        _handle_incomplete_output(response)

        message = _extract_text_content(response)
        if not message:
            message = response.choices[0].message.content or ""

        json_content = extract_json_from_codeblock(message)
        return _validate_model_from_json(response_model, json_content, context, strict)

    def build_parallel_request(
        self, tool_schemas: list[dict], user_kwargs: dict
    ) -> dict:
        """Build parallel tools request for OpenAI"""
        tools = [{"type": "function", "function": schema} for schema in tool_schemas]
        return {
            **user_kwargs,
            "tools": tools,
            "tool_choice": "required" if len(tool_schemas) == 1 else "auto",
        }

    def parse_parallel_response(
        self, completion: Any, tool_models: list[type[BaseModel]]
    ) -> list[BaseModel]:
        """Parse parallel tool calling response"""
        results = []
        message = completion.choices[0].message

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                # Find the matching model by function name
                for model in tool_models:
                    schema = self.transform_schema(model.model_json_schema())
                    if schema["name"] == tool_call.function.name:
                        instance = model.model_validate_json(
                            tool_call.function.arguments
                        )
                        results.append(instance)
                        break

        return results

    def transform_schema(self, pydantic_schema: dict) -> dict:
        """Transform Pydantic schema to OpenAI function format"""
        parameters = {
            k: v
            for k, v in pydantic_schema.items()
            if k not in ("title", "description")
        }

        # Handle required fields
        if "required" not in parameters:
            parameters["required"] = sorted(
                k
                for k, v in parameters.get("properties", {}).items()
                if "default" not in v
            )

        return {
            "name": pydantic_schema.get("title", "UnknownFunction"),
            "description": pydantic_schema.get(
                "description", f"Extract {pydantic_schema.get('title', 'data')}"
            ),
            "parameters": parameters,
        }
