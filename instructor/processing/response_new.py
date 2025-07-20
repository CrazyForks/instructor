"""
Refactored response processing module using the handler registry pattern.

This module demonstrates how the new handler registry system would work,
providing cleaner separation of concerns and lazy loading of provider-specific code.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, TypeVar, TYPE_CHECKING

from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing_extensions import ParamSpec

from ..dsl.iterable import IterableBase
from ..dsl.parallel import ParallelBase
from ..dsl.partial import PartialBase
from ..dsl.simple_type import AdapterBase

if TYPE_CHECKING:
    from .function_calls import OpenAISchema
from ..mode import Mode
from ..utils.core import prepare_response_model
from .handlers import handler_registry

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """Asynchronously process and transform LLM responses into structured models.

    This function is the async entry point for converting raw LLM responses into validated
    Pydantic models. It handles various response formats from different providers and
    supports special response types like streaming, partial objects, and parallel tool calls.

    Args:
        response (ChatCompletion or Similar API Response): The raw response from the LLM API. Despite the type hint,
            this can be responses from any supported provider (OpenAI, Anthropic, Google, etc.)
        response_model (type[T_Model | BaseModel] | None): The target Pydantic
            model to parse the response into. If None, returns the raw response unchanged.
            Can also be special DSL types like ParallelBase for parallel tool calls, or IterableBase and PartialBase for streaming.
        stream (bool): Whether this is a streaming response. Required for proper handling
            of IterableBase and PartialBase models. Defaults to False.
        validation_context (dict[str, Any] | None): Additional context passed to Pydantic
            validators during model validation. Useful for dynamic validation logic. The context
            is also used to format templated responses. Defaults to None.
        strict (bool | None): Whether to enforce strict JSON parsing. When True, the response
            must exactly match the model schema. When False, allows minor deviations.
        mode (Mode): The provider/format mode that determines how to parse the response.
            Examples: Mode.TOOLS (OpenAI), Mode.ANTHROPIC_JSON, Mode.GEMINI_TOOLS.
            Defaults to Mode.TOOLS.

    Returns:
        T_Model | ChatCompletion: The processed response. Return type depends on inputs:
            - If response_model is None: returns raw response unchanged
            - If response_model is IterableBase with stream=True: returns list of models
            - If response_model is AdapterBase: returns the adapted content
            - Otherwise: returns instance of response_model with _raw_response attached

    Raises:
        ValidationError: If the response doesn't match the expected model schema
        IncompleteOutputException: If the response was truncated due to token limits
        ValueError: If an invalid mode is specified

    Note:
        The function automatically detects special response model types (Iterable, Partial,
        Parallel, Adapter) and applies appropriate processing logic for each.
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | None:
    """Process and transform LLM responses into structured models (synchronous).

    This is the main entry point for converting raw LLM responses into validated Pydantic
    models. It acts as a dispatcher that handles various response formats from 40+ different
    provider modes and transforms them according to the specified response model type.

    Args:
        response (T_Model): The raw response from the LLM API. The actual type varies by
            provider (ChatCompletion for OpenAI, Message for Anthropic, etc.)
        response_model (type[OpenAISchema | BaseModel] | None): The target Pydantic model
            class to parse the response into. Special DSL types supported:
            - IterableBase: For streaming multiple objects from a single response
            - PartialBase: For incomplete/streaming partial objects
            - ParallelBase: For parallel tool/function calls
            - AdapterBase: For simple type adaptations (e.g., str, int)
            If None, returns the raw response unchanged.
        stream (bool): Whether this is a streaming response. Required to be True for
            proper handling of IterableBase and PartialBase models.
        validation_context (dict[str, Any] | None): Additional context passed to Pydantic
            validators. Useful for runtime validation logic based on external state.
        strict (bool | None): Controls JSON parsing strictness:
            - True: Enforce exact schema matching (no extra fields)
            - False/None: Allow minor deviations and extra fields
        mode (Mode): The provider/format mode that determines parsing strategy.
            Each mode corresponds to a specific provider and format combination:
            - Tool modes: TOOLS, ANTHROPIC_TOOLS, GEMINI_TOOLS, etc.
            - JSON modes: JSON, ANTHROPIC_JSON, VERTEXAI_JSON, etc.
            - Special modes: PARALLEL_TOOLS, MD_JSON, JSON_SCHEMA, etc.

    Returns:
        T_Model | list[T_Model] | None: The processed response:
            - If response_model is None: Original response unchanged
            - If IterableBase: List of extracted model instances
            - If ParallelBase: Special parallel response object
            - If AdapterBase: The adapted simple type (str, int, etc.)
            - Otherwise: Single instance of response_model with _raw_response attached

    Raises:
        ValidationError: Response doesn't match the expected model schema
        IncompleteOutputException: Response truncated due to token limits
        ValueError: Invalid mode specified or mode not supported
        JSONDecodeError: Malformed JSON in response (for JSON modes)

    Note:
        The function preserves the raw response by attaching it to the parsed model
        as `_raw_response`. This allows access to metadata like token usage, model
        info, and other provider-specific fields after parsing.
    """
    logger.debug(
        f"Instructor Raw Response: {response}",
    )

    if response_model is None:
        logger.debug("No response model, returning response as is")
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response

    return model


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def handle_response_model(
    response_model: type[T] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[T] | None, dict[str, Any]]:
    """
    Prepares the response model and API request parameters based on the specified mode.

    This function uses the handler registry to get the appropriate handler for the mode
    and calls its prepare_request method to format the API call parameters.

    Args:
        response_model (type[T] | None): The response model to be used for parsing the API response.
        mode (Mode): The mode to use for handling the response model. Defaults to Mode.TOOLS.
        **kwargs: Additional keyword arguments to be passed to the API call.

    Returns:
        tuple[type[T] | None, dict[str, Any]]: A tuple containing the processed response model and the updated kwargs.

    This function prepares the response model and modifies the kwargs based on the specified mode.
    It handles various modes like TOOLS, JSON, FUNCTIONS, etc., and applies the appropriate
    transformations to the response model and kwargs.
    """

    new_kwargs = kwargs.copy()

    # Only prepare response_model if it's not None
    if response_model is not None:
        response_model = prepare_response_model(response_model)

    # Use the handler registry to get the appropriate handler
    try:
        handler = handler_registry.get_handler(mode)
        response_model, new_kwargs = handler.prepare_request(response_model, new_kwargs)
    except ValueError as e:
        # If no handler is found, raise an error
        raise ValueError(f"Invalid patch mode: {mode}") from e

    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": (
                response_model.__name__
                if response_model is not None and hasattr(response_model, "__name__")
                else str(response_model)
            ),
            "new_kwargs": new_kwargs,
        },
    )
    return response_model, new_kwargs


def handle_reask_kwargs(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle validation errors by reformatting the request for retry (reask).

    When a response fails validation (e.g., missing required fields, wrong types),
    this function prepares a new request that includes information about the error.
    This allows the LLM to understand what went wrong and correct its response.

    Uses the handler registry to get the appropriate handler for the mode and calls
    its format_retry_request method.

    Args:
        kwargs (dict[str, Any]): The original request parameters that resulted in
            a validation error. Includes messages, tools, temperature, etc.
        mode (Mode): The provider/format mode that determines which reask handler
            to use. Each mode has a specific strategy for formatting error feedback.
        response (Any): The raw response from the LLM that failed validation.
            Type varies by provider:
            - OpenAI: ChatCompletion with tool_calls
            - Anthropic: Message with tool_use blocks
            - Google: GenerateContentResponse with function calls
        exception (Exception): The validation error that occurred. Usually a
            Pydantic ValidationError with details about which fields failed.

    Returns:
        dict[str, Any]: Modified kwargs for the retry request, typically including:
            - Updated messages with error context
            - Same tool/function definitions
            - Preserved generation parameters
            - Provider-specific formatting

    Reask Strategies by Provider:
        Each provider has a specific strategy for handling retries:

        **JSON Modes:**
        - Adds assistant message with failed attempt
        - Adds user message with error details

        **Tool Calls:**
        - Preserves tool definitions
        - Formats the errors as tool calls responses

    Note:
        This function is typically called internally by the retry logic when
        max_retries > 1. It ensures that each retry attempt includes context
        about previous failures, helping the LLM learn from its mistakes.
    """
    # Create a shallow copy of kwargs to avoid modifying the original
    kwargs_copy = kwargs.copy()

    # Use the handler registry to get the appropriate handler
    try:
        handler = handler_registry.get_handler(mode)
        return handler.format_retry_request(kwargs_copy, response, exception)
    except ValueError:
        # If no handler is found, use the default handler
        from .handlers.openai import DefaultHandler

        default_handler = DefaultHandler()
        return default_handler.format_retry_request(kwargs_copy, response, exception)
