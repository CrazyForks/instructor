"""V2 Instructor Client using Adapter Pattern"""

from typing import Any, Callable, Optional, TypeVar
from cohere import Usage
from pydantic import BaseModel, ValidationError
import sys
import os
import logging
from json import JSONDecodeError
from jinja2 import Template

from instructor.mode import Mode
from instructor.hooks import Hooks
from instructor.exceptions import InstructorRetryException
from tenacity import Retrying, RetryError, stop_after_attempt, stop_after_delay
from .adapters.base import ProviderAdapter

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("instructor")


class InstructorClient:
    """Instructor client using adapter pattern"""

    def __init__(
        self,
        client: Any,
        create: Callable,
        provider: str,
        mode: Mode,
        model: Optional[str] = None,
        adapter: Optional[ProviderAdapter] = None,
    ):
        self.client = client
        self.provider = provider
        self.mode = mode
        self.model = model
        self.adapter = adapter
        self._create_func = create

    def _initialize_retrying(
        self, max_retries: int | Retrying, timeout: float | None = None
    ):
        """Initialize retrying mechanism"""
        if isinstance(max_retries, int):
            stop_conditions = [stop_after_attempt(max_retries)]
            if timeout is not None:
                stop_conditions.append(stop_after_delay(timeout))

            stop_condition = stop_conditions[0]
            for condition in stop_conditions[1:]:
                stop_condition = stop_condition | condition

            return Retrying(stop=stop_condition)
        return max_retries

    def _initialize_usage(self) -> Usage:
        """Initialize usage tracking"""
        return Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            thinking_tokens=0,
        )

    def format_messages_with_context(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        # treats content of every message as a jinja2 template
        # and renders it with the context
        if context is None:
            return messages
        
        # Create a deep copy to avoid modifying the original messages
        import copy
        rendered_messages = copy.deepcopy(messages)
        
        for message in rendered_messages:
            if "content" in message and isinstance(message["content"], str):
                message["content"] = Template(message["content"]).render(**context)
        return rendered_messages

    def create(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        max_retries: int | Retrying = 1,
        strict: bool | None = None,
        context: dict[str, Any] | None = None,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> T:
        """Create a completion with structured output with retry logic"""

        # Initialize hooks if not provided
        hooks = hooks or Hooks()

        # Initialize retrying
        timeout = kwargs.get("timeout")
        retrying = self._initialize_retrying(max_retries, timeout=timeout)

        # Initialize total usage tracking
        total_usage = self.adapter.initialize_usage(self.mode)

        # Current kwargs for the request
        current_kwargs = {"messages": messages, **kwargs}

        try:
            response = None
            for attempt in retrying:
                with attempt:
                    logger.debug(
                        f"Retrying, attempt: {attempt.retry_state.attempt_number}"
                    )
                    try:
                        # Build request using adapter
                        request_kwargs = self.adapter.build_request(
                            response_model,
                            mode=self.mode,
                            messages=current_kwargs.get("messages", messages),
                            hooks=hooks,
                            **{
                                k: v
                                for k, v in current_kwargs.items()
                                if k != "messages"
                            },
                        )

                        # Add model if specified
                        if self.model:
                            request_kwargs["model"] = self.model

                        # Emit completion arguments
                        hooks.emit_completion_arguments(request_kwargs)

                        # Format messages with context
                        request_kwargs["messages"] = self.format_messages_with_context(
                            request_kwargs["messages"], context
                        )

                        # Make the actual API call
                        response = self._create_func(**request_kwargs)

                        # Emit completion response
                        hooks.emit_completion_response(response)

                        # Update usage tracking
                        response = self.adapter.update_total_usage(
                            response, total_usage
                        )

                        # Process response using adapter
                        return self.adapter.process_response(
                            response,
                            response_model=response_model,
                            mode=self.mode,
                            strict=strict,
                            context=context,
                            hooks=hooks,
                        )

                    except (ValidationError, JSONDecodeError) as e:
                        logger.debug(f"Parse error: {e}")
                        hooks.emit_parse_error(e)

                        # Use adapter's build_reask_request for re-asking
                        current_kwargs = self.adapter.build_reask_request(
                            original_kwargs=current_kwargs,
                            error=e,
                            completion=response,
                            mode=self.mode,
                        )
                        raise e

        except RetryError as e:
            logger.debug(f"Retry error: {e}")
            raise InstructorRetryException(
                e.last_attempt._exception,
                last_completion=response,
                n_attempts=attempt.retry_state.attempt_number,
                messages=current_kwargs.get("messages", []),
                create_kwargs=current_kwargs,
                total_usage=total_usage,
            ) from e
