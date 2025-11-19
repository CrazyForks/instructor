from __future__ import annotations

from json import JSONDecodeError
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, Retrying, RetryError

from ...core.exceptions import (
    AsyncValidationError,
    FailedAttempt,
    InstructorRetryException,
    ValidationError as InstructorValidationError,
)
from ...core.hooks import Hooks
from ...core.retry import (
    extract_messages,
    initialize_retrying,
    initialize_usage,
)
from ...utils import update_total_usage
from .handler import ModeHandler

T_Model = TypeVar("T_Model", bound=BaseModel)


def _handle_last_attempt(
    *,
    hooks: Hooks,
    attempt: Any,
    controller: Retrying | AsyncRetrying,
    error: Exception,
) -> None:
    if isinstance(controller, (Retrying, AsyncRetrying)) and hasattr(controller, "stop"):
        will_retry = (
            attempt.retry_state.outcome is None
            or not attempt.retry_state.outcome.failed
        )
        is_last_attempt = (
            not will_retry
            or attempt.retry_state.attempt_number
            >= getattr(
                controller.stop,
                "max_attempt_number",
                float("inf"),
            )
        )
        if is_last_attempt:
            hooks.emit_completion_last_attempt(error)


def retry_sync_v2(
    *,
    handler: ModeHandler,
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: dict[str, Any],
    context: dict[str, Any] | None,
    max_retries: int | Retrying,
    strict: bool | None,
    hooks: Hooks | None,
) -> Any:
    hooks = hooks or Hooks()
    total_usage = initialize_usage(handler.mode)
    timeout = kwargs.get("timeout")
    controller = initialize_retrying(
        max_retries,
        is_async=False,
        timeout=timeout,
    )
    stream = kwargs.get("stream", False)
    failed_attempts: list[FailedAttempt] = []

    try:
        response = None
        for attempt in controller:
            with attempt:
                try:
                    hooks.emit_completion_arguments(*args, **kwargs)
                    response = func(*args, **kwargs)
                    hooks.emit_completion_response(response)
                    response = update_total_usage(response=response, total_usage=total_usage)
                    parsed = handler.parse_response(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        stream=stream,
                        is_async=False,
                    )
                    return parsed
                except (
                    ValidationError,
                    JSONDecodeError,
                    AsyncValidationError,
                    InstructorValidationError,
                ) as error:
                    hooks.emit_parse_error(error)
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=error,
                            completion=response,
                        )
                    )
                    _handle_last_attempt(
                        hooks=hooks,
                        attempt=attempt,
                        controller=controller,
                        error=error,
                    )
                    kwargs = handler.handle_reask(
                        kwargs=kwargs,
                        response=response,
                        exception=error,
                        failed_attempts=failed_attempts,
                    )
                    raise error
                except Exception as error:
                    hooks.emit_completion_error(error)
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=error,
                            completion=response,
                        )
                    )
                    _handle_last_attempt(
                        hooks=hooks,
                        attempt=attempt,
                        controller=controller,
                        error=error,
                    )
                    raise error
    except RetryError as exc:
        raise InstructorRetryException(
            exc.last_attempt._exception,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            messages=extract_messages(kwargs),
            create_kwargs=kwargs,
            total_usage=total_usage,
            failed_attempts=failed_attempts,
        ) from exc


async def retry_async_v2(
    *,
    handler: ModeHandler,
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: dict[str, Any],
    context: dict[str, Any] | None,
    max_retries: int | AsyncRetrying,
    strict: bool | None,
    hooks: Hooks | None,
) -> Any:
    hooks = hooks or Hooks()
    total_usage = initialize_usage(handler.mode)
    timeout = kwargs.get("timeout")
    controller = initialize_retrying(
        max_retries,
        is_async=True,
        timeout=timeout,
    )
    stream = kwargs.get("stream", False)
    failed_attempts: list[FailedAttempt] = []

    try:
        response = None
        async for attempt in controller:
            with attempt:
                try:
                    hooks.emit_completion_arguments(*args, **kwargs)
                    response = await func(*args, **kwargs)
                    hooks.emit_completion_response(response)
                    response = update_total_usage(response=response, total_usage=total_usage)
                    parsed = handler.parse_response(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        stream=stream,
                        is_async=True,
                    )
                    return parsed
                except (
                    ValidationError,
                    JSONDecodeError,
                    AsyncValidationError,
                    InstructorValidationError,
                ) as error:
                    hooks.emit_parse_error(error)
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=error,
                            completion=response,
                        )
                    )
                    _handle_last_attempt(
                        hooks=hooks,
                        attempt=attempt,
                        controller=controller,
                        error=error,
                    )
                    kwargs = handler.handle_reask(
                        kwargs=kwargs,
                        response=response,
                        exception=error,
                        failed_attempts=failed_attempts,
                    )
                    raise error
                except Exception as error:
                    hooks.emit_completion_error(error)
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=error,
                            completion=response,
                        )
                    )
                    _handle_last_attempt(
                        hooks=hooks,
                        attempt=attempt,
                        controller=controller,
                        error=error,
                    )
                    raise error
    except RetryError as exc:
        raise InstructorRetryException(
            exc.last_attempt._exception,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            messages=extract_messages(kwargs),
            create_kwargs=kwargs,
            total_usage=total_usage,
            failed_attempts=failed_attempts,
        ) from exc

