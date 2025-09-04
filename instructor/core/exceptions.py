from __future__ import annotations

from typing import Any


class InstructorError(Exception):
    """Base exception for all Instructor-specific errors."""

    pass


class IncompleteOutputException(InstructorError):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)


class InstructorRetryException(InstructorError):
    """Exception raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: int,
        create_kwargs: dict[str, Any] | None = None,
        all_exceptions: list[Exception] | None = None,
        all_failed_responses: list[Any] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        self.messages = messages
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        self.create_kwargs = create_kwargs
        self.all_exceptions = all_exceptions or []
        self.all_failed_responses = all_failed_responses or []
        super().__init__(*args, **kwargs)


class ValidationError(InstructorError):
    """Exception raised when response validation fails."""

    def __init__(
        self,
        message: str,
        *args: Any,
        failed_response: Any = None,
        raw_content: str | None = None,
        **kwargs: Any,
    ):
        self.failed_response = failed_response  # Original LLM response object
        self.raw_content = raw_content  # The actual content that failed to parse
        super().__init__(message, *args, **kwargs)


class InstructorJSONDecodeError(InstructorError):
    """JSON decode error with response context."""

    def __init__(
        self,
        message: str,
        *args: Any,
        failed_response: Any = None,
        raw_json_content: str | None = None,
        original_error: Exception | None = None,
        **kwargs: Any,
    ):
        self.failed_response = failed_response  # Original LLM response object
        self.raw_json_content = raw_json_content  # The malformed JSON string
        self.original_error = original_error  # Original JSONDecodeError
        super().__init__(message, *args, **kwargs)


class ProviderError(InstructorError):
    """Exception raised for provider-specific errors."""

    def __init__(self, provider: str, message: str, *args: Any, **kwargs: Any):
        self.provider = provider
        super().__init__(f"{provider}: {message}", *args, **kwargs)


class ConfigurationError(InstructorError):
    """Exception raised for configuration-related errors."""

    pass


class ModeError(InstructorError):
    """Exception raised when an invalid mode is used for a provider."""

    def __init__(
        self,
        mode: str,
        provider: str,
        valid_modes: list[str],
        *args: Any,
        **kwargs: Any,
    ):
        self.mode = mode
        self.provider = provider
        self.valid_modes = valid_modes
        message = f"Invalid mode '{mode}' for provider '{provider}'. Valid modes: {', '.join(valid_modes)}"
        super().__init__(message, *args, **kwargs)


class ClientError(InstructorError):
    """Exception raised for client initialization or usage errors."""

    pass


class AsyncValidationError(ValueError, InstructorError):
    """Exception raised during async validation."""

    errors: list[ValueError]
