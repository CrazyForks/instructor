from __future__ import annotations

from textwrap import dedent
from typing import Any, NamedTuple
from jinja2 import Template


class InstructorError(Exception):
    """Base exception for all Instructor-specific errors."""

    failed_attempts: list[FailedAttempt] | None = None

    @classmethod
    def from_exception(
        cls, exception: Exception, failed_attempts: list[FailedAttempt] | None = None
    ):
        return cls(str(exception), failed_attempts=failed_attempts)

    def __init__(
        self,
        *args: Any,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.failed_attempts = failed_attempts
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        # If no failed attempts, use the standard exception string representation
        if not self.failed_attempts:
            return super().__str__()

        template = Template(
            dedent(
                """
                <failed_attempts>
                {% for attempt in failed_attempts %}
                <generation number="{{ attempt.attempt_number }}">
                <exception>
                    {{ attempt.exception }}
                </exception>
                <completion>
                    {{ attempt.completion }}
                </completion>
                </generation>
                {% endfor %}
                </failed_attempts>

                <last_exception>
                    {{ last_exception }}
                </last_exception>
                """
            ).strip()
        )
        return template.render(
            last_exception=super().__str__(), failed_attempts=self.failed_attempts
        )


class FailedAttempt(NamedTuple):
    """Represents a single failed retry attempt."""

    attempt_number: int
    exception: Exception
    completion: Any | None = None


class IncompleteOutputException(InstructorError):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args: Any,
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
        *args: Any,
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: int,
        create_kwargs: dict[str, Any] | None = None,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        self.messages = messages
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        self.create_kwargs = create_kwargs
        super().__init__(*args, failed_attempts=failed_attempts, **kwargs)


class ValidationError(InstructorError):
    """Exception raised when response validation fails."""

    pass


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


class ResponseParsingError(ValueError, InstructorError):
    """Exception raised when unable to parse the LLM response.

    This exception occurs when the LLM's raw response cannot be parsed
    into the expected format. Common scenarios include:
    - Malformed JSON in JSON mode
    - Missing required fields in the response
    - Unexpected response structure
    - Invalid tool call format

    Note: This exception inherits from both ValueError and InstructorError
    to maintain backwards compatibility with code that catches ValueError.

    Attributes:
        mode: The mode being used when parsing failed
        raw_response: The raw response that failed to parse (if available)

    Examples:
        ```python
        try:
            response = client.chat.completions.create(
                response_model=User,
                mode=instructor.Mode.JSON,
                ...
            )
        except ResponseParsingError as e:
            print(f"Failed to parse response in {e.mode} mode")
            print(f"Raw response: {e.raw_response}")
            # May indicate the model doesn't support this mode well
        ```

        Backwards compatible with ValueError:
        ```python
        try:
            response = client.chat.completions.create(...)
        except ValueError as e:
            # Still catches ResponseParsingError
            print(f"Parsing error: {e}")
        ```
    """

    def __init__(
        self,
        message: str,
        *args: Any,
        mode: str | None = None,
        raw_response: Any | None = None,
        **kwargs: Any,
    ):
        self.mode = mode
        self.raw_response = raw_response
        context = f" (mode: {mode})" if mode else ""
        super().__init__(f"{message}{context}", *args, **kwargs)


class MultimodalError(ValueError, InstructorError):
    """Exception raised for multimodal content processing errors.

    This exception is raised when there are issues processing multimodal
    content (images, audio, PDFs, etc.), such as:
    - Unsupported file formats
    - File not found
    - Invalid base64 encoding
    - Provider doesn't support multimodal content

    Note: This exception inherits from both ValueError and InstructorError
    to maintain backwards compatibility with code that catches ValueError.

    Attributes:
        content_type: The type of content that failed (e.g., 'image', 'audio', 'pdf')
        file_path: The file path if applicable

    Examples:
        ```python
        from instructor import Image

        try:
            response = client.chat.completions.create(
                response_model=Analysis,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image"},
                        Image.from_path("/invalid/path.jpg")
                    ]
                }]
            )
        except MultimodalError as e:
            print(f"Multimodal error with {e.content_type}: {e}")
            if e.file_path:
                print(f"File path: {e.file_path}")
        ```

        Backwards compatible with ValueError:
        ```python
        try:
            img = Image.from_path("/path/to/image.jpg")
        except ValueError as e:
            # Still catches MultimodalError
            print(f"Image error: {e}")
        ```
    """

    def __init__(
        self,
        message: str,
        *args: Any,
        content_type: str | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ):
        self.content_type = content_type
        self.file_path = file_path
        context_parts = []
        if content_type:
            context_parts.append(f"content_type: {content_type}")
        if file_path:
            context_parts.append(f"file_path: {file_path}")
        context = f" ({', '.join(context_parts)})" if context_parts else ""
        super().__init__(f"{message}{context}", *args, **kwargs)
