"""Base classes and protocols for response handlers."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class ResponseHandler(Protocol):
    """Protocol for response handlers."""

    def prepare_request(
        self, response_model: type[T] | None, kwargs: dict[str, Any]
    ) -> tuple[type[T] | None, dict[str, Any]]:
        """Prepare and format the API request parameters for the provider.

        This method takes the response model and raw kwargs and formats them
        according to the provider's specific requirements (e.g., adding tools,
        formatting schemas, setting tool_choice parameters).

        Args:
            response_model: The Pydantic model to use for parsing the response
            kwargs: The raw API call parameters

        Returns:
            Tuple of (processed response_model, formatted kwargs for API call)
        """
        ...

    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format the retry request with validation error feedback.

        When validation fails, this method prepares a new request that includes
        information about the error, allowing the LLM to understand what went
        wrong and correct its response.

        Args:
            kwargs: The original request parameters
            response: The raw response that failed validation
            exception: The validation error that occurred

        Returns:
            Modified kwargs for the retry request with error context
        """
        ...


class ResponseHandlerBase(ABC):
    """Base class for response handlers with common functionality."""

    @abstractmethod
    def prepare_request(
        self, response_model: type[T] | None, kwargs: dict[str, Any]
    ) -> tuple[type[T] | None, dict[str, Any]]:
        """Prepare and format the API request parameters for the provider."""
        pass

    @abstractmethod
    def format_retry_request(
        self, kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        """Format the retry request with validation error feedback."""
        pass

    def _copy_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Create a shallow copy of kwargs."""
        return kwargs.copy()
