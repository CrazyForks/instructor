"""Base adapter class for provider implementations"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union
from pydantic import BaseModel
from instructor import process_response
from instructor.mode import Mode
from instructor.hooks import Hooks

T_Model = TypeVar("T_Model", bound=BaseModel)


class Usage(BaseModel):
    """Usage tracking object"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: int


class ProviderAdapter(ABC):
    """Base class for provider-specific adapters"""

    id: str  # e.g. "openai"
    modes: set[Mode]  # modes this adapter understands

    # Core request/response handling
    @abstractmethod
    def build_request(
        self,
        response_model: type[BaseModel],
        mode: Mode,
        messages: list[dict[str, Any]],
        hooks: Hooks | None,
        **kwargs: Any,
    ) -> dict:
        """Build request arguments for this provider"""
        ...

    @abstractmethod
    def process_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        mode: Mode,
        strict: bool | None,
        context: dict[str, Any] | None,
        hooks: Hooks | None,
    ) -> BaseModel:
        """Process response"""
        ...

    @abstractmethod
    def parse_response(
        self,
        response_model: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
        hooks: Hooks | None = None,
    ) -> BaseModel:
        """Parse completion response into validated model"""
        ...

    # Usage tracking methods
    @abstractmethod
    def initialize_usage(self, mode: Mode) -> Usage:
        """Initialize usage tracking object for this provider"""
        return Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            thinking_tokens=0,
        )

    @abstractmethod
    def extract_usage(self, response: Any) -> Usage:
        """Extract usage information from response"""
        ...

    @abstractmethod
    def update_total_usage(self, response: Any, total_usage: Usage) -> Usage:
        """Update total usage with usage from current response"""
        ...

    # Parallel tool calling support
    @abstractmethod
    def build_parallel_request(
        self, tool_schemas: list[dict], user_kwargs: dict
    ) -> dict:
        """Build request for parallel tool calling (PARALLEL_TOOLS modes)"""
        raise NotImplementedError(f"{self.id} does not support parallel tools")

    @abstractmethod
    def parse_parallel_response(
        self, completion: Any, tool_models: list[type[BaseModel]]
    ) -> list[BaseModel]:
        """Parse parallel tool calling response"""
        raise NotImplementedError(f"{self.id} does not support parallel tools")

    # Re-asking/retry support
    @abstractmethod
    def build_reask_request(
        self, original_kwargs: dict, error: Exception, completion: Any, mode: Mode
    ) -> dict:
        """Build re-ask request after validation error"""
        # Each adapter must implement provider-specific re-asking logic
        ...

    @abstractmethod
    def _format_assistant_message(self, completion: Any) -> dict[str, Any]:
        """Format assistant message from completion for re-asking"""
        ...

    def _create_error_message(self, error: Exception, mode: Mode) -> dict[str, Any]:
        """Create error message for re-asking (common logic)"""
        if mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
            return {
                "role": "user",
                "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{error}",
            }
        else:
            return {
                "role": "user",
                "content": f"Recall the function correctly, fix the errors, exceptions found\n{error}",
            }

    # Schema transformation
    @abstractmethod
    def transform_schema(self, pydantic_schema: dict) -> dict:
        """Transform Pydantic schema to provider-specific format"""
        ...

    # Provider capabilities
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses"""
        return True  # Most providers support streaming

    def supports_parallel_tools(self) -> bool:
        """Whether this provider supports parallel tool calling"""
        return Mode.PARALLEL_TOOLS in self.modes or any(
            "PARALLEL" in str(mode) for mode in self.modes
        )
