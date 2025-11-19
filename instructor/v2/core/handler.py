from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from ...mode import Mode
from ...utils.providers import Provider

T_Model = TypeVar("T_Model", bound=BaseModel)


@dataclass
class ModeHandler:
    """Base class for provider/mode specific request and response handlers."""

    provider: Provider
    mode: Mode

    def prepare_request(
        self,
        response_model: type[T_Model] | None,
        **kwargs: Any,
    ) -> tuple[type[T_Model] | None, dict[str, Any]]:
        """Prepare provider specific kwargs before the API call."""
        raise NotImplementedError

    def parse_response(
        self,
        response: Any,
        *,
        response_model: type[T_Model] | None,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
        stream: bool,
        is_async: bool,
    ) -> T_Model | Any:
        """Parse a provider response into the requested response model."""
        raise NotImplementedError

    def handle_reask(
        self,
        *,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
        failed_attempts: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Modify kwargs with validation error feedback before a retry."""
        raise NotImplementedError

