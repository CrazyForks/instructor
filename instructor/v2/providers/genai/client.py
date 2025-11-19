from __future__ import annotations

from typing import Any, Literal, overload

from google.genai import Client

from ....core.client import AsyncInstructor, Instructor
from ....core.exceptions import ClientError, ModeError
from ....mode import Mode
from ....utils.providers import Provider
from ...core.patch import patch_v2
from ...core.registry import normalize_mode

VALID_MODES = {
    Mode.TOOLS,
    Mode.JSON,
    # Backwards compatibility
    Mode.GENAI_TOOLS,
    Mode.GENAI_JSON,
    Mode.GENAI_STRUCTURED_OUTPUTS,
}


@overload
def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: Literal[False],
    **kwargs: Any,
) -> Instructor: ...


def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """
    Create a v2 Instructor client from a google.genai.Client instance.
    
    Supports generic modes (TOOLS, JSON) and backwards-compatible provider-specific modes
    (GENAI_TOOLS, GENAI_JSON, GENAI_STRUCTURED_OUTPUTS).
    """

    if mode not in VALID_MODES:
        raise ModeError(
            mode=str(mode),
            provider="GenAI",
            valid_modes=[str(m) for m in VALID_MODES],
        )

    if not isinstance(client, Client):
        raise ClientError(
            f"Client must be an instance of google.genai.Client. Got: {type(client).__name__}"
        )

    # Normalize mode for handler lookup (preserve original for client)
    normalized_mode = normalize_mode(Provider.GENAI, mode)

    if use_async:

        async def async_wrapper(*args: Any, **call_kwargs: Any) -> Any:
            if call_kwargs.pop("stream", False):
                return await client.aio.models.generate_content_stream(*args, **call_kwargs)  # type: ignore[attr-defined]
            return await client.aio.models.generate_content(*args, **call_kwargs)  # type: ignore[attr-defined]

        patched = patch_v2(
            create=async_wrapper,
            provider=Provider.GENAI,
            mode=normalized_mode,
        )
        return AsyncInstructor(
            client=client,
            create=patched,
            provider=Provider.GENAI,
            mode=mode,  # Keep original mode for client
            **kwargs,
        )

    def sync_wrapper(*args: Any, **call_kwargs: Any) -> Any:
        if call_kwargs.pop("stream", False):
            return client.models.generate_content_stream(*args, **call_kwargs)
        return client.models.generate_content(*args, **call_kwargs)

    patched = patch_v2(
        create=sync_wrapper,
        provider=Provider.GENAI,
        mode=normalized_mode,
    )
    return Instructor(
        client=client,
        create=patched,
        provider=Provider.GENAI,
        mode=mode,  # Keep original mode for client
        **kwargs,
    )

