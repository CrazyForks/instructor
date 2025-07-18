"""Auto Client using Adapter Pattern"""

from typing import Any, Optional
import sys
import os
import openai

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from instructor.mode import Mode
from .client import InstructorClient
from .adapters.registry import registry


def from_provider(
    provider: str, mode: Optional[Mode] = None, **kwargs: Any
) -> InstructorClient:
    """Create an instructor client from a provider string"""

    # Parse provider string (e.g., "openai/gpt-4o-mini")
    if "/" in provider:
        provider_name, model = provider.split("/", 1)
    else:
        provider_name = provider
        model = None

    assert provider_name == "openai", "Only OpenAI is supported for now"

    # Default mode for OpenAI
    if mode is None:
        mode = Mode.TOOLS

    # Get adapter
    adapter = registry.get_adapter_by_id(provider_name)
    if adapter is None:
        raise ValueError(f"No adapter found for provider: {provider_name}")

    client = openai.OpenAI()

    # Return wrapped client
    return InstructorClient(
        client=client,
        create=client.chat.completions.create,
        provider=provider,
        mode=mode,
        model=model,
        adapter=adapter,
    )
