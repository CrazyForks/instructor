from .mode import Mode
from .process_response import handle_response_model
from .distil import FinetuneFormat, Instructions
from .multimodal import Image, Audio
from .dsl import (
    CitationMixin,
    Maybe,
    Partial,
    IterableModel,
    llm_validator,
    openai_moderation,
)
from .function_calls import OpenAISchema, openai_schema
from .patch import apatch, patch
from .process_response import handle_parallel_model
from .client import (
    Instructor,
    AsyncInstructor,
    from_openai,
    from_litellm,
    Provider,
)
from .auto_client import from_provider

# Import all available client functions
from . import clients

__all__ = [
    "Instructor",
    "Image",
    "Audio",
    "from_openai",
    "from_litellm",
    "from_provider",
    "AsyncInstructor",
    "Provider",
    "OpenAISchema",
    "CitationMixin",
    "IterableModel",
    "Maybe",
    "Partial",
    "openai_schema",
    "Mode",
    "patch",
    "apatch",
    "llm_validator",
    "openai_moderation",
    "FinetuneFormat",
    "Instructions",
    "handle_parallel_model",
    "handle_response_model",
]

# Add all available client functions to this module's namespace and __all__
for name in clients.__all__:
    if (
        name != "from_openai"
    ):  # Skip from_openai since we import it directly from .client
        globals()[name] = getattr(clients, name)
        __all__.append(name)
