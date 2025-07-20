"""Handler registry with lazy loading for optional dependencies."""

import importlib.util
import logging
from typing import Callable, Optional

from ...mode import Mode
from .base import ResponseHandler

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """Registry for response handlers with lazy loading."""

    def __init__(self):
        self._handler_factories: dict[Mode, Callable[[], ResponseHandler]] = {}
        self._loaded_handlers: dict[Mode, ResponseHandler] = {}
        self._mode_to_module: dict[Mode, tuple[str, str, Optional[str]]] = {}

    def register_lazy(
        self,
        mode: Mode,
        module_path: str,
        handler_class_name: str,
        required_spec: Optional[str] = None,
    ):
        """Register a handler factory that loads only when needed.

        Args:
            mode: The mode to register the handler for
            module_path: The module path relative to instructor.processing.handlers
            handler_class_name: The name of the handler class in the module
            required_spec: Optional package spec to check (e.g., 'anthropic')
        """
        self._mode_to_module[mode] = (module_path, handler_class_name, required_spec)

        def factory():
            # Check if required package is available
            if required_spec and not importlib.util.find_spec(required_spec):
                raise ImportError(
                    f"Package '{required_spec}' is required for mode {mode} but not installed"
                )

            # Import the module and get the handler class
            full_module_path = f"instructor.processing.handlers.{module_path}"
            module = importlib.import_module(full_module_path)
            handler_class = getattr(module, handler_class_name)

            return handler_class()

        self._handler_factories[mode] = factory

    def register(self, mode: Mode, handler: ResponseHandler):
        """Register a handler instance directly."""
        self._loaded_handlers[mode] = handler

    def get_handler(self, mode: Mode) -> ResponseHandler:
        """Get handler instance for mode, loading it if necessary."""
        # Check if already loaded
        if mode in self._loaded_handlers:
            return self._loaded_handlers[mode]

        # Try to load from factory
        if mode in self._handler_factories:
            try:
                handler = self._handler_factories[mode]()
                self._loaded_handlers[mode] = handler
                logger.debug(f"Loaded handler for mode {mode}")
                return handler
            except ImportError as e:
                logger.warning(f"Failed to load handler for mode {mode}: {e}")
                # Fall back to default handler
                return self._get_default_handler()

        # No handler registered for this mode
        raise ValueError(f"No handler registered for mode: {mode}")

    def _get_default_handler(self) -> ResponseHandler:
        """Get the default fallback handler."""
        from .openai import DefaultHandler

        return DefaultHandler()

    def has_handler(self, mode: Mode) -> bool:
        """Check if a handler is available for the mode."""
        if mode in self._loaded_handlers:
            return True

        if mode in self._mode_to_module:
            _, _, required_spec = self._mode_to_module[mode]
            if required_spec:
                return importlib.util.find_spec(required_spec) is not None
            return True

        return False


# Global registry instance
handler_registry = HandlerRegistry()


def _register_all_handlers():
    """Register all available handlers based on installed packages."""

    # OpenAI handlers (always available as it's a core dependency)
    handler_registry.register_lazy(Mode.TOOLS, "openai", "ToolsHandler")
    handler_registry.register_lazy(Mode.TOOLS_STRICT, "openai", "ToolsStrictHandler")
    handler_registry.register_lazy(Mode.FUNCTIONS, "openai", "FunctionsHandler")
    handler_registry.register_lazy(Mode.JSON, "openai", "JSONHandler")
    handler_registry.register_lazy(Mode.MD_JSON, "openai", "MDJSONHandler")
    handler_registry.register_lazy(Mode.JSON_SCHEMA, "openai", "JSONSchemaHandler")
    handler_registry.register_lazy(Mode.JSON_O1, "openai", "JSONO1Handler")
    handler_registry.register_lazy(
        Mode.PARALLEL_TOOLS, "openai", "ParallelToolsHandler"
    )
    handler_registry.register_lazy(
        Mode.RESPONSES_TOOLS, "openai", "ResponsesToolsHandler"
    )
    handler_registry.register_lazy(
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        "openai",
        "ResponsesToolsWithInbuiltHandler",
    )
    handler_registry.register_lazy(
        Mode.OPENROUTER_STRUCTURED_OUTPUTS, "openai", "OpenRouterHandler"
    )

    # Anthropic handlers
    handler_registry.register_lazy(
        Mode.ANTHROPIC_TOOLS, "anthropic", "AnthropicToolsHandler", "anthropic"
    )
    handler_registry.register_lazy(
        Mode.ANTHROPIC_REASONING_TOOLS,
        "anthropic",
        "AnthropicReasoningToolsHandler",
        "anthropic",
    )
    handler_registry.register_lazy(
        Mode.ANTHROPIC_JSON, "anthropic", "AnthropicJSONHandler", "anthropic"
    )
    handler_registry.register_lazy(
        Mode.ANTHROPIC_PARALLEL_TOOLS,
        "anthropic",
        "AnthropicParallelToolsHandler",
        "anthropic",
    )

    # Mistral handlers
    handler_registry.register_lazy(
        Mode.MISTRAL_TOOLS, "mistral", "MistralToolsHandler", "mistralai"
    )
    handler_registry.register_lazy(
        Mode.MISTRAL_STRUCTURED_OUTPUTS,
        "mistral",
        "MistralStructuredOutputsHandler",
        "mistralai",
    )

    # Cohere handlers
    handler_registry.register_lazy(
        Mode.COHERE_TOOLS, "cohere", "CohereToolsHandler", "cohere"
    )
    handler_registry.register_lazy(
        Mode.COHERE_JSON_SCHEMA, "cohere", "CohereJSONSchemaHandler", "cohere"
    )

    # Gemini/Google handlers
    handler_registry.register_lazy(
        Mode.GEMINI_JSON, "gemini", "GeminiJSONHandler", "google.generativeai"
    )
    handler_registry.register_lazy(
        Mode.GEMINI_TOOLS, "gemini", "GeminiToolsHandler", "google.generativeai"
    )
    handler_registry.register_lazy(
        Mode.GENAI_TOOLS, "genai", "GenAIToolsHandler", "google.genai"
    )
    handler_registry.register_lazy(
        Mode.GENAI_STRUCTURED_OUTPUTS,
        "genai",
        "GenAIStructuredOutputsHandler",
        "google.genai",
    )

    # VertexAI handlers
    handler_registry.register_lazy(
        Mode.VERTEXAI_TOOLS, "vertexai", "VertexAIToolsHandler", "vertexai"
    )
    handler_registry.register_lazy(
        Mode.VERTEXAI_JSON, "vertexai", "VertexAIJSONHandler", "vertexai"
    )
    handler_registry.register_lazy(
        Mode.VERTEXAI_PARALLEL_TOOLS,
        "vertexai",
        "VertexAIParallelToolsHandler",
        "vertexai",
    )

    # Cerebras handlers
    handler_registry.register_lazy(
        Mode.CEREBRAS_JSON, "cerebras", "CerebrasJSONHandler", "cerebras"
    )
    handler_registry.register_lazy(
        Mode.CEREBRAS_TOOLS, "cerebras", "CerebrasToolsHandler", "cerebras"
    )

    # Fireworks handlers
    handler_registry.register_lazy(
        Mode.FIREWORKS_JSON, "fireworks", "FireworksJSONHandler", "fireworks"
    )
    handler_registry.register_lazy(
        Mode.FIREWORKS_TOOLS, "fireworks", "FireworksToolsHandler", "fireworks"
    )

    # Writer handlers
    handler_registry.register_lazy(
        Mode.WRITER_JSON, "writer", "WriterJSONHandler", "writerai"
    )
    handler_registry.register_lazy(
        Mode.WRITER_TOOLS, "writer", "WriterToolsHandler", "writerai"
    )

    # Bedrock handlers
    handler_registry.register_lazy(
        Mode.BEDROCK_JSON, "bedrock", "BedrockJSONHandler", "boto3"
    )
    handler_registry.register_lazy(
        Mode.BEDROCK_TOOLS, "bedrock", "BedrockToolsHandler", "boto3"
    )

    # Perplexity handlers
    handler_registry.register_lazy(
        Mode.PERPLEXITY_JSON, "perplexity", "PerplexityJSONHandler", "openai"
    )

    # XAI handlers
    handler_registry.register_lazy(Mode.XAI_JSON, "xai", "XAIJSONHandler", "xai_sdk")
    handler_registry.register_lazy(Mode.XAI_TOOLS, "xai", "XAIToolsHandler", "xai_sdk")


# Initialize the registry with all handlers
_register_all_handlers()
