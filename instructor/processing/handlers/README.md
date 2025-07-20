# Response Handler Registry System

This directory contains the refactored response handler system that replaces the previous dictionary-based approach with a more maintainable, type-safe, and extensible handler registry pattern.

## Overview

The handler registry system provides:

1. **Lazy Loading**: Handlers are only loaded when their corresponding provider libraries are available
2. **Better Method Names**: `prepare_request()` and `format_retry_request()` clearly describe what each method does
3. **Type Safety**: Uses protocols and base classes for better type checking
4. **Provider Organization**: Handlers are organized by provider for better maintainability
5. **Extensibility**: Easy to add new providers or custom handlers

## Architecture

### Core Components

- **`base.py`**: Defines the `ResponseHandler` protocol and `ResponseHandlerBase` abstract class
- **`registry.py`**: Implements the `HandlerRegistry` class for managing handlers with lazy loading
- **Provider modules** (`openai.py`, `anthropic.py`, etc.): Contain provider-specific handler implementations

### Handler Interface

Each handler implements two methods:

```python
def prepare_request(
    self, 
    response_model: type[T] | None, 
    kwargs: dict[str, Any]
) -> tuple[type[T] | None, dict[str, Any]]:
    """Prepare and format the API request parameters for the provider."""

def format_retry_request(
    self, 
    kwargs: dict[str, Any], 
    response: Any, 
    exception: Exception
) -> dict[str, Any]:
    """Format the retry request with validation error feedback."""
```

## Usage

### Basic Usage

```python
from instructor.processing.handlers import handler_registry
from instructor.mode import Mode

# Get a handler for a specific mode
handler = handler_registry.get_handler(Mode.ANTHROPIC_TOOLS)

# Prepare a request
response_model, formatted_kwargs = handler.prepare_request(MyModel, raw_kwargs)

# Format a retry request after validation error
retry_kwargs = handler.format_retry_request(kwargs, response, validation_error)
```

### Registering Custom Handlers

```python
from instructor.processing.handlers import handler_registry
from instructor.processing.handlers.base import ResponseHandlerBase

class CustomHandler(ResponseHandlerBase):
    def prepare_request(self, response_model, kwargs):
        # Custom logic
        return response_model, kwargs
    
    def format_retry_request(self, kwargs, response, exception):
        # Custom retry logic
        return kwargs

# Register directly
handler_registry.register(Mode.CUSTOM, CustomHandler())

# Or register with lazy loading
handler_registry.register_lazy(
    Mode.CUSTOM,
    "custom",  # module path relative to handlers/
    "CustomHandler",  # class name
    "custom_package"  # optional: required package
)
```

## Provider Handler Modules

Each provider has its own module with handlers for different modes:

### OpenAI (`openai.py`)
- `ToolsHandler` - Standard tools mode
- `ToolsStrictHandler` - Strict tools mode
- `FunctionsHandler` - Legacy functions mode
- `JSONHandler` - JSON response format
- `MDJSONHandler` - Markdown JSON format
- `JSONSchemaHandler` - JSON Schema mode
- `JSONO1Handler` - O1 model JSON mode
- `ParallelToolsHandler` - Parallel tool calls
- `ResponsesToolsHandler` - Response tools mode
- `OpenRouterHandler` - OpenRouter structured outputs

### Anthropic (`anthropic.py`)
- `AnthropicToolsHandler` - Tools mode
- `AnthropicReasoningToolsHandler` - Reasoning tools mode
- `AnthropicJSONHandler` - JSON mode
- `AnthropicParallelToolsHandler` - Parallel tools mode

### Other Providers
Similar patterns for Mistral, Cohere, Gemini, VertexAI, Cerebras, Fireworks, Writer, Bedrock, Perplexity, and XAI.

## Migration from Old System

The old dictionary-based system can be gradually migrated:

```python
# Old system
mode_handlers = {
    Mode.TOOLS: handle_tools,
    Mode.ANTHROPIC_TOOLS: handle_anthropic_tools,
    # ...
}

# New system
handler = handler_registry.get_handler(mode)
response_model, kwargs = handler.prepare_request(response_model, kwargs)
```

## Benefits

1. **Cleaner Code**: No more giant dictionaries mapping modes to functions
2. **Better Testing**: Each handler can be tested in isolation
3. **Lazy Loading**: Reduces import time and memory usage
4. **Type Safety**: Protocols ensure handlers implement required methods
5. **Extensibility**: Easy to add new providers without modifying core code
6. **Better Organization**: Provider-specific logic is contained in dedicated modules

## Future Improvements

1. **Mode Metadata**: Enhance the Mode enum to include handler information
2. **Handler Composition**: Allow combining handlers for complex scenarios
3. **Handler Middleware**: Add pre/post processing hooks
4. **Performance Monitoring**: Track handler performance and usage
5. **Dynamic Registration**: Allow runtime registration from external packages