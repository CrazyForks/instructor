# Contributing a New Client to Instructor

This guide explains how to add support for a new LLM provider to the Instructor library.

## Overview

Each client module in the `instructor/clients/` directory provides a factory function that wraps a provider's native client with Instructor's structured output capabilities. The factory function returns either an `Instructor` or `AsyncInstructor` instance depending on the client type.

## File Structure

Create a new file named `client_<provider>.py` in the `instructor/clients/` directory. For example:
- `client_openai.py` for OpenAI
- `client_anthropic.py` for Anthropic
- `client_gemini.py` for Google Gemini

## Basic Template

Here's a template for creating a new client module:

```python
from __future__ import annotations

import instructor
from typing import overload, Any, Union


# Sync overload
@overload
def from_<provider>(
    client: <ProviderClient>,
    mode: instructor.Mode = instructor.Mode.<DEFAULT_MODE>,
    **kwargs: Any,
) -> instructor.Instructor: ...


# Async overload
@overload
def from_<provider>(
    client: <AsyncProviderClient>,
    mode: instructor.Mode = instructor.Mode.<DEFAULT_MODE>,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


# Implementation
def from_<provider>(
    client: Union[<ProviderClient>, <AsyncProviderClient>],
    mode: instructor.Mode = instructor.Mode.<DEFAULT_MODE>,
    **kwargs: Any,
) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
    """Create an Instructor instance from a <Provider> client.

    Args:
        client: An instance of <Provider> client (sync or async)
        mode: The mode to use for the client
        **kwargs: Additional arguments to pass to the patching function

    Returns:
        Instructor or AsyncInstructor instance
    """
    assert mode in {<SUPPORTED_MODES>}, f"Mode {mode} is not supported"
    
    # Determine if client is async
    is_async = hasattr(client, 'acompletion') or 'async' in type(client).__name__.lower()
    
    # Return appropriate instructor instance
    if is_async:
        return instructor.AsyncInstructor(
            client=client,
            mode=mode,
            **kwargs,
        )
    return instructor.Instructor(
        client=client,
        mode=mode,
        **kwargs,
    )
```

## Integration Steps

### 1. Update `__init__.py`

Add your factory function to `instructor/clients/__init__.py`:

```python
from .client_<provider> import from_<provider>

__all__ = [
    # ... existing exports ...
    "from_<provider>",
]
```

### 2. Update Main `__init__.py`

Add conditional import in `instructor/__init__.py`:

```python
if importlib.util.find_spec("<provider_package>") is not None:
    from .clients.client_<provider> import from_<provider>
    
    __all__ += ["from_<provider>"]
```

### 3. Update `auto_client.py`

Add your provider to the `from_provider` function in `instructor/auto_client.py`:

```python
elif provider == "<provider>":
    try:
        import <provider_package>
        from instructor.clients import from_<provider>
        
        client = <provider_package>.AsyncClient() if async_client else <provider_package>.Client()
        return from_<provider>(client, model=model_name, **kwargs)
    except ImportError:
        raise ImportError(
            "The <provider_package> package is required to use the <Provider> provider. "
            "Install it with `pip install <provider_package>`."
        ) from None
```

Don't forget to add your provider to the `supported_providers` list.

### 4. Mode Support

Each provider may support different modes:
- `Mode.TOOLS` - For providers that support function/tool calling
- `Mode.JSON` - For providers that support JSON mode
- `Mode.<PROVIDER>_TOOLS` - Provider-specific tool mode
- `Mode.<PROVIDER>_JSON` - Provider-specific JSON mode

### 5. Provider-Specific Features

Some providers may need special handling:

```python
# Example: Provider with special message formatting
def <provider>_message_parser(message: dict[str, Any]) -> <ProviderMessage>:
    """Convert a message dict to provider-specific format."""
    # Implementation here
    pass

# Example: Provider with special response processing
def <provider>_process_response(
    response: <ProviderResponse>,
    response_model: type[T],
) -> T:
    """Process provider-specific response."""
    # Implementation here
    pass
```

## Testing

Create test files in `tests/llm/test_<provider>/`:

1. **Basic extraction test** (`test_simple.py`):
```python
from instructor.clients import from_<provider>

def test_basic_extraction(client):
    instructor_client = from_<provider>(client)
    # Test implementation
```

2. **Streaming test** (`test_stream.py`):
```python
def test_streaming(client):
    instructor_client = from_<provider>(client)
    # Test streaming implementation
```

3. **Async test** (`test_async.py`):
```python
import pytest

@pytest.mark.asyncio
async def test_async_extraction(async_client):
    instructor_client = from_<provider>(async_client)
    # Test async implementation
```

## Documentation

Create documentation in `docs/integrations/<provider>.md`:

```markdown
# <Provider>

Instructor supports <Provider> through the `from_<provider>` factory function.

## Installation

```bash
pip install instructor <provider-package>
```

## Usage

```python
import instructor
from <provider_package> import Client

client = Client()
instructor_client = instructor.from_<provider>(client)

# Use instructor_client for structured outputs
```

## Supported Modes

- `Mode.TOOLS` - Function calling mode
- `Mode.JSON` - JSON output mode

## Examples

[Include practical examples here]
```

## Checklist

Before submitting your PR:

- [ ] Client module created in `instructor/clients/`
- [ ] Factory function follows naming convention `from_<provider>`
- [ ] Type overloads for sync and async clients
- [ ] Added to `instructor/clients/__init__.py`
- [ ] Conditional import in main `instructor/__init__.py`
- [ ] Added to `auto_client.py` with proper error handling
- [ ] Tests cover basic extraction, streaming, and async (if supported)
- [ ] Documentation created in `docs/integrations/`
- [ ] All tests pass
- [ ] Code follows project style guidelines