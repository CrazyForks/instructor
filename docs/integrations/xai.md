# xAI

Instructor supports xAI's Grok models through the `xai-sdk` package. This integration allows you to leverage xAI's powerful language models for structured outputs using Pydantic models.

## Requirements

- Python 3.10 or higher (required by `xai-sdk`)
- xAI API key from [x.ai](https://x.ai)

## Installation

Install the xAI integration using pip:

```bash
pip install "instructor[xai]"
```

Or using uv:

```bash
uv pip install "instructor[xai]"
```

Or install the dependencies directly:

```bash
pip install xai-sdk python-dotenv
# or
uv pip install xai-sdk python-dotenv
```

## Using `from_provider` (Recommended)

The easiest way to use xAI with Instructor is through the `from_provider` method:

```python
import instructor
from pydantic import BaseModel

# Auto-configure the xAI client
client = instructor.from_provider("xai/grok-3-mini")


class User(BaseModel):
    name: str
    email: str


user = client.chat.completions.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the user info: John Doe's email is john@example.com",
        }
    ],
)

print(user)
# User(name='John Doe', email='john@example.com')
```

## Direct Client Usage

If you need more control, you can use the xAI client directly:

```python
from xai_sdk.sync.client import Client
from pydantic import BaseModel
import instructor
import os

# Initialize the xAI client
xai_client = Client(api_key=os.environ["XAI_API_KEY"])

# Patch with Instructor
client = instructor.from_xai(xai_client)


class User(BaseModel):
    name: str
    age: int


# Extract structured data
user = client.chat.completions.create(
    model="grok-3-mini",
    response_model=User,
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
)

print(user)
# User(name='Jason', age=25)
```

## Async Support

xAI supports async operations through `from_provider`:

```python
import instructor
import asyncio
from pydantic import BaseModel


async def extract_user():
    # Auto-configure async client
    client = instructor.from_provider("xai/grok-3-mini", async_client=True)

    class User(BaseModel):
        name: str
        age: int

    user = await client.chat.completions.create(
        response_model=User,
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
```

Or using the async client directly:

```python
from xai_sdk.aio.client import Client as AsyncClient
import instructor
import asyncio
from pydantic import BaseModel


async def extract_user():
    # Initialize async client
    xai_client = AsyncClient(api_key=os.environ["XAI_API_KEY"])
    client = instructor.from_xai(xai_client)

    class User(BaseModel):
        name: str
        age: int

    user = await client.chat.completions.create(
        model="grok-3-mini",
        response_model=User,
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
```

## Supported Modes

xAI supports the following modes:

- `Mode.JSON` - Uses JSON mode for structured outputs (default)
- `Mode.TOOLS` - Uses function calling for structured outputs

```python
import instructor
from instructor import Mode

# Using JSON mode (default)
client = instructor.from_provider("xai/grok-3-mini", mode=Mode.JSON)

# Using TOOLS mode  
client = instructor.from_provider("xai/grok-3-mini", mode=Mode.TOOLS)
```

## Limitations

- **Streaming**: Streaming responses (`create_iterable` and `create_partial`) are not yet supported due to differences in xAI's streaming API
- **Python Version**: Requires Python 3.10 or higher (xAI SDK requirement)

## Available Models

xAI provides access to the following models:

- `grok-3` - The most capable Grok model
- `grok-3-mini` - A smaller, faster version of Grok-3

## Best Practices

1. **API Key Management**: Store your xAI API key securely using environment variables:

   ```bash
   export XAI_API_KEY="your-api-key-here"
   ```

2. **Model Selection**:
   - Use `grok-3-mini` for faster responses and lower costs
   - Use `grok-3` for more complex tasks requiring advanced reasoning

3. **Error Handling**: Always handle potential API errors:

   ```python
   try:
       user = client.chat.completions.create(
           response_model=User,
           messages=[{"role": "user", "content": "Extract user data"}],
       )
   except Exception as e:
       print(f"Error: {e}")
   ```

## Related Resources

- [xAI Documentation](https://docs.x.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)
