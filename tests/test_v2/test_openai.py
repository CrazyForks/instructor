from pydantic import BaseModel
import pytest
from typing import Literal, Union
from collections.abc import Iterable
from instructor.v2.auto_client import from_provider
from instructor.mode import Mode

modes = [
    Mode.TOOLS,
    Mode.TOOLS_STRICT,
    Mode.JSON,
    Mode.JSON_SCHEMA,
    Mode.PARALLEL_TOOLS,
]


@pytest.mark.parametrize("mode", modes)
def test_openai_client(mode: Mode):
    client = from_provider("openai/gpt-4o-mini", mode=mode)
    assert client is not None

    response = client.create(
        response_model=None,
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    assert response is not None


@pytest.mark.parametrize("mode", modes)
def test_openai_client_with_context(mode: Mode):
    class ResponseModel(BaseModel):
        name: str
        age: int

    client = from_provider("openai/gpt-4o-mini", mode=mode)
    response = client.create(
        response_model=ResponseModel,
        messages=[
            {
                "role": "user",
                "content": "Extract name and age from the following text: {{text}}",
            }
        ],
        context={"text": "My name is John and I am 30 years old."},
    )

    assert response is not None
    assert response.name == "John"
    assert response.age == 30


def test_parallel_tools_basic():
    """Test PARALLEL_TOOLS mode with a basic model"""
    
    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]
    
    client = from_provider("openai/gpt-4o-mini", mode=Mode.PARALLEL_TOOLS)
    
    # For v2, we'll test with a simple model first
    # Full Iterable support would require implementing ParallelModel handling
    response = client.create(
        response_model=Weather,
        messages=[
            {"role": "system", "content": "Extract weather information"},
            {
                "role": "user",
                "content": "What is the weather in Toronto? Use metric units.",
            },
        ],
    )
    
    assert response is not None
    assert response.location.lower() == "toronto"
    assert response.units == "metric"


# Note: Full parallel tools support with Iterable[Model] would require:
# 1. Implementing ParallelModel class
# 2. Special handling in build_request to convert Iterable types
# 3. Special parsing logic to handle multiple tool calls
# This is a complex feature that would need significant code migration

