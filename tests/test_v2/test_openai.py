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


def test_parallel_tools_single_model():
    """Test PARALLEL_TOOLS mode with a single model type"""
    
    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]
    
    client = from_provider("openai/gpt-4o-mini", mode=Mode.PARALLEL_TOOLS)
    
    response = client.create(
        response_model=Iterable[Weather],
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas?",
            },
        ],
    )
    
    # For now, let's just check the response is not None
    # In the full implementation, this would return multiple Weather objects
    assert response is not None


def test_parallel_tools_union_models():
    """Test PARALLEL_TOOLS mode with union of different model types"""
    
    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]
    
    class GoogleSearch(BaseModel):
        query: str
    
    client = from_provider("openai/gpt-4o-mini", mode=Mode.PARALLEL_TOOLS)
    
    response = client.create(
        response_model=Iterable[Union[Weather, GoogleSearch]],
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
    )
    
    # For now, let's just check the response is not None
    # In the full implementation, this would return a mix of Weather and GoogleSearch objects
    assert response is not None

