from pydantic import BaseModel
import pytest
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
