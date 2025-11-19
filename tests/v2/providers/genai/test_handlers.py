from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.core import prepare_response_model
from instructor.utils.providers import Provider
from instructor.v2.providers.genai.handlers import (
    GenAIStructuredOutputsHandler,
    GenAIToolsHandler,
)

try:
    from google.genai import types
except ImportError:  # pragma: no cover - dependency not installed in CI by default
    pytest.skip("google-genai package is not installed", allow_module_level=True)


class Contact(BaseModel):
    name: str
    age: int


def test_tools_prepare_request_includes_function_schema():
    handler = GenAIToolsHandler(provider=Provider.GENAI, mode=Mode.TOOLS)
    response_model, kwargs = handler.prepare_request(
        Contact,
        messages=[{"role": "user", "content": "Jane is 28."}],
    )

    assert response_model is not None
    assert "contents" in kwargs
    assert kwargs["config"].tool_config.function_calling_config.allowed_function_names == [
        "Contact"
    ]


def test_structured_prepare_request_wraps_streaming_models():
    handler = GenAIStructuredOutputsHandler(
        provider=Provider.GENAI, mode=Mode.JSON
    )
    response_model, _ = handler.prepare_request(
        Contact,
        messages=[{"role": "user", "content": "Tom is 42."}],
        stream=True,
    )

    from instructor.dsl.partial import PartialBase

    assert response_model is not None
    assert issubclass(response_model, PartialBase)


def test_tools_parse_response_filters_thought_parts():
    handler = GenAIToolsHandler(provider=Provider.GENAI, mode=Mode.TOOLS)
    response_model = prepare_response_model(Contact)

    thought_part = types.Part.from_text(text="thinking...")
    setattr(thought_part, "thought", True)
    function_part = types.Part.from_function_call(
        name="Contact",
        args={"name": "Alicia", "age": 35},
    )
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[thought_part, function_part],
                )
            )
        ]
    )

    result = handler.parse_response(
        response=response,
        response_model=response_model,
        validation_context=None,
        strict=None,
        stream=False,
        is_async=False,
    )

    assert result.name == "Alicia"
    assert result.age == 35


def test_structured_handle_reask_appends_feedback():
    handler = GenAIStructuredOutputsHandler(
        provider=Provider.GENAI, mode=Mode.JSON
    )

    class DummyResponse:
        text = '{"name": "invalid"}'

    new_kwargs = handler.handle_reask(
        kwargs={"contents": []},
        response=DummyResponse(),
        exception=ValueError("bad json"),
    )

    assert len(new_kwargs["contents"]) == 1

