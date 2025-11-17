"""Isolated tests for Anthropic JSON parsing helpers."""

from anthropic.types import Message, Usage
import pytest
from pydantic import BaseModel, ValidationError

import instructor
from instructor.core.exceptions import ConfigurationError
from instructor.processing.response import handle_response_model


CONTROL_CHAR_JSON = """{
"data": "Claude likes
control
characters"
}"""


class _AnthropicTestModel(instructor.OpenAISchema):  # type: ignore[misc]
    data: str


def _build_message(data_content: str) -> Message:
    return Message(
        id="test_id",
        content=[{"type": "text", "text": data_content}],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def test_parse_anthropic_json_strict_control_characters() -> None:
    message = _build_message(CONTROL_CHAR_JSON)

    with pytest.raises(ValidationError):
        _AnthropicTestModel.parse_anthropic_json(message, strict=True)  # type: ignore[arg-type]


def test_parse_anthropic_json_non_strict_preserves_control_characters() -> None:
    message = _build_message(CONTROL_CHAR_JSON)

    model = _AnthropicTestModel.parse_anthropic_json(message, strict=False)  # type: ignore[arg-type]

    assert model.data == "Claude likes\ncontrol\ncharacters"


class ContactInfo(BaseModel):
    name: str
    email: str
    plan_interest: str
    demo_requested: bool


def test_handle_structured_outputs_prepares_output_format() -> None:
    response_model, kwargs = handle_response_model(
        ContactInfo,
        mode=instructor.Mode.ANTHROPIC_STRUCTURED_OUTPUTS,
        messages=[
            {"role": "system", "content": "Return contact info as JSON."},
            {"role": "user", "content": "John wants Enterprise and a demo."},
        ],
        betas=["early-access"],
    )

    assert response_model.__name__ == "ContactInfo"
    assert kwargs["messages"] == [
        {"role": "user", "content": "John wants Enterprise and a demo."}
    ]
    assert kwargs["system"]
    assert kwargs["output_format"]["type"] == "json_schema"
    assert kwargs["output_format"]["schema"]["title"] == "ContactInfo"
    assert "structured-outputs-2025-11-13" in kwargs["betas"]
    assert "early-access" in kwargs["betas"]
    assert kwargs["betas"].count("structured-outputs-2025-11-13") == 1


def test_handle_structured_outputs_requires_model() -> None:
    with pytest.raises(ConfigurationError):
        handle_response_model(
            None,
            mode=instructor.Mode.ANTHROPIC_STRUCTURED_OUTPUTS,
            messages=[{"role": "user", "content": "Fill the schema."}],
        )
