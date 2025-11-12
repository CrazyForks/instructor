"""
LLM tests for v2 Anthropic provider.

These tests make actual API calls and require ANTHROPIC_API_KEY.
"""

import instructor
import pytest
from typing import Literal
from collections.abc import Iterable
from pydantic import BaseModel


class Answer(BaseModel):
    """Simple answer model."""

    answer: float


class Weather(BaseModel):
    """Weather tool."""

    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    """Search tool."""

    query: str


# Test basic TOOLS mode with v2
def test_v2_tools_mode_basic():
    """Test basic v2 TOOLS mode (single tool)."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=instructor.Mode.TOOLS,
    )
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "What is 2 + 2? Reply with a number.",
            },
        ],
        max_tokens=1000,
    )

    assert isinstance(response, Answer)
    assert response.answer == 4.0


# Test TOOLS mode with thinking parameter (auto-detected)
def test_v2_tools_mode_with_thinking():
    """Test v2 TOOLS mode with thinking parameter auto-detection."""
    # Note: Thinking requires Claude 3.7 Sonnet, but the test is optional
    pytest.skip("Thinking mode requires paid API access to Claude 3.7 Sonnet")
    client = instructor.from_provider(
        "anthropic/claude-3-7-sonnet-20250219",
        mode=instructor.Mode.TOOLS,
    )
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "Which is larger, 9.11 or 9.8? Think carefully.",
            },
        ],
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    assert isinstance(response, Answer)
    assert response.answer == 9.8


# Test parallel tools mode (auto-detected from Iterable[Union[...]])
def test_v2_parallel_tools_auto_detection():
    """Test v2 TOOLS mode auto-detecting parallel from Iterable[Union[...]]."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=instructor.Mode.TOOLS,
    )
    response = client.chat.completions.create(
        response_model=Iterable[Weather | GoogleSearch],
        messages=[
            {
                "role": "system",
                "content": "You must always use tools. Use them simultaneously when appropriate.",
            },
            {
                "role": "user",
                "content": "Get weather for Toronto and search for current events.",
            },
        ],
        max_tokens=1000,
    )

    result = list(response)
    assert len(result) >= 1
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)


# Test async TOOLS mode
@pytest.mark.asyncio
async def test_v2_tools_mode_async():
    """Test async v2 TOOLS mode."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=instructor.Mode.TOOLS,
        async_client=True,
    )
    response = await client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "What is 3 + 3? Reply with a number.",
            },
        ],
        max_tokens=1000,
    )

    assert isinstance(response, Answer)
    assert response.answer == 6.0


# Test async parallel tools
@pytest.mark.asyncio
@pytest.mark.skip(reason="Async parallel generators need async wrapper implementation")
async def test_v2_parallel_tools_async():
    """Test async v2 TOOLS mode with parallel auto-detection."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=instructor.Mode.TOOLS,
        async_client=True,
    )
    response = await client.chat.completions.create(
        response_model=Iterable[Weather | GoogleSearch],
        messages=[
            {
                "role": "system",
                "content": "Use tools simultaneously.",
            },
            {
                "role": "user",
                "content": "Get weather for New York and Dallas.",
            },
        ],
        max_tokens=1000,
    )

    result = [r async for r in response]
    assert len(result) >= 1
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)


# Test JSON mode (should still work)
def test_v2_json_mode():
    """Test v2 JSON mode still works."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=instructor.Mode.JSON,
    )
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "What is 5 + 5? Reply with a number.",
            },
        ],
        max_tokens=1000,
    )

    assert isinstance(response, Answer)
    assert response.answer == 10.0
