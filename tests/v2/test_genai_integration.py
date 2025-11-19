from __future__ import annotations

from types import SimpleNamespace

import pytest

from instructor.mode import Mode
from instructor.utils.providers import Provider

try:
    from instructor.v2 import from_genai
    from instructor.v2.core import mode_registry
    from google.genai import types
except ModuleNotFoundError:
    pytest.skip("google-genai package is not installed", allow_module_level=True)


class DummyModels:
    def __init__(self):
        self.called = False
        self.stream_called = False

    def generate_content(self, *args, **kwargs):
        self.called = True
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )

    def generate_content_stream(self, *args, **kwargs):
        self.stream_called = True
        yield types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )


class DummyAsyncModels:
    def __init__(self):
        self.called = False

    async def generate_content(self, *args, **kwargs):
        self.called = True
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )

    async def generate_content_stream(self, *args, **kwargs):
        self.called = True
        async def _gen():
            yield types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
            )
        return _gen()


class DummyClient:
    def __init__(self):
        self.models = DummyModels()
        self.aio = SimpleNamespace(models=DummyAsyncModels())


def test_mode_registry_has_genai_handlers():
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS)


def test_from_genai_sync(monkeypatch):
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.GENAI_TOOLS, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )

    assert client.models.called


@pytest.mark.asyncio
async def test_from_genai_async(monkeypatch):
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )
    client = DummyClient()
    instructor = from_genai(client, mode=Mode.GENAI_TOOLS, use_async=True)
    await instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.aio.models.called

