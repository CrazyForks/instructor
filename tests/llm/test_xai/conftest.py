import pytest
import os
from xai_sdk.sync.client import Client as SyncClient
from xai_sdk.aio.client import Client as AsyncClient


@pytest.fixture(scope="function")
def client():
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY not set")
    yield SyncClient(api_key=api_key)


@pytest.fixture(scope="function")
def aclient():
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY not set")
    yield AsyncClient(api_key=api_key)
