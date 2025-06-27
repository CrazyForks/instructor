"""Tests for the provider base classes and registry."""

import pytest
from typing import Any, Callable
from instructor.providers.base import BaseProvider, ProviderRegistry
from instructor.mode import Mode


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def package_name(self) -> str:
        return "mock_package"

    @property
    def extra_packages(self) -> list[str]:
        return ["extra_mock"]

    def create_client(
        self, model_name: str, async_client: bool = False, **kwargs
    ) -> Any:
        return {"model": model_name, "async": async_client, "kwargs": kwargs}

    def from_client(self, client: Any, **kwargs) -> Any:
        return {"client": client, "kwargs": kwargs}

    def get_mode_handlers(self) -> dict[Mode, Callable]:
        def mock_handler(
            response_model: type, new_kwargs: dict[str, Any]
        ) -> tuple[type, dict[str, Any]]:
            new_kwargs["mock_handled"] = True
            return response_model, new_kwargs

        return {
            Mode.TOOLS: mock_handler,
            Mode.JSON: mock_handler,
        }


class TestBaseProvider:
    def test_abstract_methods(self):
        """Test that BaseProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore

    def test_mock_provider_properties(self):
        """Test provider properties."""
        provider = MockProvider()
        assert provider.name == "mock"
        assert provider.package_name == "mock_package"
        assert provider.extra_packages == ["extra_mock"]

    def test_supported_modes(self):
        """Test that supported_modes returns correct modes."""
        provider = MockProvider()
        assert set(provider.supported_modes) == {Mode.TOOLS, Mode.JSON}

    def test_get_handler(self):
        """Test getting handlers for specific modes."""
        provider = MockProvider()

        # Should return handler for supported modes
        assert provider.get_handler(Mode.TOOLS) is not None
        assert provider.get_handler(Mode.JSON) is not None

        # Should return None for unsupported modes
        assert provider.get_handler(Mode.ANTHROPIC_TOOLS) is None

    def test_create_client(self):
        """Test client creation."""
        provider = MockProvider()

        # Test sync client
        client = provider.create_client(
            "test-model", async_client=False, api_key="test"
        )
        assert client == {
            "model": "test-model",
            "async": False,
            "kwargs": {"api_key": "test"},
        }

        # Test async client
        client = provider.create_client("test-model", async_client=True)
        assert client == {"model": "test-model", "async": True, "kwargs": {}}

    def test_from_client(self):
        """Test creating instructor from client."""
        provider = MockProvider()
        mock_client = {"type": "mock_client"}

        result = provider.from_client(mock_client, mode=Mode.TOOLS)
        assert result == {"client": mock_client, "kwargs": {"mode": Mode.TOOLS}}


class TestProviderRegistry:
    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry.clear()

    def test_register_provider(self):
        """Test registering a provider."""
        provider = MockProvider()
        ProviderRegistry.register(provider)

        assert ProviderRegistry.get_provider("mock") == provider

    def test_get_provider_not_found(self):
        """Test getting non-existent provider."""
        assert ProviderRegistry.get_provider("nonexistent") is None

    def test_get_all_providers(self):
        """Test getting all providers."""
        provider1 = MockProvider()

        # Create second mock provider with different name
        class MockProvider2(MockProvider):
            @property
            def name(self) -> str:
                return "mock2"

        provider2 = MockProvider2()

        ProviderRegistry.register(provider1)
        ProviderRegistry.register(provider2)

        all_providers = ProviderRegistry.get_all_providers()
        assert len(all_providers) == 2
        assert all_providers["mock"] == provider1
        assert all_providers["mock2"] == provider2

    def test_get_handler_for_mode(self):
        """Test finding handler for a specific mode."""
        provider = MockProvider()
        ProviderRegistry.register(provider)

        # Should find handler for supported modes
        result = ProviderRegistry.get_handler_for_mode(Mode.TOOLS)
        assert result is not None
        provider_found, handler = result
        assert provider_found == provider
        assert handler is not None

        # Test that handler works
        response_model = str  # dummy type
        kwargs = {"test": "value"}
        new_model, new_kwargs = handler(response_model, kwargs)
        assert new_model == response_model
        assert new_kwargs["mock_handled"] is True
        assert new_kwargs["test"] == "value"

        # Should return None for unsupported modes
        assert ProviderRegistry.get_handler_for_mode(Mode.ANTHROPIC_TOOLS) is None

    def test_clear_registry(self):
        """Test clearing the registry."""
        provider = MockProvider()
        ProviderRegistry.register(provider)

        assert len(ProviderRegistry.get_all_providers()) == 1

        ProviderRegistry.clear()
        assert len(ProviderRegistry.get_all_providers()) == 0
        assert ProviderRegistry.get_provider("mock") is None
