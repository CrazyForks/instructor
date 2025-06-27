"""Base classes and registry for provider implementations."""

from typing import Optional, Callable, Any, TypeVar
from abc import ABC, abstractmethod
from instructor.mode import Mode

T = TypeVar("T")


class BaseProvider(ABC):
    """Base class for all provider implementations.

    Each provider must implement:
    - name: Provider identifier (e.g., 'openai', 'anthropic')
    - package_name: Required package for this provider
    - create_client: Factory for creating native client instances
    - from_client: Factory for creating Instructor instances from native clients
    - get_mode_handlers: Mapping of modes to handler functions
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def package_name(self) -> str:
        """Package to check for import (e.g., 'openai', 'anthropic')."""
        pass

    @property
    def extra_packages(self) -> list[str]:
        """Additional packages to check (e.g., ['jsonref'] for vertexai)."""
        return []

    @abstractmethod
    def create_client(
        self, model_name: str, async_client: bool = False, **kwargs
    ) -> Any:
        """Create the underlying client for from_provider()."""
        pass

    @abstractmethod
    def from_client(self, client: Any, **kwargs) -> Any:
        """Create an Instructor instance from a native client."""
        pass

    @abstractmethod
    def get_mode_handlers(
        self,
    ) -> dict[
        Mode, Callable[[type[T], dict[str, Any]], tuple[type[T], dict[str, Any]]]
    ]:
        """Return mapping of modes to handler functions."""
        pass

    @property
    def supported_modes(self) -> list[Mode]:
        """List of modes supported by this provider."""
        return list(self.get_mode_handlers().keys())

    def get_handler(self, mode: Mode) -> Optional[Callable]:
        """Get handler function for a specific mode."""
        return self.get_mode_handlers().get(mode)


class ProviderRegistry:
    """Registry for managing provider implementations."""

    _providers: dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, provider: BaseProvider) -> None:
        """Register a provider implementation."""
        cls._providers[provider.name] = provider

    @classmethod
    def get_provider(cls, name: str) -> Optional[BaseProvider]:
        """Get a provider by name."""
        return cls._providers.get(name)

    @classmethod
    def get_all_providers(cls) -> dict[str, BaseProvider]:
        """Get all registered providers."""
        return cls._providers.copy()

    @classmethod
    def get_handler_for_mode(
        cls, mode: Mode
    ) -> Optional[tuple[BaseProvider, Callable]]:
        """Find the first provider that handles a given mode.

        Returns:
            Tuple of (provider, handler) if found, None otherwise.
        """
        for provider in cls._providers.values():
            handler = provider.get_handler(mode)
            if handler is not None:
                return provider, handler
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (useful for testing)."""
        cls._providers.clear()
