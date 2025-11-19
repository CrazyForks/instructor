from __future__ import annotations

from typing import Type

from ...core.exceptions import ModeError
from ...mode import Mode
from ...utils.providers import Provider
from .handler import ModeHandler


class ModeRegistry:
    """Registry of (provider, mode) -> handler class mappings."""

    def __init__(self) -> None:
        self._registry: dict[Provider, dict[Mode, Type[ModeHandler]]] = {}

    def register(
        self,
        provider: Provider,
        mode: Mode,
    ):
        """Decorator to register a handler for a provider/mode pair."""

        def wrapper(handler_cls: Type[ModeHandler]) -> Type[ModeHandler]:
            provider_entry = self._registry.setdefault(provider, {})
            provider_entry[mode] = handler_cls
            return handler_cls

        return wrapper

    def get_handler_class(
        self,
        provider: Provider,
        mode: Mode,
    ) -> Type[ModeHandler]:
        """Return a handler class for the requested provider/mode."""
        provider_entry = self._registry.get(provider)
        if provider_entry is None or mode not in provider_entry:
            raise ModeError(
                mode=str(mode),
                provider=provider.value,
                valid_modes=[
                    str(registered_mode) for registered_mode in (provider_entry or {}).keys()
                ],
            )
        return provider_entry[mode]

    def get_modes_for_provider(self, provider: Provider) -> list[Mode]:
        """List modes registered for a specific provider."""
        return sorted(self._registry.get(provider, {}).keys(), key=lambda mode: mode.value)

    def get_providers_for_mode(self, mode: Mode) -> list[Provider]:
        """List providers that registered the given mode."""
        providers = [
            provider
            for provider, modes in self._registry.items()
            if mode in modes
        ]
        return sorted(providers, key=lambda provider: provider.value)

    def is_registered(self, provider: Provider, mode: Mode) -> bool:
        """Return True if a provider/mode pair has a registered handler."""
        return mode in self._registry.get(provider, {})


mode_registry = ModeRegistry()
register_mode_handler = mode_registry.register

