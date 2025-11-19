from __future__ import annotations

from typing import Type

from ...core.exceptions import ModeError
from ...mode import Mode
from ...utils.providers import Provider
from .handler import ModeHandler


def normalize_mode(provider: Provider, mode: Mode) -> Mode:
    """
    Normalize provider-specific modes to generic modes for backwards compatibility.
    
    Maps:
    - GENAI_TOOLS, GENAI_STRUCTURED_OUTPUTS -> TOOLS, JSON
    - ANTHROPIC_TOOLS, ANTHROPIC_JSON -> TOOLS, JSON
    """
    mode_mapping: dict[tuple[Provider, Mode], Mode] = {
        (Provider.GENAI, Mode.GENAI_TOOLS): Mode.TOOLS,
        (Provider.GENAI, Mode.GENAI_JSON): Mode.JSON,
        (Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS): Mode.JSON,
        (Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS): Mode.TOOLS,
        (Provider.ANTHROPIC, Mode.ANTHROPIC_JSON): Mode.JSON,
    }
    
    normalized = mode_mapping.get((provider, mode))
    return normalized if normalized is not None else mode


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
        # Normalize provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        
        provider_entry = self._registry.get(provider)
        if provider_entry is None or normalized_mode not in provider_entry:
            raise ModeError(
                mode=str(mode),
                provider=provider.value,
                valid_modes=[
                    str(registered_mode) for registered_mode in (provider_entry or {}).keys()
                ],
            )
        return provider_entry[normalized_mode]

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
        normalized_mode = normalize_mode(provider, mode)
        return normalized_mode in self._registry.get(provider, {})


mode_registry = ModeRegistry()
register_mode_handler = mode_registry.register

__all__ = ["ModeRegistry", "mode_registry", "register_mode_handler", "normalize_mode"]

