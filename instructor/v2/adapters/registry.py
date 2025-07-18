"""Registry for provider adapters"""

from typing import Dict, Optional
from instructor.mode import Mode
from .base import ProviderAdapter


class AdapterRegistry:
    """Registry to map modes to provider adapters"""

    def __init__(self):
        self._adapters: Dict[str, ProviderAdapter] = {}
        self._mode_mapping: Dict[Mode, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter):
        """Register a provider adapter"""
        self._adapters[adapter.id] = adapter

        # Map all modes to this adapter
        for mode in adapter.modes:
            self._mode_mapping[mode] = adapter

    def get_adapter(self, mode: Mode) -> Optional[ProviderAdapter]:
        """Get adapter for a specific mode"""
        return self._mode_mapping.get(mode)

    def get_adapter_by_id(self, adapter_id: str) -> Optional[ProviderAdapter]:
        """Get adapter by ID"""
        return self._adapters.get(adapter_id)

    def list_adapters(self) -> Dict[str, ProviderAdapter]:
        """List all registered adapters"""
        return self._adapters.copy()


# Global registry instance
registry = AdapterRegistry()
