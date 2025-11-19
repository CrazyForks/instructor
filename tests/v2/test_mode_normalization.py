from __future__ import annotations

import pytest

from instructor.mode import Mode
from instructor.utils.providers import Provider

try:
    from instructor.v2.core import normalize_mode, mode_registry
except ModuleNotFoundError:
    pytest.skip("v2 module not available", allow_module_level=True)


@pytest.mark.parametrize(
    "provider,mode,expected",
    [
        (Provider.GENAI, Mode.GENAI_TOOLS, Mode.TOOLS),
        (Provider.GENAI, Mode.GENAI_JSON, Mode.JSON),
        (Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS, Mode.JSON),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS, Mode.TOOLS),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_JSON, Mode.JSON),
        # Generic modes should pass through unchanged
        (Provider.GENAI, Mode.TOOLS, Mode.TOOLS),
        (Provider.GENAI, Mode.JSON, Mode.JSON),
        (Provider.ANTHROPIC, Mode.TOOLS, Mode.TOOLS),
        (Provider.ANTHROPIC, Mode.JSON, Mode.JSON),
    ],
)
def test_normalize_mode(provider: Provider, mode: Mode, expected: Mode):
    """Test that mode normalization works correctly."""
    result = normalize_mode(provider, mode)
    assert result == expected


def test_genai_handlers_registered_with_generic_modes():
    """Test that GenAI handlers are registered with generic modes."""
    assert mode_registry.is_registered(Provider.GENAI, Mode.TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.JSON)


def test_genai_backwards_compatibility():
    """Test that old provider-specific modes still work."""
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_JSON)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS)
    
    # Should be able to get handlers with old modes
    handler1 = mode_registry.get_handler_class(Provider.GENAI, Mode.GENAI_TOOLS)
    handler2 = mode_registry.get_handler_class(Provider.GENAI, Mode.GENAI_JSON)
    handler3 = mode_registry.get_handler_class(Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS)
    
    assert handler1 is not None
    assert handler2 is not None
    assert handler3 is not None
    
    # All should resolve to the same handlers as generic modes
    assert handler1 == mode_registry.get_handler_class(Provider.GENAI, Mode.TOOLS)
    assert handler2 == mode_registry.get_handler_class(Provider.GENAI, Mode.JSON)
    assert handler3 == mode_registry.get_handler_class(Provider.GENAI, Mode.JSON)
