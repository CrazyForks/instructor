from __future__ import annotations

import pytest

from instructor.mode import Mode
from instructor.utils.providers import Provider

try:
    from instructor.v2.core import mode_registry
except ModuleNotFoundError:
    pytest.skip("google-genai package is not installed", allow_module_level=True)


@pytest.mark.parametrize(
    "mode",
    [Mode.TOOLS, Mode.JSON],
)
def test_each_genai_mode_has_registered_handler(mode: Mode):
    """Test that generic modes work."""
    handler_cls = mode_registry.get_handler_class(Provider.GENAI, mode)
    handler = handler_cls(provider=Provider.GENAI, mode=mode)
    _, prepared_kwargs = handler.prepare_request(
        response_model=None,
        messages=[{"role": "user", "content": "Sample input"}],
    )
    assert "contents" in prepared_kwargs


@pytest.mark.parametrize(
    "old_mode,expected_normalized",
    [
        (Mode.GENAI_TOOLS, Mode.TOOLS),
        (Mode.GENAI_JSON, Mode.JSON),
        (Mode.GENAI_STRUCTURED_OUTPUTS, Mode.JSON),
    ],
)
def test_backwards_compatibility_mode_normalization(old_mode: Mode, expected_normalized: Mode):
    """Test that old provider-specific modes normalize correctly."""
    from instructor.v2.core import normalize_mode
    
    normalized = normalize_mode(Provider.GENAI, old_mode)
    assert normalized == expected_normalized
    
    # Should be able to get handler with old mode
    handler_cls = mode_registry.get_handler_class(Provider.GENAI, old_mode)
    assert handler_cls is not None

