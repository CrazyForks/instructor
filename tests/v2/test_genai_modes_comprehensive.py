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
    mode_registry.get_modes_for_provider(Provider.GENAI),
)
def test_each_genai_mode_has_registered_handler(mode: Mode):
    handler_cls = mode_registry.get_handler_class(Provider.GENAI, mode)
    handler = handler_cls(provider=Provider.GENAI, mode=mode)
    _, prepared_kwargs = handler.prepare_request(
        response_model=None,
        messages=[{"role": "user", "content": "Sample input"}],
    )
    assert "contents" in prepared_kwargs

