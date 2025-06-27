"""Tests for OpenAI provider implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from instructor.providers.provider_openai import OpenAIProvider
from instructor.providers.base import ProviderRegistry
from instructor.mode import Mode
from instructor.exceptions import ConfigurationError


class UserModel(BaseModel):
    """Test model for structured outputs."""

    name: str
    age: int


class TestOpenAIProvider:
    def setup_method(self):
        """Clear registry and set up provider."""
        ProviderRegistry.clear()
        self.provider = OpenAIProvider()
        ProviderRegistry.register(self.provider)

    def test_provider_properties(self):
        """Test basic provider properties."""
        assert self.provider.name == "openai"
        assert self.provider.package_name == "openai"
        assert self.provider.extra_packages == []

    def test_supported_modes(self):
        """Test that all expected modes are supported."""
        expected_modes = {
            Mode.TOOLS,
            Mode.TOOLS_STRICT,
            Mode.FUNCTIONS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.JSON_O1,
            Mode.PARALLEL_TOOLS,
            Mode.OPENROUTER_STRUCTURED_OUTPUTS,
            Mode.RESPONSES_TOOLS,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }
        assert set(self.provider.supported_modes) == expected_modes

    def test_create_sync_client(self):
        """Test creating sync OpenAI client."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: MagicMock()
            if name == "openai"
            else __import__(name, *args),
        ) as mock_import:
            mock_openai = MagicMock()
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            mock_import.return_value = mock_openai

            client = self.provider.create_client(
                "gpt-4", async_client=False, api_key="test-key"
            )

            assert client == mock_client
            mock_openai.OpenAI.assert_called_once_with(api_key="test-key")
        """Test creating sync OpenAI client."""
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        client = self.provider.create_client(
            "gpt-4", async_client=False, api_key="test-key"
        )

        assert client == mock_client
        mock_openai.OpenAI.assert_called_once_with(api_key="test-key")

    @patch("instructor.providers.provider_openai.openai")
    def test_create_async_client(self, mock_openai):
        """Test creating async OpenAI client."""
        mock_client = Mock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        client = self.provider.create_client(
            "gpt-4", async_client=True, api_key="test-key"
        )

        assert client == mock_client
        mock_openai.AsyncOpenAI.assert_called_once_with(api_key="test-key")

    def test_create_client_import_error(self):
        """Test error when openai package not installed."""
        with patch(
            "instructor.providers.provider_openai.openai", side_effect=ImportError
        ):
            with pytest.raises(ImportError, match="OpenAI package is not installed"):
                self.provider.create_client("gpt-4")

    @patch("instructor.providers.provider_openai.openai")
    @patch("instructor.providers.provider_openai.Instructor")
    @patch("instructor.providers.provider_openai.patch")
    def test_from_client_sync(self, mock_patch, mock_instructor, mock_openai):
        """Test creating Instructor from sync OpenAI client."""
        # Setup mocks
        mock_client = Mock()
        mock_client.__class__ = mock_openai.OpenAI
        mock_client.chat.completions.create = Mock()

        mock_patched = Mock()
        mock_patch.return_value = mock_patched

        mock_instructor_instance = Mock()
        mock_instructor.return_value = mock_instructor_instance

        # Call from_client
        result = self.provider.from_client(mock_client, mode=Mode.TOOLS)

        # Verify patch was called
        mock_patch.assert_called_once_with(
            create=mock_client.chat.completions.create, mode=Mode.TOOLS
        )

        # Verify Instructor was created
        assert result == mock_instructor_instance

    @patch("instructor.providers.provider_openai.openai")
    @patch("instructor.providers.provider_openai.AsyncInstructor")
    @patch("instructor.providers.provider_openai.patch")
    def test_from_client_async(self, mock_patch, mock_async_instructor, mock_openai):
        """Test creating AsyncInstructor from async OpenAI client."""
        # Setup mocks
        mock_client = Mock()
        mock_client.__class__ = mock_openai.AsyncOpenAI
        mock_client.chat.completions.create = Mock()

        mock_patched = Mock()
        mock_patch.return_value = mock_patched

        mock_instructor_instance = Mock()
        mock_async_instructor.return_value = mock_instructor_instance

        # Call from_client
        result = self.provider.from_client(mock_client, mode=Mode.TOOLS)

        # Verify AsyncInstructor was created
        assert result == mock_instructor_instance

    def test_handle_tools(self):
        """Test TOOLS mode handler."""
        mock_model = Mock()
        mock_model.openai_schema = {"name": "TestModel", "parameters": {}}

        kwargs = {}
        model, new_kwargs = self.provider.handle_tools(mock_model, kwargs)

        assert model == mock_model
        assert new_kwargs["tools"] == [
            {
                "type": "function",
                "function": mock_model.openai_schema,
            }
        ]
        assert new_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "TestModel"},
        }

    @patch("instructor.providers.provider_openai.pydantic_function_tool")
    def test_handle_tools_strict(self, mock_pydantic_tool):
        """Test TOOLS_STRICT mode handler."""
        mock_model = Mock()
        mock_schema = {
            "function": {"name": "TestModel", "parameters": {}, "strict": False}
        }
        mock_pydantic_tool.return_value = mock_schema.copy()

        kwargs = {}
        model, new_kwargs = self.provider.handle_tools_strict(mock_model, kwargs)

        assert model == mock_model
        assert new_kwargs["tools"][0]["function"]["strict"] is True
        assert new_kwargs["tool_choice"]["function"]["name"] == "TestModel"

    @patch("instructor.providers.provider_openai.Mode.warn_mode_functions_deprecation")
    def test_handle_functions(self, mock_warn):
        """Test FUNCTIONS mode handler (deprecated)."""
        mock_model = Mock()
        mock_model.openai_schema = {"name": "TestModel", "parameters": {}}

        kwargs = {}
        model, new_kwargs = self.provider.handle_functions(mock_model, kwargs)

        mock_warn.assert_called_once()
        assert model == mock_model
        assert new_kwargs["functions"] == [mock_model.openai_schema]
        assert new_kwargs["function_call"] == {"name": "TestModel"}

    def test_handle_json_modes(self):
        """Test JSON mode handlers."""
        mock_model = Mock()
        mock_model.model_json_schema.return_value = {"type": "object", "properties": {}}
        mock_model.__name__ = "TestModel"

        # Test JSON mode
        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        model, new_kwargs = self.provider.handle_json(mock_model, kwargs.copy())

        assert new_kwargs["response_format"] == {"type": "json_object"}
        assert new_kwargs["messages"][0]["role"] == "system"
        assert "json_schema" in new_kwargs["messages"][0]["content"]

        # Test JSON_SCHEMA mode
        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        model, new_kwargs = self.provider.handle_json_schema(mock_model, kwargs.copy())

        assert new_kwargs["response_format"]["type"] == "json_object"
        assert "schema" in new_kwargs["response_format"]

        # Test MD_JSON mode
        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        model, new_kwargs = self.provider.handle_md_json(mock_model, kwargs.copy())

        assert any(
            "```json" in msg["content"]
            for msg in new_kwargs["messages"]
            if msg["role"] == "user"
        )

    def test_handle_json_o1(self):
        """Test JSON_O1 mode handler."""
        mock_model = Mock()
        mock_model.model_json_schema.return_value = {"type": "object"}

        # Should work without system messages
        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        model, new_kwargs = self.provider.handle_json_o1(mock_model, kwargs)

        assert len(new_kwargs["messages"]) == 2
        assert "json_schema" in new_kwargs["messages"][-1]["content"]

        # Should raise with system messages
        kwargs = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "test"},
            ]
        }
        with pytest.raises(ValueError, match="System messages are not supported"):
            self.provider.handle_json_o1(mock_model, kwargs)

    @patch("instructor.providers.provider_openai.handle_parallel_model")
    @patch("instructor.providers.provider_openai.ParallelModel")
    def test_handle_parallel_tools(self, mock_parallel_model, mock_handle):
        """Test PARALLEL_TOOLS mode handler."""
        mock_model = Mock()
        mock_handle.return_value = [{"tool": "schema"}]
        mock_parallel_instance = Mock()
        mock_parallel_model.return_value = mock_parallel_instance

        # Should work without streaming
        kwargs = {"stream": False}
        model, new_kwargs = self.provider.handle_parallel_tools(mock_model, kwargs)

        assert model == mock_parallel_instance
        assert new_kwargs["tools"] == [{"tool": "schema"}]
        assert new_kwargs["tool_choice"] == "auto"

        # Should raise with streaming
        kwargs = {"stream": True}
        with pytest.raises(ConfigurationError, match="stream=True is not supported"):
            self.provider.handle_parallel_tools(mock_model, kwargs)

    def test_handle_openrouter_structured_outputs(self):
        """Test OPENROUTER_STRUCTURED_OUTPUTS mode handler."""
        mock_model = Mock()
        mock_model.model_json_schema.return_value = {"type": "object", "properties": {}}
        mock_model.__name__ = "TestModel"

        kwargs = {}
        model, new_kwargs = self.provider.handle_openrouter_structured_outputs(
            mock_model, kwargs
        )

        assert model == mock_model
        assert new_kwargs["response_format"]["type"] == "json_schema"
        assert new_kwargs["response_format"]["json_schema"]["name"] == "TestModel"
        assert new_kwargs["response_format"]["json_schema"]["strict"] is True
        assert (
            new_kwargs["response_format"]["json_schema"]["schema"][
                "additionalProperties"
            ]
            is False
        )

    @patch("instructor.providers.provider_openai.pydantic_function_tool")
    def test_handle_responses_tools(self, mock_pydantic_tool):
        """Test RESPONSES_TOOLS mode handler."""
        mock_model = Mock()
        mock_model.openai_schema = {"name": "TestModel"}
        mock_model.__name__ = "TestModel"

        mock_schema = {
            "function": {
                "name": "TestModel",
                "parameters": {"type": "object"},
                "description": "Test description",
                "strict": True,
            }
        }
        mock_pydantic_tool.return_value = mock_schema.copy()

        # Test with max_tokens
        kwargs = {"max_tokens": 100}
        model, new_kwargs = self.provider.handle_responses_tools(mock_model, kwargs)

        assert model == mock_model
        assert "max_output_tokens" in new_kwargs
        assert new_kwargs["max_output_tokens"] == 100
        assert "max_tokens" not in new_kwargs
        assert new_kwargs["tools"][0]["name"] == "TestModel"
        assert new_kwargs["tool_choice"]["name"] == "TestModel"

    @patch("instructor.providers.provider_openai.pydantic_function_tool")
    def test_handle_responses_tools_with_inbuilt_tools(self, mock_pydantic_tool):
        """Test RESPONSES_TOOLS_WITH_INBUILT_TOOLS mode handler."""
        mock_model = Mock()
        mock_model.openai_schema = {"name": "TestModel"}
        mock_model.__name__ = "TestModel"

        mock_schema = {
            "function": {
                "name": "TestModel",
                "parameters": {"type": "object"},
                "strict": True,
            }
        }
        mock_pydantic_tool.return_value = mock_schema.copy()

        # Test without existing tools
        kwargs = {}
        model, new_kwargs = self.provider.handle_responses_tools_with_inbuilt_tools(
            mock_model, kwargs
        )

        assert len(new_kwargs["tools"]) == 1
        assert new_kwargs["tool_choice"]["name"] == "TestModel"

        # Test with existing tools
        existing_tool = {"type": "function", "name": "ExistingTool"}
        kwargs = {"tools": [existing_tool]}
        model, new_kwargs = self.provider.handle_responses_tools_with_inbuilt_tools(
            mock_model, kwargs
        )

        assert len(new_kwargs["tools"]) == 2
        assert new_kwargs["tools"][0] == existing_tool
        assert (
            "tool_choice" not in new_kwargs
        )  # Should not set tool_choice when appending


class TestOpenAIProviderIntegration:
    """Integration tests with the registry."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry.clear()

    def test_provider_registration(self):
        """Test that provider registers correctly."""
        provider = OpenAIProvider()
        ProviderRegistry.register(provider)

        assert ProviderRegistry.get_provider("openai") == provider

        # Test finding handlers by mode
        result = ProviderRegistry.get_handler_for_mode(Mode.TOOLS)
        assert result is not None
        found_provider, handler = result
        assert found_provider == provider
        assert handler == provider.handle_tools

    def test_backwards_compatibility(self):
        """Test that the provider maintains backwards compatibility."""
        provider = OpenAIProvider()

        # All original modes should be supported
        original_modes = [
            Mode.TOOLS,
            Mode.TOOLS_STRICT,
            Mode.FUNCTIONS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ]

        for mode in original_modes:
            assert provider.get_handler(mode) is not None, f"Mode {mode} not supported"
