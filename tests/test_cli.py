"""Tests for CLI functionality."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_client.cli import cli, echo, echo_error, echo_success


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_client():
    """Create a mock LLMClient."""
    client = Mock()
    client.api_choice = "openai"
    client.llm = "gpt-4o-mini"
    client.chat_completion = Mock(return_value="Test response")
    client.chat_completion_stream = Mock(return_value=iter(["Test ", "stream"]))
    return client


class TestChatCommand:
    """Tests for the chat command."""

    @patch("llm_client.cli.LLMClient")
    def test_basic_chat(self, mock_llm, runner, mock_client):
        """Test basic chat command."""
        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        mock_client.chat_completion.assert_called_once()

    @patch("llm_client.cli.LLMClient")
    def test_chat_with_provider(self, mock_llm, runner, mock_client):
        """Test chat with specific provider."""
        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello", "--provider", "openai"])

        assert result.exit_code == 0
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["api_choice"] == "openai"

    @patch("llm_client.cli.LLMClient")
    def test_chat_with_model(self, mock_llm, runner, mock_client):
        """Test chat with specific model."""
        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello", "--model", "gpt-4o"])

        assert result.exit_code == 0
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["llm"] == "gpt-4o"

    @patch("llm_client.cli.LLMClient")
    def test_chat_streaming(self, mock_llm, runner, mock_client):
        """Test streaming chat."""
        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello", "--stream"])

        assert result.exit_code == 0
        assert "Test stream" in result.output
        mock_client.chat_completion_stream.assert_called_once()

    @patch("llm_client.cli.LLMClient")
    def test_chat_with_temperature(self, mock_llm, runner, mock_client):
        """Test chat with custom temperature."""
        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello", "--temperature", "0.9"])

        assert result.exit_code == 0
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @patch("llm_client.cli.LLMClient")
    def test_chat_with_max_tokens(self, mock_llm, runner, mock_client):
        """Test chat with max tokens."""
        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello", "--max-tokens", "1024"])

        assert result.exit_code == 0
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["max_tokens"] == 1024

    @patch("llm_client.cli.LLMClient.from_config")
    def test_chat_from_config(self, mock_from_config, runner, mock_client, tmp_path):
        """Test chat loading from config file."""
        # Create temp config
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("default_provider: openai\nproviders:\n  openai:\n    model: gpt-4o")

        mock_from_config.return_value = mock_client

        result = runner.invoke(cli, ["chat", "Hello", "--config", str(config_file)])

        assert result.exit_code == 0
        mock_from_config.assert_called_once()

    @patch("llm_client.cli.LLMClient")
    def test_chat_error_handling(self, mock_llm, runner):
        """Test chat error handling."""
        mock_llm.side_effect = Exception("API Error")

        result = runner.invoke(cli, ["chat", "Hello"])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestInteractiveCommand:
    """Tests for the interactive command."""

    @patch("llm_client.cli.LLMClient")
    @patch("llm_client.cli.Prompt.ask")
    def test_interactive_basic(self, mock_ask, mock_llm, runner, mock_client):
        """Test basic interactive mode."""
        mock_llm.return_value = mock_client
        mock_ask.side_effect = ["Hello", "exit"]

        result = runner.invoke(cli, ["interactive"])

        # Should process first message then exit
        assert result.exit_code == 0
        assert mock_client.chat_completion.call_count >= 1

    @patch("llm_client.cli.LLMClient")
    @patch("llm_client.cli.Prompt.ask")
    def test_interactive_clear_command(self, mock_ask, mock_llm, runner, mock_client):
        """Test clear command in interactive mode."""
        mock_llm.return_value = mock_client
        mock_ask.side_effect = ["Hello", "clear", "exit"]

        result = runner.invoke(cli, ["interactive"])

        assert result.exit_code == 0
        assert "cleared" in result.output.lower()

    @patch("llm_client.cli.LLMClient")
    @patch("llm_client.cli.Prompt.ask")
    def test_interactive_switch_provider(self, mock_ask, mock_llm, runner, mock_client):
        """Test switch provider command."""
        mock_llm.return_value = mock_client
        mock_ask.side_effect = ["switch groq", "exit"]

        result = runner.invoke(cli, ["interactive"])

        assert result.exit_code == 0
        mock_client.switch_provider.assert_called_once_with("groq")

    @patch("llm_client.cli.LLMClient")
    @patch("llm_client.cli.Prompt.ask")
    def test_interactive_with_system_message(self, mock_ask, mock_llm, runner, mock_client):
        """Test interactive with system message."""
        mock_llm.return_value = mock_client
        mock_ask.side_effect = ["exit"]

        result = runner.invoke(cli, ["interactive", "--system", "You are helpful"])

        assert result.exit_code == 0


class TestTokensCommand:
    """Tests for the tokens command."""

    @patch("llm_client.cli.TokenCounter")
    def test_count_tokens(self, mock_counter_class, runner):
        """Test token counting."""
        mock_instance = Mock()
        mock_instance.count_string_tokens.return_value = 42
        mock_counter_class.return_value = mock_instance

        result = runner.invoke(cli, ["tokens", "Hello world"])

        assert result.exit_code == 0
        assert "42" in result.output

    @patch("llm_client.cli.TokenCounter")
    def test_count_tokens_with_model(self, mock_counter_class, runner):
        """Test token counting with specific model."""
        mock_instance = Mock()
        mock_instance.count_string_tokens.return_value = 42
        mock_counter_class.return_value = mock_instance

        result = runner.invoke(cli, ["tokens", "Hello", "--model", "gpt-4o"])

        assert result.exit_code == 0
        mock_instance.count_string_tokens.assert_called_with("Hello", model="gpt-4o")


class TestConfigCommands:
    """Tests for config subcommands."""

    @patch("llm_client.cli.generate_config_template")
    def test_config_generate(self, mock_generate, runner):
        """Test config generation."""
        result = runner.invoke(cli, ["config", "generate"])

        assert result.exit_code == 0
        mock_generate.assert_called_once()

    @patch("llm_client.cli.generate_config_template")
    def test_config_generate_json(self, mock_generate, runner):
        """Test JSON config generation."""
        result = runner.invoke(cli, ["config", "generate", "--format", "json"])

        assert result.exit_code == 0
        mock_generate.assert_called_with("llm_config.yaml", format="json")

    @patch("llm_client.cli.LLMConfig.from_file")
    def test_config_validate(self, mock_from_file, runner, tmp_path):
        """Test config validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        mock_config = Mock()
        mock_config.validate.return_value = (True, [])
        mock_config.default_provider = "openai"
        mock_config.list_providers.return_value = ["openai"]
        mock_from_file.return_value = mock_config

        result = runner.invoke(cli, ["config", "validate", str(config_file)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    @patch("llm_client.cli.LLMConfig.from_file")
    def test_config_validate_invalid(self, mock_from_file, runner, tmp_path):
        """Test validation of invalid config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        mock_config = Mock()
        mock_config.validate.return_value = (False, ["Error 1", "Error 2"])
        mock_from_file.return_value = mock_config

        result = runner.invoke(cli, ["config", "validate", str(config_file)])

        assert result.exit_code == 1
        assert "invalid" in result.output.lower()

    @patch("llm_client.cli.LLMConfig.from_file")
    def test_config_show(self, mock_from_file, runner, tmp_path):
        """Test config display."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        mock_config = Mock()
        mock_config.list_providers.return_value = ["openai", "groq"]
        mock_config.default_provider = "openai"
        mock_config.get_provider_config.return_value = {"model": "gpt-4o"}
        mock_from_file.return_value = mock_config

        result = runner.invoke(cli, ["config", "show", str(config_file)])

        assert result.exit_code == 0


class TestProvidersCommand:
    """Tests for providers command."""

    @patch("llm_client.cli.ProviderFactory.get_available_providers")
    def test_list_providers(self, mock_get_available, runner):
        """Test listing providers."""
        mock_get_available.return_value = ["openai", "groq"]

        result = runner.invoke(cli, ["providers"])

        assert result.exit_code == 0
        assert "openai" in result.output
        assert "groq" in result.output


class TestAnalyzeCommand:
    """Tests for analyze command."""

    @patch("llm_client.cli.LLMClient")
    def test_analyze_file(self, mock_llm, runner, mock_client, tmp_path):
        """Test file analysis."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        mock_llm.return_value = mock_client

        result = runner.invoke(cli, ["analyze", str(test_file)])

        assert result.exit_code == 0
        mock_client.chat_completion.assert_called_once()

        # Check that file content was included
        call_args = mock_client.chat_completion.call_args[0][0]
        assert any("print('hello')" in str(msg) for msg in call_args)

    @patch("llm_client.cli.LLMClient")
    def test_analyze_with_system_message(self, mock_llm, runner, mock_client, tmp_path):
        """Test analysis with custom system message."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        mock_llm.return_value = mock_client

        result = runner.invoke(
            cli, ["analyze", str(test_file), "--system", "Custom system message"]
        )

        assert result.exit_code == 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_echo_basic(self, capsys):
        """Test basic echo function."""
        echo("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_echo_error(self, capsys):
        """Test error echo."""
        echo_error("Error message")
        captured = capsys.readouterr()
        assert "Error" in captured.out or "Error" in captured.err

    def test_echo_success(self, capsys):
        """Test success echo."""
        echo_success("Success message")
        captured = capsys.readouterr()
        assert "Success" in captured.out or "✓" in captured.out


class TestCLIVersion:
    """Tests for version and help."""

    def test_version_option(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.3.0" in result.output

    def test_help_option(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "chat" in result.output
        assert "interactive" in result.output

    def test_command_help(self, runner):
        """Test command-specific help."""
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "PROMPT" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
