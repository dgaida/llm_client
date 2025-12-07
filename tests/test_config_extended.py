"""Extended tests for Config module to increase coverage."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_client.config import LLMConfig, create_default_config, generate_config_template


class TestLLMConfigValidation:
    """Tests for config validation."""

    def test_config_missing_providers_raises_error(self):
        """Test: ValueError when providers section is missing."""
        config_dict = {
            "default_provider": "openai"
            # Missing 'providers' key
        }

        with pytest.raises(ValueError, match="must contain 'providers' section"):
            LLMConfig(config_dict)

    def test_config_empty_providers_raises_error(self):
        """Test: ValueError when providers dict is empty."""
        config_dict = {"default_provider": "openai", "providers": {}}

        with pytest.raises(ValueError, match="must contain 'providers' section"):
            LLMConfig(config_dict)

    def test_config_validation_invalid_provider_dict(self):
        """Test: Validation fails for non-dict provider config."""
        config_dict = {
            "default_provider": "openai",
            "providers": {"openai": "not-a-dict"},  # Should be dict
        }

        config = LLMConfig(config_dict)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("must be a dictionary" in error for error in errors)

    def test_config_validation_missing_required_field(self):
        """Test: Validation fails when required field is missing."""
        config_dict = {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "temperature": 0.7
                    # Missing 'model'
                }
            },
        }

        config = LLMConfig(config_dict)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("missing required field: model" in error for error in errors)


class TestLLMConfigFileOperations:
    """Tests for file operations."""

    def test_from_file_yaml_requires_pyyaml(self, tmp_path):
        """Test: Loading YAML file requires pyyaml."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("default_provider: openai\nproviders:\n  openai:\n    model: gpt-4o")

        with (
            patch("llm_client.config.YAML_AVAILABLE", False),
            pytest.raises(ImportError, match="pyyaml is required"),
        ):
            LLMConfig.from_file(config_path)

    def test_to_file_yaml_requires_pyyaml(self, tmp_path):
        """Test: Saving to YAML requires pyyaml."""
        config = LLMConfig.from_dict(create_default_config())
        output_path = tmp_path / "output.yaml"

        with (
            patch("llm_client.config.YAML_AVAILABLE", False),
            pytest.raises(ImportError, match="pyyaml is required"),
        ):
            config.to_file(output_path)

    def test_from_file_unsupported_format(self, tmp_path):
        """Test: Unsupported file format raises ValueError."""
        config_path = tmp_path / "config.txt"
        config_path.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            LLMConfig.from_file(config_path)

    def test_to_file_unsupported_format(self, tmp_path):
        """Test: Saving to unsupported format raises ValueError."""
        config = LLMConfig.from_dict(create_default_config())
        output_path = tmp_path / "output.xml"

        with pytest.raises(ValueError, match="Unsupported file format"):
            config.to_file(output_path)

    def test_to_file_json(self, tmp_path):
        """Test: Save config to JSON file."""
        config = LLMConfig.from_dict(create_default_config())
        output_path = tmp_path / "config.json"

        config.to_file(output_path)

        assert output_path.exists()

        # Verify it's valid JSON
        with open(output_path) as f:
            loaded = json.load(f)

        assert "default_provider" in loaded
        assert "providers" in loaded

    def test_from_file_json(self, tmp_path):
        """Test: Load config from JSON file."""
        config_dict = {
            "default_provider": "groq",
            "providers": {"groq": {"model": "llama-3.3-70b-versatile", "temperature": 0.5}},
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_dict))

        config = LLMConfig.from_file(config_path)

        assert config.default_provider == "groq"
        assert "groq" in config.providers


class TestLLMConfigMethods:
    """Tests for config methods."""

    def test_get_default_config(self):
        """Test: Get default provider configuration."""
        config = LLMConfig.from_dict(create_default_config())

        default_config = config.get_default_config()

        assert "model" in default_config
        assert isinstance(default_config, dict)

    def test_list_providers(self):
        """Test: List all providers."""
        config = LLMConfig.from_dict(create_default_config())

        providers = config.list_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "groq" in providers
        assert "gemini" in providers
        assert "ollama" in providers

    def test_to_dict(self):
        """Test: Export config to dict."""
        original_dict = create_default_config()
        config = LLMConfig.from_dict(original_dict)

        exported_dict = config.to_dict()

        assert "default_provider" in exported_dict
        assert "providers" in exported_dict
        assert "global_settings" in exported_dict

    def test_merge_with_defaults(self):
        """Test: Merge config with defaults."""
        config_dict = {
            "default_provider": "openai",
            "global_settings": {"temperature": 0.8},
            "providers": {"openai": {"model": "gpt-4o", "max_tokens": 1024}},
        }

        config = LLMConfig(config_dict)
        defaults = {"temperature": 0.5, "max_tokens": 512, "top_p": 1.0}

        merged = config.merge_with_defaults(defaults)

        # Provider config should override defaults
        assert merged["model"] == "gpt-4o"
        assert merged["max_tokens"] == 1024

        # Global settings should override defaults
        assert merged["temperature"] == 0.8

        # Defaults should fill missing keys
        assert merged["top_p"] == 1.0

    def test_merge_with_defaults_no_global_settings(self):
        """Test: Merge without global settings."""
        config_dict = {"default_provider": "openai", "providers": {"openai": {"model": "gpt-4o"}}}

        config = LLMConfig(config_dict)
        defaults = {"temperature": 0.7}

        merged = config.merge_with_defaults(defaults)

        assert merged["temperature"] == 0.7

    def test_repr(self):
        """Test: String representation of config."""
        config = LLMConfig.from_dict(create_default_config())

        repr_str = repr(config)

        assert "LLMConfig" in repr_str
        assert "default=" in repr_str
        assert "providers=" in repr_str


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_create_default_config_structure(self):
        """Test: Default config has correct structure."""
        config_dict = create_default_config()

        assert "default_provider" in config_dict
        assert "global_settings" in config_dict
        assert "providers" in config_dict

    def test_create_default_config_providers(self):
        """Test: Default config contains all providers."""
        config_dict = create_default_config()

        assert "openai" in config_dict["providers"]
        assert "groq" in config_dict["providers"]
        assert "gemini" in config_dict["providers"]
        assert "ollama" in config_dict["providers"]

    def test_create_default_config_values(self):
        """Test: Default config has expected values."""
        config_dict = create_default_config()

        assert config_dict["default_provider"] == "openai"
        assert config_dict["global_settings"]["temperature"] == 0.7
        assert config_dict["global_settings"]["max_tokens"] == 512


class TestGenerateConfigTemplate:
    """Tests for generate_config_template function."""

    def test_generate_yaml_template(self, tmp_path):
        """Test: Generate YAML template."""
        pytest.importorskip("yaml")

        output_path = tmp_path / "template.yaml"

        generate_config_template(output_path, format="yaml")

        assert output_path.exists()

        # Verify it can be loaded
        config = LLMConfig.from_file(output_path)
        assert config.default_provider is not None

    def test_generate_json_template(self, tmp_path):
        """Test: Generate JSON template."""
        output_path = tmp_path / "template.json"

        generate_config_template(output_path, format="json")

        assert output_path.exists()

        # Verify it can be loaded
        config = LLMConfig.from_file(output_path)
        assert config.default_provider is not None

    def test_generate_template_adds_yaml_extension(self, tmp_path):
        """Test: Adds .yaml extension if missing."""
        pytest.importorskip("yaml")

        output_path = tmp_path / "template"

        generate_config_template(output_path, format="yaml")

        yaml_path = tmp_path / "template.yaml"
        assert yaml_path.exists()

    def test_generate_template_adds_json_extension(self, tmp_path):
        """Test: Adds .json extension if missing."""
        output_path = tmp_path / "template"

        generate_config_template(output_path, format="json")

        json_path = tmp_path / "template.json"
        assert json_path.exists()


class TestLLMConfigEdgeCases:
    """Tests for edge cases."""

    def test_config_with_extra_keys(self):
        """Test: Config handles extra keys gracefully."""
        config_dict = {
            "default_provider": "openai",
            "providers": {"openai": {"model": "gpt-4o"}},
            "extra_key": "extra_value",
        }

        config = LLMConfig(config_dict)

        assert config.default_provider == "openai"

    def test_get_provider_config_nonexistent_provider(self):
        """Test: KeyError for nonexistent provider."""
        config = LLMConfig.from_dict(create_default_config())

        with pytest.raises(KeyError, match="Provider 'nonexistent' not found"):
            config.get_provider_config("nonexistent")

    def test_validation_multiple_errors(self):
        """Test: Validation reports multiple errors."""
        config_dict = {
            "default_provider": "nonexistent",
            "providers": {
                "openai": "not-a-dict",
                "groq": {"temperature": 0.5},  # Missing model
            },
        }

        config = LLMConfig(config_dict)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert len(errors) >= 2

    def test_config_with_no_global_settings(self):
        """Test: Config works without global_settings."""
        config_dict = {"default_provider": "openai", "providers": {"openai": {"model": "gpt-4o"}}}

        config = LLMConfig(config_dict)

        assert config.global_settings == {}

    def test_from_file_with_pathlib(self, tmp_path):
        """Test: from_file accepts pathlib.Path."""
        pytest.importorskip("yaml")

        config_path = tmp_path / "config.yaml"
        generate_config_template(config_path)

        # Use Path object directly
        config = LLMConfig.from_file(Path(config_path))

        assert config.default_provider is not None

    def test_to_file_with_pathlib(self, tmp_path):
        """Test: to_file accepts pathlib.Path."""
        pytest.importorskip("yaml")

        config = LLMConfig.from_dict(create_default_config())
        output_path = Path(tmp_path / "output.yaml")

        config.to_file(output_path)

        assert output_path.exists()


class TestLLMConfigIntegration:
    """Integration tests for config."""

    def test_full_workflow_yaml(self, tmp_path):
        """Test: Full workflow with YAML config."""
        pytest.importorskip("yaml")

        # Create config
        config_dict = {
            "default_provider": "groq",
            "global_settings": {"temperature": 0.5},
            "providers": {
                "groq": {"model": "llama-3.3-70b-versatile", "temperature": 0.7, "max_tokens": 1024}
            },
        }

        # Save to file
        config_path = tmp_path / "config.yaml"
        config = LLMConfig.from_dict(config_dict)
        config.to_file(config_path)

        # Load from file
        loaded_config = LLMConfig.from_file(config_path)

        # Verify
        assert loaded_config.default_provider == "groq"
        assert loaded_config.get_provider_config("groq")["model"] == "llama-3.3-70b-versatile"

    def test_full_workflow_json(self, tmp_path):
        """Test: Full workflow with JSON config."""
        # Create config
        config_dict = {
            "default_provider": "gemini",
            "providers": {"gemini": {"model": "gemini-2.5-flash", "temperature": 0.8}},
        }

        # Save to file
        config_path = tmp_path / "config.json"
        config = LLMConfig.from_dict(config_dict)
        config.to_file(config_path)

        # Load from file
        loaded_config = LLMConfig.from_file(config_path)

        # Verify
        assert loaded_config.default_provider == "gemini"
        assert loaded_config.get_provider_config("gemini")["model"] == "gemini-2.5-flash"
