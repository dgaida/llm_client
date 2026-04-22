"""Configuration file support for LLMClient."""

import json
from pathlib import Path
from typing import Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class LLMConfig:
    """Configuration loader for LLMClient.

    Supports YAML and JSON configuration files for managing
    multiple provider configurations.

    Examples:
        >>> config = LLMConfig.from_file("llm_config.yaml")
        >>> print(config.default_provider)
        'openai'
        >>> print(config.get_provider_config("openai"))
        {'model': 'gpt-4o-mini', 'temperature': 0.7}
    """

    def __init__(self, config_dict: dict[str, Any]):
        """Initialize configuration from dictionary.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary.

        Raises:
            ValueError: If required fields are missing.
        """
        self.raw_config = config_dict
        self.default_provider = config_dict.get("default_provider", "openai")
        self.providers = config_dict.get("providers", {})
        self.global_settings = config_dict.get("global_settings", {})

        # Validate providers
        if not self.providers:
            raise ValueError("Configuration must contain 'providers' section")

    @classmethod
    def from_file(cls, file_path: str | Path) -> "LLMConfig":
        """Load configuration from YAML or JSON file.

        Args:
            file_path (str | Path): Path to configuration file (.yaml, .yml, or .json).

        Returns:
            LLMConfig: LLMConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported or invalid.
            ImportError: If YAML file but pyyaml not installed.

        Examples:
            >>> config = LLMConfig.from_file("config.yaml")
            >>> config = LLMConfig.from_file("config.json")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        suffix = path.suffix.lower()

        with open(path, encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "pyyaml is required to load YAML files. " "Install with: pip install pyyaml"
                    )
                config_dict = yaml.safe_load(f)
            elif suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. " f"Use .yaml, .yml, or .json"
                )

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LLMConfig":
        """Create configuration from dictionary.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary.

        Returns:
            LLMConfig: LLMConfig instance.

        Examples:
            >>> config_dict = {
            ...     "default_provider": "openai",
            ...     "providers": {
            ...         "openai": {"model": "gpt-4o", "temperature": 0.7}
            ...     }
            ... }
            >>> config = LLMConfig.from_dict(config_dict)
        """
        return cls(config_dict)

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider (str): Provider name (openai, groq, gemini, ollama).

        Returns:
            dict[str, Any]: Provider configuration dictionary.

        Raises:
            KeyError: If provider not found in configuration.

        Examples:
            >>> config.get_provider_config("openai")
            {'model': 'gpt-4o-mini', 'temperature': 0.7}
        """
        if provider not in self.providers:
            raise KeyError(
                f"Provider '{provider}' not found in configuration. "
                f"Available providers: {list(self.providers.keys())}"
            )
        return self.providers[provider]

    def get_default_config(self) -> dict[str, Any]:
        """Get configuration for the default provider.

        Returns:
            dict[str, Any]: Default provider configuration.

        Examples:
            >>> config.get_default_config()
            {'model': 'gpt-4o-mini', 'temperature': 0.7}
        """
        return self.get_provider_config(self.default_provider)

    def list_providers(self) -> list[str]:
        """List all configured providers.

        Returns:
            list[str]: List of provider names.

        Examples:
            >>> config.list_providers()
            ['openai', 'groq', 'gemini', 'ollama']
        """
        return list(self.providers.keys())

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            dict[str, Any]: Configuration dictionary.

        Examples:
            >>> config_dict = config.to_dict()
        """
        return {
            "default_provider": self.default_provider,
            "providers": self.providers,
            "global_settings": self.global_settings,
        }

    def to_file(self, file_path: str | Path) -> None:
        """Save configuration to file.

        Args:
            file_path (str | Path): Path to save configuration (.yaml or .json).

        Raises:
            ValueError: If file format is unsupported.
            ImportError: If YAML file but pyyaml not installed.

        Examples:
            >>> config.to_file("output.yaml")
            >>> config.to_file("output.json")
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        config_dict = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "pyyaml is required to save YAML files. " "Install with: pip install pyyaml"
                    )
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            elif suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. " f"Use .yaml, .yml, or .json"
                )

    def merge_with_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Merge provider config with default values.

        Args:
            defaults (dict[str, Any]): Default configuration values.

        Returns:
            dict[str, Any]: Merged configuration dictionary.

        Examples:
            >>> defaults = {"temperature": 0.7, "max_tokens": 512}
            >>> merged = config.merge_with_defaults(defaults)
        """
        provider_config = self.get_default_config().copy()

        # Apply global settings first
        for key, value in self.global_settings.items():
            if key not in provider_config:
                provider_config[key] = value

        # Apply defaults for missing keys
        for key, value in defaults.items():
            if key not in provider_config:
                provider_config[key] = value

        return provider_config

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration.

        Returns:
            tuple[bool, list[str]]: Tuple of (is_valid, error_messages).

        Examples:
            >>> is_valid, errors = config.validate()
            >>> if not is_valid:
            ...     print(f"Errors: {errors}")
        """
        errors = []

        # Check default provider exists
        if self.default_provider not in self.providers:
            errors.append(f"Default provider '{self.default_provider}' not found in providers")

        # Check each provider has required fields
        required_fields = ["model"]  # model is always required

        for provider_name, provider_config in self.providers.items():
            if not isinstance(provider_config, dict):
                errors.append(f"Provider '{provider_name}' configuration must be a dictionary")
                continue

            for field in required_fields:
                if field not in provider_config:
                    errors.append(f"Provider '{provider_name}' missing required field: {field}")

        return len(errors) == 0, errors

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"LLMConfig(default={self.default_provider}, "
            f"providers={list(self.providers.keys())})"
        )


def create_default_config() -> dict[str, Any]:
    """Create a default configuration dictionary.

    Returns:
        dict[str, Any]: Default configuration dictionary.

    Examples:
        >>> config_dict = create_default_config()
        >>> config = LLMConfig.from_dict(config_dict)
    """
    return {
        "default_provider": "openai",
        "global_settings": {
            "temperature": 0.7,
            "max_tokens": 512,
        },
        "providers": {
            "openai": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 512,
            },
            "groq": {
                "model": "qwen/qwen3-32b",
                "temperature": 0.5,
                "max_tokens": 1024,
            },
            "gemini": {
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.8,
                "max_tokens": 2048,
            },
            "ollama": {
                "model": "llama3.2:1b",
                "temperature": 0.7,
                "max_tokens": 512,
                "keep_alive": "5m",
            },
        },
    }


def generate_config_template(output_path: str | Path, format: str = "yaml") -> None:
    """Generate a template configuration file.

    Args:
        output_path (str | Path): Path to save template.
        format (str): File format ('yaml' or 'json').

    Examples:
        >>> generate_config_template("llm_config.yaml")
        >>> generate_config_template("llm_config.json", format="json")
    """
    config_dict = create_default_config()
    config = LLMConfig.from_dict(config_dict)

    path = Path(output_path)
    if format == "yaml" and path.suffix not in [".yaml", ".yml"]:
        path = path.with_suffix(".yaml")
    elif format == "json" and path.suffix != ".json":
        path = path.with_suffix(".json")

    config.to_file(path)
