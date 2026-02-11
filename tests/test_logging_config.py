"""Tests for logging configuration."""

import logging
import os
import sys
from unittest.mock import patch

import pytest

from llm_client.utils.logging_config import (
    disable_logging,
    enable_logging,
    get_logger,
    setup_logging,
)


def test_get_logger():
    """Test: get_logger returns a logger with the correct name."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_setup_logging_defaults():
    """Test: setup_logging with default parameters."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        mock_logger.handlers = []

        setup_logging()

        mock_get_logger.assert_called_with("llm_client")
        mock_logger.setLevel.assert_called()


def test_setup_logging_custom_level():
    """Test: setup_logging with custom level."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        mock_logger.handlers = []

        setup_logging(level="DEBUG")

        mock_logger.setLevel.assert_called_with(logging.DEBUG)


def test_setup_logging_env_var():
    """Test: setup_logging uses environment variable."""
    with patch.dict(os.environ, {"LLM_CLIENT_LOG_LEVEL": "INFO"}):
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            mock_logger.handlers = []

            setup_logging()

            mock_logger.setLevel.assert_called_with(logging.INFO)


def test_setup_logging_force():
    """Test: setup_logging with force=True clears handlers."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_logger.handlers = [mock_stream_handler]

            setup_logging(force=True)

            assert len(mock_logger.handlers) == 0


def test_disable_logging():
    """Test: disable_logging correctly disables the logger."""
    disable_logging()
    logger = logging.getLogger("llm_client")
    assert logger.level == logging.CRITICAL + 1
    assert logger.disabled is True


def test_enable_logging():
    """Test: enable_logging correctly enables the logger."""
    enable_logging(level="DEBUG")
    logger = logging.getLogger("llm_client")
    assert logger.disabled is False
    assert logger.level == logging.DEBUG


def test_setup_logging_adds_handler():
    """Test: setup_logging adds a handler if none exist."""
    # Reset logger for this test
    logger = logging.getLogger("llm_client")
    logger.handlers.clear()

    setup_logging(level="INFO")

    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
