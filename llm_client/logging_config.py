"""Logging configuration for llm_client package.

This module provides centralized logging configuration for the entire package.
It supports different log levels, formatters, and can be configured via
environment variables.
"""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Name of the module (typically __name__).

    Returns:
        Configured logger instance.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Operation completed")
    """
    return logging.getLogger(name)


def setup_logging(
    level: str | None = None,
    format_string: str | None = None,
    force: bool = False,
) -> None:
    """Configure logging for the llm_client package.

    This function sets up the root logger with appropriate handlers and formatters.
    It can be called by users to customize logging behavior.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses LLM_CLIENT_LOG_LEVEL env var or defaults to WARNING.
        format_string: Custom format string for log messages.
                      If None, uses a sensible default.
        force: If True, remove existing handlers before configuring.

    Examples:
        >>> # Enable debug logging
        >>> setup_logging(level="DEBUG")

        >>> # Custom format
        >>> setup_logging(format_string="%(levelname)s - %(message)s")

        >>> # Via environment variable
        >>> os.environ["LLM_CLIENT_LOG_LEVEL"] = "INFO"
        >>> setup_logging()
    """
    # Determine log level
    if level is None:
        level = os.getenv("LLM_CLIENT_LOG_LEVEL", "WARNING").upper()

    # Convert string to log level
    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get root logger for llm_client
    logger = logging.getLogger("llm_client")
    logger.setLevel(numeric_level)

    # Remove existing handlers if force is True
    if force:
        logger.handlers.clear()

    # Only add handler if none exist
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def disable_logging() -> None:
    """Disable all logging from llm_client package.

    Examples:
        >>> disable_logging()
    """
    logger = logging.getLogger("llm_client")
    logger.setLevel(logging.CRITICAL + 1)
    logger.disabled = True


def enable_logging(level: str = "INFO") -> None:
    """Enable logging at specified level.

    Args:
        level: Logging level to enable.

    Examples:
        >>> enable_logging("DEBUG")
    """
    logger = logging.getLogger("llm_client")
    logger.disabled = False
    setup_logging(level=level, force=True)


# Initialize with default settings (WARNING level)
# Users can call setup_logging() to reconfigure
setup_logging()
