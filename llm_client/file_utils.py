"""File utilities for handling uploads to LLM providers."""

import base64
import mimetypes
from pathlib import Path
from typing import Literal

FileType = Literal["image", "pdf", "video", "audio", "document"]


def detect_file_type(file_path: str | Path) -> FileType:
    """Detect the type of a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        File type category.

    Raises:
        ValueError: If file type cannot be determined or is unsupported.

    Examples:
        >>> detect_file_type("image.jpg")
        'image'
        >>> detect_file_type("document.pdf")
        'pdf'
    """
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(path))

    if mime_type is None:
        raise ValueError(f"Could not determine file type for {file_path}")

    if mime_type.startswith("image/"):
        return "image"
    elif mime_type == "application/pdf":
        return "pdf"
    elif mime_type.startswith("video/"):
        return "video"
    elif mime_type.startswith("audio/"):
        return "audio"
    elif mime_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]:
        return "document"
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


def get_mime_type(file_path: str | Path) -> str:
    """Get the MIME type of a file.

    Args:
        file_path: Path to the file.

    Returns:
        MIME type string.

    Raises:
        ValueError: If MIME type cannot be determined.

    Examples:
        >>> get_mime_type("image.jpg")
        'image/jpeg'
        >>> get_mime_type("document.pdf")
        'application/pdf'
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    return mime_type


def encode_file_base64(file_path: str | Path) -> str:
    """Encode a file to base64 string.

    Args:
        file_path: Path to the file.

    Returns:
        Base64 encoded string.

    Raises:
        FileNotFoundError: If file doesn't exist.

    Examples:
        >>> encoded = encode_file_base64("image.jpg")
        >>> len(encoded) > 0
        True
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def validate_file_for_provider(
    file_path: str | Path,
    provider: str,
) -> tuple[bool, str | None]:
    """Validate if a file is supported by a provider.

    Args:
        file_path: Path to the file.
        provider: Name of the provider (openai, gemini, groq, ollama).

    Returns:
        Tuple of (is_valid, error_message).

    Examples:
        >>> is_valid, error = validate_file_for_provider("image.jpg", "openai")
        >>> is_valid
        True
        >>> is_valid, error = validate_file_for_provider("video.mp4", "groq")
        >>> is_valid
        False
    """
    try:
        file_type = detect_file_type(file_path)
    except ValueError as e:
        return False, str(e)

    # Provider-specific file type support
    provider_support = {
        "openai": ["image", "pdf"],
        "gemini": ["image", "pdf", "video", "audio"],
        "groq": ["image"],  # Limited vision support
        "ollama": ["image"],  # Vision models only
    }

    supported_types = provider_support.get(provider.lower(), [])

    if file_type not in supported_types:
        return False, (
            f"{provider} does not support {file_type} files. "
            f"Supported types: {', '.join(supported_types)}"
        )

    return True, None


def prepare_file_for_openai(file_path: str | Path) -> dict:
    """Prepare a file for OpenAI API format.

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary with file data in OpenAI format.

    Examples:
        >>> file_data = prepare_file_for_openai("image.jpg")
        >>> "type" in file_data and "image_url" in file_data
        True
    """
    file_type = detect_file_type(file_path)
    mime_type = get_mime_type(file_path)
    base64_data = encode_file_base64(file_path)

    if file_type == "image":
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}}
    else:
        # For PDFs and other documents
        return {"type": "file", "file": {"data": base64_data, "mime_type": mime_type}}


def prepare_file_for_gemini(file_path: str | Path) -> dict:
    """Prepare a file for Gemini API format (via OpenAI compatibility).

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary with file data in Gemini format.

    Examples:
        >>> file_data = prepare_file_for_gemini("image.jpg")
        >>> "type" in file_data
        True
    """
    # Gemini uses OpenAI compatibility mode
    return prepare_file_for_openai(file_path)


def prepare_files_for_provider(
    file_paths: list[str | Path],
    provider: str,
) -> list[dict]:
    """Prepare multiple files for a specific provider.

    Args:
        file_paths: List of file paths.
        provider: Name of the provider.

    Returns:
        List of file data dictionaries.

    Raises:
        ValueError: If any file is not supported by the provider.
        FileNotFoundError: If any file doesn't exist.

    Examples:
        >>> files = prepare_files_for_provider(["img1.jpg", "img2.png"], "openai")
        >>> len(files) == 2
        True
    """
    prepared_files = []

    for file_path in file_paths:
        # Validate file
        is_valid, error = validate_file_for_provider(file_path, provider)
        if not is_valid:
            raise ValueError(error)

        # Prepare based on provider
        if provider.lower() in ["openai", "gemini"]:
            prepared_files.append(prepare_file_for_openai(file_path))
        elif provider.lower() in ["groq", "ollama"]:
            # Same format as OpenAI
            prepared_files.append(prepare_file_for_openai(file_path))

    return prepared_files
