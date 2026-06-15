"""Tests for file upload functionality."""

import base64
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_client import LLMClient
from llm_client.utils.file_utils import (
    detect_file_type,
    encode_file_base64,
    get_mime_type,
    prepare_files_for_provider,
    validate_file_for_provider,
)


@pytest.fixture
def mock_image_file(tmp_path):
    """Create a mock image file."""
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image data")
    return str(image_path)


@pytest.fixture
def mock_pdf_file(tmp_path):
    """Create a mock PDF file."""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(b"fake pdf data")
    return str(pdf_path)


class TestFileUtils:
    """Tests for file utility functions."""

    def test_detect_file_type_image(self, mock_image_file):
        """Test detecting image file type."""
        assert detect_file_type(mock_image_file) == "image"

    def test_detect_file_type_pdf(self, mock_pdf_file):
        """Test detecting PDF file type."""
        assert detect_file_type(mock_pdf_file) == "pdf"

    def test_detect_file_type_audio_video(self, tmp_path):
        """Test detecting audio and video file types."""
        audio = tmp_path / "test.mp3"
        audio.write_text("fake audio")
        assert detect_file_type(str(audio)) == "audio"

        video = tmp_path / "test.mp4"
        video.write_text("fake video")
        assert detect_file_type(str(video)) == "video"

    def test_detect_file_type_unsupported(self, tmp_path):
        """Test detection of unsupported file type (could not determine MIME type)."""
        file_path = tmp_path / "test.unknown"
        file_path.write_text("test")

        with (
            patch("mimetypes.guess_type", return_value=(None, None)),
            pytest.raises(ValueError, match="Could not determine file type"),
        ):
            detect_file_type(file_path)

    def test_detect_file_type_unsupported_mime(self, tmp_path):
        """Test detection of unsupported file type (MIME type determined but not supported)."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("test")

        with (
            patch("mimetypes.guess_type", return_value=("chemical/x-xyz", None)),
            pytest.raises(ValueError, match="Unsupported file type"),
        ):
            detect_file_type(file_path)

    def test_get_mime_type_image(self, mock_image_file):
        """Test getting MIME type for image."""
        assert get_mime_type(mock_image_file) == "image/jpeg"

    def test_get_mime_type_pdf(self, mock_pdf_file):
        """Test getting MIME type for PDF."""
        assert get_mime_type(mock_pdf_file) == "application/pdf"

    def test_encode_file_base64(self, mock_image_file):
        """Test base64 encoding of file."""
        encoded = encode_file_base64(mock_image_file)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == b"fake image data"

    def test_encode_file_base64_not_found(self):
        """Test encoding non-existent file."""
        with pytest.raises(FileNotFoundError):
            encode_file_base64("nonexistent.jpg")

    def test_validate_file_for_provider_openai_image(self, mock_image_file):
        """Test validating image for OpenAI."""
        is_valid, error = validate_file_for_provider(mock_image_file, "openai")
        assert is_valid
        assert error is None

    def test_validate_file_for_provider_openai_pdf(self, mock_pdf_file):
        """Test validating PDF for OpenAI."""
        is_valid, error = validate_file_for_provider(mock_pdf_file, "openai")
        assert is_valid
        assert error is None

    def test_validate_file_for_provider_groq_pdf(self, mock_pdf_file):
        """Test validating PDF for Groq (should fail)."""
        is_valid, error = validate_file_for_provider(mock_pdf_file, "groq")
        assert not is_valid
        assert "does not support pdf files" in error.lower()

    def test_prepare_files_for_provider_openai(self, mock_image_file):
        """Test preparing files for OpenAI."""
        prepared = prepare_files_for_provider([mock_image_file], "openai")
        assert len(prepared) == 1
        assert prepared[0]["type"] == "image_url"
        assert "image_url" in prepared[0]

    def test_prepare_files_for_provider_invalid(self, mock_pdf_file):
        """Test preparing unsupported file for provider."""
        with pytest.raises(ValueError, match="does not support"):
            prepare_files_for_provider([mock_pdf_file], "groq")

    def test_prepare_files_for_provider_gemini(self, mock_image_file, tmp_path):
        """Test preparing various files for Gemini."""
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"wav data")

        prepared = prepare_files_for_provider([mock_image_file, str(audio)], "gemini")
        assert len(prepared) == 2
        assert prepared[0]["type"] == "image_url"
        assert prepared[1]["type"] == "image_url"

    def test_prepare_files_for_provider_ollama(self, mock_image_file):
        """Test preparing files for Ollama."""
        prepared = prepare_files_for_provider([mock_image_file], "ollama")
        assert len(prepared) == 1
        assert prepared[0]["type"] == "image_url"

    def test_get_mime_type_error(self, tmp_path):
        """Test MIME type error."""
        file_path = tmp_path / "test.unknown"
        file_path.write_text("test")
        with (
            patch("mimetypes.guess_type", return_value=(None, None)),
            pytest.raises(ValueError, match="Could not determine MIME type"),
        ):
            get_mime_type(file_path)

    def test_validate_file_invalid_type(self):
        """Test validation with invalid file type."""
        with patch(
            "llm_client.utils.file_utils.detect_file_type", side_effect=ValueError("Invalid")
        ):
            is_valid, error = validate_file_for_provider("test.txt", "openai")
            assert is_valid is False
            assert error == "Invalid"

    def test_prepare_file_for_openai_pdf(self, mock_pdf_file):
        """Test preparing PDF for OpenAI."""
        from llm_client.utils.file_utils import prepare_file_for_openai

        prepared = prepare_file_for_openai(mock_pdf_file)
        assert prepared["type"] == "file"
        assert "file" in prepared

    def test_detect_file_type_document(self, tmp_path):
        """Test detecting document file type."""
        doc = tmp_path / "test.txt"
        doc.write_text("plain text")
        assert detect_file_type(str(doc)) == "document"


class TestLLMClientFileUpload:
    """Tests for LLMClient file upload functionality."""

    @patch("llm_client.llm_client.ProviderFactory.create_provider")
    def test_chat_completion_with_files_basic(self, mock_factory, mock_image_file):
        """Test basic file upload with chat completion."""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_completion_with_files.return_value = "Image shows a cat"
        mock_provider.llm = "gpt-4o"
        mock_factory.return_value = mock_provider

        # Create client
        client = LLMClient(api_choice="openai")

        # Test file upload
        messages = [{"role": "user", "content": "What's in this image?"}]
        response = client.chat_completion_with_files(messages, files=[mock_image_file])

        assert response == "Image shows a cat"
        mock_provider.chat_completion_with_files.assert_called_once_with(
            messages, [mock_image_file]
        )

    @patch("llm_client.llm_client.ProviderFactory.create_provider")
    def test_chat_completion_with_files_not_found(self, mock_factory):
        """Test file upload with non-existent file."""
        mock_provider = Mock()
        mock_provider.llm = "gpt-4o"
        mock_factory.return_value = mock_provider

        client = LLMClient(api_choice="openai")

        messages = [{"role": "user", "content": "Analyze"}]
        with pytest.raises(FileNotFoundError):
            client.chat_completion_with_files(messages, files=["nonexistent.jpg"])

    @patch("llm_client.llm_client.ProviderFactory.create_provider")
    def test_chat_completion_with_files_multiple(self, mock_factory, mock_image_file, tmp_path):
        """Test uploading multiple files."""
        # Create second image
        image2 = tmp_path / "image2.png"
        image2.write_bytes(b"fake image 2")

        mock_provider = Mock()
        mock_provider.chat_completion_with_files.return_value = "Two images analyzed"
        mock_provider.llm = "gpt-4o"
        mock_factory.return_value = mock_provider

        client = LLMClient(api_choice="openai")

        messages = [{"role": "user", "content": "Compare these images"}]
        response = client.chat_completion_with_files(messages, files=[mock_image_file, str(image2)])

        assert response == "Two images analyzed"
        mock_provider.chat_completion_with_files.assert_called_once()

    @patch("llm_client.llm_client.ProviderFactory.create_provider")
    def test_chat_completion_with_files_no_files(self, mock_factory):
        """Test that method works without files (fallback to normal chat)."""
        mock_provider = Mock()
        mock_provider.chat_completion_with_files.return_value = "Normal response"
        mock_provider.llm = "gpt-4o"
        mock_factory.return_value = mock_provider

        client = LLMClient(api_choice="openai")

        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat_completion_with_files(messages, files=None)

        assert response == "Normal response"
        mock_provider.chat_completion_with_files.assert_called_once_with(messages, None)


class TestProviderFileUpload:
    """Tests for provider-specific file upload implementations."""

    @patch("llm_client.providers.providers.OpenAI")
    def test_openai_provider_with_files(self, mock_openai_class, mock_image_file):
        """Test OpenAI provider file upload."""
        from llm_client.providers.providers import OpenAIProvider

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Image description"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create provider
        provider = OpenAIProvider(llm="gpt-4o", api_key="test-key")

        # Test file upload
        messages = [{"role": "user", "content": "Describe this image"}]
        response = provider.chat_completion_with_files(messages, [mock_image_file])

        assert response == "Image description"
        mock_client.chat.completions.create.assert_called_once()

    @patch("llm_client.providers.providers.OpenAI")
    def test_gemini_provider_with_files(self, mock_openai_class, mock_image_file):
        """Test Gemini provider file upload."""
        from llm_client.providers.providers import GeminiProvider

        # Mock client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Gemini analysis"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create provider
        provider = GeminiProvider(llm="gemini-3.1-flash-lite", api_key="test-key")

        # Test file upload
        messages = [{"role": "user", "content": "Analyze"}]
        response = provider.chat_completion_with_files(messages, [mock_image_file])

        assert response == "Gemini analysis"

    @patch("llm_client.providers.providers.Client")
    def test_ollama_provider_with_files(self, mock_client_class, mock_image_file):
        """Test Ollama provider file upload (vision model)."""
        from llm_client.providers.providers import OllamaProvider

        # Mock Ollama client
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Local vision analysis"}}
        mock_client_class.return_value = mock_client

        # Create provider
        provider = OllamaProvider(llm="llava:7b")

        # Test file upload
        messages = [{"role": "user", "content": "What's this?"}]
        response = provider.chat_completion_with_files(messages, [mock_image_file])

        assert response == "Local vision analysis"
        mock_client.chat.assert_called_once()


@pytest.mark.asyncio
class TestAsyncFileUpload:
    """Tests for async file upload functionality."""

    @patch("llm_client.llm_client.ProviderFactory.create_provider")
    async def test_achat_completion_with_files(self, mock_factory, mock_image_file):
        """Test async file upload."""
        # Mock async provider
        mock_provider = Mock()

        async def mock_async_upload(messages, files):
            return "Async response"

        mock_provider.achat_completion_with_files = mock_async_upload
        mock_provider.llm = "gpt-4o"
        mock_factory.return_value = mock_provider

        # Create async client
        client = LLMClient(api_choice="openai", use_async=True)

        # Test async file upload
        messages = [{"role": "user", "content": "Analyze"}]
        response = await client.achat_completion_with_files(messages, files=[mock_image_file])

        assert response == "Async response"
