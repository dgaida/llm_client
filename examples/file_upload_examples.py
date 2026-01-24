"""Examples for file upload functionality with LLM Client.

This module demonstrates how to upload and analyze files (images, PDFs, etc.)
with different LLM providers.
"""

import asyncio

from llm_client import LLMClient
from llm_client.exceptions import FileUploadNotSupportedError


def example_basic_image_analysis():
    """Basic example: Analyze a single image with OpenAI."""
    print("=" * 60)
    print("Example 1: Basic Image Analysis with OpenAI")
    print("=" * 60)

    # Create client
    client = LLMClient(api_choice="openai", llm="gpt-4o")

    # Prepare messages
    messages = [
        {"role": "user", "content": "What do you see in this image? Describe it in detail."}
    ]

    # Upload and analyze image
    try:
        response = client.chat_completion_with_files(messages, files=["vacation_photo.jpg"])
        print(f"\nResponse:\n{response}\n")
    except FileNotFoundError:
        print("Note: Please provide a 'vacation_photo.jpg' file to run this example")


def example_multiple_images():
    """Example: Analyze multiple images at once."""
    print("=" * 60)
    print("Example 2: Analyze Multiple Images")
    print("=" * 60)

    client = LLMClient(api_choice="gemini")

    messages = [
        {
            "role": "user",
            "content": "Compare these images. What are the similarities and differences?",
        }
    ]

    files = ["image1.jpg", "image2.jpg", "image3.png"]

    try:
        response = client.chat_completion_with_files(messages, files=files)
        print(f"\nComparison:\n{response}\n")
    except FileNotFoundError as e:
        print(f"Note: {e}")


def example_pdf_analysis():
    """Example: Analyze PDF document with Gemini."""
    print("=" * 60)
    print("Example 3: PDF Document Analysis")
    print("=" * 60)

    client = LLMClient(api_choice="gemini")

    messages = [
        {
            "role": "user",
            "content": "Please summarize this document and extract the main findings.",
        }
    ]

    try:
        response = client.chat_completion_with_files(messages, files=["research_paper.pdf"])
        print(f"\nSummary:\n{response}\n")
    except FileNotFoundError:
        print("Note: Please provide a 'research_paper.pdf' file to run this example")


def example_mixed_files():
    """Example: Upload both images and PDFs together."""
    print("=" * 60)
    print("Example 4: Mixed File Types (Images + PDFs)")
    print("=" * 60)

    client = LLMClient(api_choice="gemini")

    messages = [
        {
            "role": "user",
            "content": "Analyze these files and create a comprehensive report combining "
            "the information from the images and documents.",
        }
    ]

    files = ["chart.png", "data_table.jpg", "report.pdf"]

    try:
        response = client.chat_completion_with_files(messages, files=files)
        print(f"\nComprehensive Report:\n{response}\n")
    except FileNotFoundError as e:
        print(f"Note: {e}")


def example_vision_with_ollama():
    """Example: Use Ollama vision models (llava) for image analysis."""
    print("=" * 60)
    print("Example 5: Local Vision Model with Ollama")
    print("=" * 60)

    # Use a vision-capable model like llava
    client = LLMClient(api_choice="ollama", llm="llava:7b")

    messages = [{"role": "user", "content": "Describe this image in detail."}]

    try:
        response = client.chat_completion_with_files(messages, files=["photo.jpg"])
        print(f"\nDescription:\n{response}\n")
    except FileNotFoundError:
        print("Note: Please provide a 'photo.jpg' file to run this example")
    except FileUploadNotSupportedError as e:
        print(f"Error: {e}")
        print("Make sure you have a vision model installed: ollama pull llava:7b")


async def example_async_file_upload():
    """Example: Async file upload for faster processing."""
    print("=" * 60)
    print("Example 6: Async File Upload")
    print("=" * 60)

    # Create async client
    client = LLMClient(api_choice="openai", llm="gpt-4o", use_async=True)

    # Process multiple images concurrently
    images = ["img1.jpg", "img2.jpg", "img3.jpg"]
    questions = [
        "What objects are in this image?",
        "What is the dominant color?",
        "Describe the scene.",
    ]

    async def analyze_image(img_path: str, question: str):
        """Analyze a single image."""
        messages = [{"role": "user", "content": question}]
        try:
            return await client.achat_completion_with_files(messages, files=[img_path])
        except FileNotFoundError:
            return f"File not found: {img_path}"

    # Run all analyses concurrently
    tasks = [analyze_image(img, q) for img, q in zip(images, questions, strict=False)]
    results = await asyncio.gather(*tasks)

    for img, question, result in zip(images, questions, results, strict=False):
        print(f"\nImage: {img}")
        print(f"Question: {question}")
        print(f"Answer: {result[:100]}...")


def example_file_validation():
    """Example: Check file support before uploading."""
    print("=" * 60)
    print("Example 7: File Validation")
    print("=" * 60)

    from llm_client.file_utils import validate_file_for_provider

    test_files = [
        ("image.jpg", "openai"),
        ("document.pdf", "openai"),
        ("video.mp4", "groq"),  # Should fail
        ("image.png", "ollama"),
    ]

    for file_path, provider in test_files:
        is_valid, error = validate_file_for_provider(file_path, provider)
        if is_valid:
            print(f"✓ {file_path} is supported by {provider}")
        else:
            print(f"✗ {file_path} is NOT supported by {provider}: {error}")


def example_chart_analysis():
    """Example: Analyze charts and extract data."""
    print("=" * 60)
    print("Example 8: Chart and Data Visualization Analysis")
    print("=" * 60)

    client = LLMClient(api_choice="gemini")

    messages = [
        {
            "role": "user",
            "content": "Analyze this chart. Extract the key data points, trends, "
            "and provide insights. Format the response as a structured report.",
        }
    ]

    try:
        response = client.chat_completion_with_files(messages, files=["sales_chart.png"])
        print(f"\nChart Analysis:\n{response}\n")
    except FileNotFoundError:
        print("Note: Please provide a 'sales_chart.png' file to run this example")


def example_with_context():
    """Example: Upload files with conversation context."""
    print("=" * 60)
    print("Example 9: Files with Conversation Context")
    print("=" * 60)

    client = LLMClient(api_choice="openai", llm="gpt-4o")

    # Build conversation with context
    messages = [
        {
            "role": "system",
            "content": "You are an expert data analyst. Provide detailed, " "structured analysis.",
        },
        {
            "role": "user",
            "content": "I need help understanding this quarterly report. "
            "Please analyze the key metrics and trends.",
        },
    ]

    try:
        response = client.chat_completion_with_files(messages, files=["q4_report.pdf"])
        print(f"\nAnalysis:\n{response}\n")

        # Continue conversation
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "What are the main risks based on this data?"},
            ]
        )

        follow_up = client.chat_completion(messages)
        print(f"\nFollow-up:\n{follow_up}\n")

    except FileNotFoundError:
        print("Note: Please provide a 'q4_report.pdf' file to run this example")


def main():
    """Run all examples."""
    examples = [
        example_basic_image_analysis,
        example_multiple_images,
        example_pdf_analysis,
        example_mixed_files,
        example_vision_with_ollama,
        example_file_validation,
        example_chart_analysis,
        example_with_context,
    ]

    print("\n" + "=" * 60)
    print("LLM Client - File Upload Examples")
    print("=" * 60 + "\n")

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Error in example {i}: {e}\n")

    # Async example
    print("\nRunning async example...")
    try:
        asyncio.run(example_async_file_upload())
    except Exception as e:
        print(f"Error in async example: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
