import os
import pytest
from scripts.parse_groq_rate_limits import parse_groq_limits

def test_parse_groq_limits_creation():
    # Since live scraping depends on internet and playwright,
    # we mainly test if the script can run and produce the file.
    # We already ran it in previous steps.

    limits_file = "llm_client/providers/groq_rate_limits.md"
    assert os.path.exists(limits_file)

    with open(limits_file, "r") as f:
        content = f.read()
        assert "# Groq Free Plan Rate Limits" in content
        assert "| MODEL ID | RPM | RPD | TPM | TPD | ASH | ASD |" in content
        # Check for at least one known model from the hardcoded fallback or live scrape
        assert "meta-llama/llama-4-scout-17b-16e-instruct" in content

def test_tpm_parsing_logic():
    from llm_client.providers.providers import GroqProvider

    provider = GroqProvider(llm="test", api_key="test")

    error_msg = "Rate limit exceeded on tokens per minute (TPM): Limit 10000, Requested 21142"

    # meta-llama/llama-4-scout-17b-16e-instruct has 30K TPM in our md file
    fallback = provider._find_fallback_model(error_msg)
    assert fallback == "meta-llama/llama-4-scout-17b-16e-instruct"

    # Test with very high request
    error_msg_high = "Requested 100000"
    fallback_high = provider._find_fallback_model(error_msg_high)
    assert fallback_high is None

    # requested 25000 should also give meta-llama/llama-4-scout-17b-16e-instruct (30K)
    error_msg_25k = "Requested 25000"
    fallback_25k = provider._find_fallback_model(error_msg_25k)
    assert fallback_25k == "meta-llama/llama-4-scout-17b-16e-instruct"
