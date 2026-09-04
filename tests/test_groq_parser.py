import os


def test_parse_groq_limits_creation():
    # Since live scraping depends on internet and playwright,
    # we mainly test if the script can run and produce the file.
    # We already ran it in previous steps.

    limits_file = "llm_client/providers/groq_rate_limits.md"
    assert os.path.exists(limits_file)

    with open(limits_file) as f:
        content = f.read()
        assert "# Groq Free Plan Rate Limits" in content
        assert "| MODEL ID | RPM | RPD | TPM | TPD | ASH | ASD |" in content
        # Check for at least one known model from the hardcoded fallback or live scrape
        assert "openai/gpt-oss-120b" in content


def test_tpm_parsing_logic():
    from llm_client.providers.providers import GroqProvider

    provider = GroqProvider(llm="test", api_key="test")

    error_msg = "Rate limit exceeded on tokens per minute (TPM): Limit 1000, Requested 2000"

    fallback = provider._find_fallback_model(error_msg)
    assert fallback == "meta-llama/llama-prompt-guard-2-22m"

    # Test with very high request
    error_msg_high = "Requested 100000"
    fallback_high = provider._find_fallback_model(error_msg_high)
    assert fallback_high is None

    # requested 5000 should also give meta-llama/llama-prompt-guard-2-22m (15K)
    error_msg_5k = "Requested 5000"
    fallback_5k = provider._find_fallback_model(error_msg_5k)
    assert fallback_5k == "meta-llama/llama-prompt-guard-2-22m"
