#!/bin/bash
# CLI Examples for LLM Client
# Run these commands to see the CLI in action

echo "=== LLM Client CLI Examples ==="
echo

# ============================================================
# BASIC CHAT COMMANDS
# ============================================================

echo "1. Simple query with auto-detection"
llm-client chat "What is Python in one sentence?"
echo

echo "2. Query with specific provider"
llm-client chat "Explain machine learning" --provider openai
echo

echo "3. Streaming response"
llm-client chat "Tell me a short story about AI" --stream
echo

echo "4. Different temperature settings"
llm-client chat "Write a creative poem" --temperature 1.5
llm-client chat "What is 2+2?" --temperature 0.1
echo

# ============================================================
# INTERACTIVE MODE
# ============================================================

echo "5. Interactive mode (exit with 'quit')"
# This will start an interactive session
# llm-client interactive --provider openai

echo "6. Interactive with system message"
# llm-client interactive --system "You are a helpful Python tutor"

# ============================================================
# TOKEN COUNTING
# ============================================================

echo "7. Count tokens in text"
llm-client tokens "Hello, how are you doing today?"
echo

echo "8. Count tokens for specific model"
llm-client tokens "This is a longer piece of text" --model gpt-4o
echo

# ============================================================
# CONFIGURATION MANAGEMENT
# ============================================================

echo "9. Generate config template"
llm-client config generate --output my_config.yaml
echo

echo "10. Validate configuration"
llm-client config validate llm_config.yaml
echo

echo "11. Show configuration"
llm-client config show llm_config.yaml
echo

echo "12. Show specific provider config"
llm-client config show llm_config.yaml --provider openai
echo

# ============================================================
# PROVIDER INFORMATION
# ============================================================

echo "13. List available providers"
llm-client providers
echo

# ============================================================
# FILE ANALYSIS
# ============================================================

echo "14. Analyze Python file"
llm-client analyze examples/usage_examples.py --provider openai
echo

echo "15. Analyze with custom system message"
llm-client analyze README.md --system "Summarize this documentation"
echo

# ============================================================
# ADVANCED USAGE
# ============================================================

echo "16. Use config file with specific provider"
llm-client chat "Hello" --config llm_config.yaml --provider groq
echo

echo "17. Override config settings"
llm-client chat "Test" --config llm_config.yaml --temperature 0.9 --max-tokens 1024
echo

echo "18. Ollama Cloud usage"
llm-client chat "Complex reasoning task" --provider ollama --model gpt-oss:120b-cloud
echo

# ============================================================
# PIPING AND SCRIPTING
# ============================================================

echo "19. Pipe output to file"
llm-client chat "Explain Python" > output.txt
cat output.txt
echo

echo "20. Use with other commands"
echo "def hello(): pass" | llm-client chat "Explain this code"
echo

# ============================================================
# ERROR HANDLING
# ============================================================

echo "21. Invalid provider (shows error)"
llm-client chat "Test" --provider invalid 2>&1 || echo "Error handled"
echo

echo "22. Missing API key (shows helpful error)"
# This will fail if no keys are set
# llm-client chat "Test" --provider openai 2>&1 || echo "Need API key"

# ============================================================
# BATCH PROCESSING
# ============================================================

echo "23. Process multiple queries"
for query in "What is AI?" "What is ML?" "What is DL?"; do
    echo "Query: $query"
    llm-client chat "$query" --provider groq
    echo "---"
done
echo

# ============================================================
# COMBINING WITH OTHER TOOLS
# ============================================================

echo "24. Analyze git diff"
# git diff | llm-client chat "Summarize these code changes"

echo "25. Code review"
# cat main.py | llm-client chat "Review this code for best practices"

echo "26. Generate commit message"
# git diff --staged | llm-client chat "Generate a commit message"

# ============================================================
# HELP AND VERSION
# ============================================================

echo "27. Show help"
llm-client --help
echo

echo "28. Show version"
llm-client --version
echo

echo "29. Command-specific help"
llm-client chat --help
llm-client interactive --help
llm-client config --help
echo

echo "=== Examples Complete ==="
