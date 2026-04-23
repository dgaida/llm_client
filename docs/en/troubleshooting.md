# Troubleshooting

This page lists common problems and their solutions.

## General Issues

### API Key not found

**Problem**: The application cannot find your API key.

**Solution**:
- Ensure you have a `.env` or `secrets.env` file.
- Check environment variable names (e.g., `OPENAI_API_KEY`).

## Provider Specific Issues

### Ollama not running

**Problem**: Connection to local Ollama fails.

**Solution**:
Check if Ollama is running using `ollama list` or start it with `ollama serve`.
