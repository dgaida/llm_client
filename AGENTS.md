# Agent Guide for LLM Client Repository

Welcome, Agent! This repository contains the `llm_client` project, a universal Python client for various LLM providers.

## Project Structure  
- `llm_client/`: Core package containing the `LLMClient` and provider implementations.  
- `docs/`: Documentation source files (bilingual: German/English).  
- `notebooks/`: Example Jupyter notebooks.  
- `tests/`: Comprehensive test suite.  
- `scripts/`: Automation scripts for documentation and rate limits.  

## Coding Standards  
- **Python Version**: 3.11+  
- **Style**: Use `black` for formatting and `ruff` for linting.  
- **Docstrings**: Follow the **Google-style** docstring standard.  
- **Bilingual Documentation**: Ensure all documentation is available in both German (default) and English.  
- **Docstring Coverage**: Maintain at least 95% coverage (enforced by `interrogate`).  

## Key Workflows  
- **Version Bumping**: Automated via `.github/workflows/auto-version.yml`. Use conventional commits.  
- **Documentation Build**: Use `mkdocs build` to verify documentation changes.  
- **Testing**: Run tests with `python -m pytest --cov=llm_client --cov-report=term-missing tests/`.  

## Documentation (Bilingual Architecture)  
- **Primary Language**: German (`docs/de/`)  
- **Secondary Language**: English (`docs/en/`)  
- **Mapping**: Multilingual mapping is handled by `mkdocs-static-i18n` with `docs_structure: folder`.  
- **API Docs**: Programmatically generated via `scripts/generate_api_docs.py`.  

## Useful Commands  
- Install dev dependencies: `pip install -e ".[all]"`  
- Run linting: `ruff check .`  
- Run formatting: `black .`  
- Check docstring coverage: `interrogate -v llm_client/`  

Follow these instructions to maintain high code quality and documentation standards.

## AI Skills
This project uses AI skills to automate and standardize workflows. These skills are defined in the [auto-version-action repository](https://github.com/dgaida/auto-version-action/tree/main/skills).

- **github-repo-review**: Perform a deep, holistic code review of a GitHub repository and propose specific, actionable improvements for maintainability, clarity, correctness, and long-term scalability.  
- **mkdocs-documentation**: Generate a complete, production-ready MkDocs documentation ecosystem for a Python GitHub repository. Use this whenever setting up, improving, or automating documentation.  

## Documentation Assets  
- **Colab Badges**: To ensure that "Open In Colab" badges are correctly displayed in the documentation without being blocked by browser privacy settings or mixed content restrictions, always store the Colab badge SVG locally (e.g., in `docs/assets/colab-badge.svg`) and reference it using a relative path.  
