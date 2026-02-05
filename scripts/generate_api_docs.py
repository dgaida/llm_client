import os
from pathlib import Path

def generate_api_docs():
    """Programmatically generate Markdown files for API documentation."""
    api_dir = Path("docs/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    package_dir = Path("llm_client")

    # Mapping of module paths to their display titles
    titles = {
        "llm_client.llm_client": "LLMClient",
        "llm_client.config": "Configuration",
        "llm_client.exceptions": "Exceptions",
        "llm_client.cli": "CLI",
        "llm_client.providers.base_provider": "Base Provider",
        "llm_client.providers.providers": "Sync Providers",
        "llm_client.providers.async_providers": "Async Providers",
        "llm_client.providers.provider_factory": "Provider Factory",
        "llm_client.providers.adapter": "LlamaIndex Adapter",
        "llm_client.utils.token_counter": "Token Counter",
        "llm_client.utils.file_utils": "File Utilities",
        "llm_client.utils.logging_config": "Logging Configuration",
    }

    generated_files = []

    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Construct module path
                rel_path = Path(root) / file
                module_path = rel_path.with_suffix("").as_posix().replace("/", ".")

                title = titles.get(module_path, module_path.split(".")[-1].replace("_", " ").title())

                # Create a unique filename for the doc file
                # Use the relative path to avoid collisions (e.g., utils.logging vs providers.logging)
                doc_rel_path = rel_path.relative_to(package_dir.parent).with_suffix(".md")
                doc_path = api_dir.parent / "reference" / doc_rel_path.relative_to("llm_client")

                # Or just put them all in docs/api/ with names reflecting their structure
                safe_name = module_path.replace("llm_client.", "").replace(".", "_") + ".md"
                target_path = api_dir / safe_name

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n\n")
                    f.write(f"::: {module_path}\n")

                generated_files.append(target_path)

    print(f"Generated {len(generated_files)} API documentation files in {api_dir}")

if __name__ == "__main__":
    generate_api_docs()
