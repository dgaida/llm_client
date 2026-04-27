import os
import shutil
from pathlib import Path

def generate_api_docs():
    """Generate API documentation only in the default language folder to avoid duplicate identifiers."""
    package_dir = Path("llm_client")
    api_dir = Path("docs/de/api")
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

    en_api_dir = Path("docs/en/api")
    if en_api_dir.exists():
        shutil.rmtree(en_api_dir)

    titles_de = {
        "llm_client.llm_client": "LLMClient",
        "llm_client.config": "Konfiguration",
        "llm_client.exceptions": "Ausnahmen (Exceptions)",
        "llm_client.cli": "CLI",
        "llm_client.providers.base_provider": "Basis-Provider",
        "llm_client.providers.providers": "Sync-Provider",
        "llm_client.providers.async_providers": "Async-Provider",
        "llm_client.providers.provider_factory": "Provider-Factory",
        "llm_client.providers.adapter": "LlamaIndex-Adapter",
        "llm_client.utils.token_counter": "Token-Zähler",
        "llm_client.utils.file_utils": "Datei-Utilities",
        "llm_client.utils.logging_config": "Logging-Konfiguration",
    }
    index_content = ["# API-Referenz\n\n"]
    for root, _, files in sorted(os.walk(package_dir)):
        for file in sorted(files):
            if file.endswith(".py") and file != "__init__.py":
                rel_path = Path(root) / file
                module_path = rel_path.with_suffix("").as_posix().replace("/", ".")
                title = titles_de.get(module_path, module_path.split(".")[-1].replace("_", " ").title())
                safe_name = module_path.replace("llm_client.", "").replace(".", "_") + ".md"
                target_path = api_dir / safe_name
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n\n::: {module_path}\n")
                index_content.append(f"- [{title}]({safe_name})\n")
    with open(api_dir / "index.md", "w", encoding="utf-8") as f:
        f.writelines(index_content)

if __name__ == "__main__":
    generate_api_docs()
