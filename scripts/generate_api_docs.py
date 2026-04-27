import os
import shutil
from pathlib import Path


def generate_api_docs():
    """Programmatically generate Markdown files for API documentation in both German and English.

    To avoid duplicate primary URL warnings from autorefs in a multi-language setup,
    only the default language (German) contains the full mkdocstrings documentation.
    The English versions provide headers and links to the default language version.
    """
    package_dir = Path("llm_client")

    titles_en = {
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

    langs = {
        "de": {
            "dir": Path("docs/de/api"),
            "titles": titles_de,
            "heading": "API-Referenz",
            "is_default": True,
        },
        "en": {
            "dir": Path("docs/en/api"),
            "titles": titles_en,
            "heading": "API Reference",
            "is_default": False,
        },
    }

    for config in langs.values():
        api_dir = config["dir"]
        if api_dir.exists():
            shutil.rmtree(api_dir)
        api_dir.mkdir(parents=True, exist_ok=True)

        titles = config["titles"]
        index_content = [f"# {config['heading']}\n\n"]

        if not config["is_default"]:
            index_content.append(
                "*Note: Detailed API documentation is currently available in the "
                "default language version to avoid build conflicts.*\n\n"
            )

        for root, _, files in sorted(os.walk(package_dir)):
            for file in sorted(files):
                if file.endswith(".py") and file != "__init__.py":
                    rel_path = Path(root) / file
                    module_path = rel_path.with_suffix("").as_posix().replace("/", ".")
                    title = titles.get(
                        module_path, module_path.split(".")[-1].replace("_", " ").title()
                    )
                    safe_name = module_path.replace("llm_client.", "").replace(".", "_") + ".md"
                    target_path = api_dir / safe_name

                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(f"# {title}\n\n")
                        if config["is_default"]:
                            f.write(f"::: {module_path}\n")
                        else:
                            f.write(
                                "The detailed API documentation for this module is available "
                                f"in the [default language version](../../api/{safe_name}).\n"
                            )

                    index_content.append(f"- [{title}]({safe_name})\n")

        with open(api_dir / "index.md", "w", encoding="utf-8") as f:
            f.writelines(index_content)
    print("Generated API documentation for both languages (English as placeholder links).")


if __name__ == "__main__":
    generate_api_docs()
