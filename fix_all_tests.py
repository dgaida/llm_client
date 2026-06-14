import os
import re


def fix_file(filepath):
    with open(filepath) as f:
        content = f.read()

    # 1. Implement list_models in ConcreteProvider
    if "class ConcreteProvider(BaseProvider):" in content:
        if "def list_models(self)" not in content:
            # Add it after is_available
            content = re.sub(
                r"(@staticmethod\s+def is_available\(\):.*?return True)",
                r'\1\n\n    def list_models(self) -> list[str]:\n        """Return a mock list of models."""\n        return ["test-model", "other-model"]',
                content,
                flags=re.DOTALL,
            )
            print(f"Updated {filepath}: added list_models to ConcreteProvider")

    # 2. Fix other providers that inherit directly from BaseProvider but are not ConcreteProvider
    # In test_base_provider.py, AlmostCompleteProvider is INTENDED to be incomplete, so leave it.

    # 3. Check for other providers in the same file
    # Some providers are defined inside test methods

    # Find all occurrences of class ... (BaseProvider) or (ConcreteProvider)
    # Actually, if they inherit from ConcreteProvider, they get list_models automatically.

    with open(filepath, "w") as f:
        f.write(content)


# Apply to tests/test_base_provider.py
fix_file("tests/test_base_provider.py")

# Check other files
for root, dirs, files in os.walk("tests"):
    for file in files:
        if file.endswith(".py") and file != "test_base_provider.py":
            fix_file(os.path.join(root, file))
