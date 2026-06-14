import re

filename = "tests/test_base_provider.py"
with open(filename) as f:
    content = f.read()

# Add list_models to ConcreteProvider
content = re.sub(
    r'(def is_available\(\):\s+"""Return True for testing\."""\s+return True)',
    r'\1\n\n    def list_models(self):\n        """Return a mock list of models."""\n        return ["test-model", "other-model"]',
    content,
)

# Add list_models to AlmostCompleteProvider (it still needs to be incomplete in other ways)
# It was missing is_available before. Now it also misses list_models if I don't add it.
# Actually, the test test_all_abstract_methods_must_be_implemented checks that it fails if ONE is missing.
# So I should keep it missing something.

with open(filename, "w") as f:
    f.write(content)
