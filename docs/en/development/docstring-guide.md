# Docstring Style Guide

We use the **Google Style** for docstrings. This ensures our API documentation is correctly generated with `mkdocstrings`.

## Example

```python
def example_function(param1: int, param2: str = "default") -> bool:
    \"\"\"Short single-line description.

    Longer description if necessary. Can span multiple paragraphs.

    Args:
        param1 (int): Description of the first parameter.
        param2 (str): Description of the second. Defaults to "default".

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: When this error occurs.

    Example:
        >>> example_function(42)
        True
    \"\"\"
    return True
```
