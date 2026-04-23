# Docstring Guide

This project uses the **Google-style** for docstrings. This ensures consistent, readable, and automatically extractable documentation.

## Standard Format

Every public member (classes, functions, methods, modules) must have a docstring.

```python
def example_function(name: str, age: int = 0) -> str:
    """Short one-line description.

    Longer description if necessary. Can span multiple paragraphs.

    Args:
        name (str): The name of the person.
        age (int): The age of the person. Defaults to 0.

    Returns:
        str: A greeting message.

    Raises:
        ValueError: If the name is empty.

    Example:
        >>> example_function("Alice", 30)
        "Hello Alice, you are 30 years old."
    """
    if not name:
        raise ValueError("Name cannot be empty")
    return f"Hello {name}, you are {age} years old."
```

## Requirements

- **Type Hints**: Always use Python type hints in the function signature.
- **Completeness**: All parameters must be documented in the `Args` section.
- **Return Values**: Document the return value in the `Returns` section.
- **Exceptions**: Document all explicitly raised errors in the `Raises` section.

## Tools

We use [interrogate](https://interrogate.readthedocs.io/) to enforce docstring coverage. Coverage must be at least **95%**.
