# Docstring Style Guide

Wir verwenden den **Google-Style** für Docstrings. Dies stellt sicher, dass unsere API-Dokumentation mit `mkdocstrings` korrekt generiert wird.

## Beispiel

```python
def example_function(param1: int, param2: str = "default") -> bool:
    \"\"\"Kurze Einzeilen-Beschreibung.

    Längere Beschreibung falls nötig. Kann mehrere Absätze
    umfassen.

    Args:
        param1 (int): Beschreibung des ersten Parameters.
        param2 (str): Beschreibung des zweiten. Defaults to "default".

    Returns:
        bool: Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Wann dieser Fehler auftritt.

    Example:
        >>> example_function(42)
        True
    \"\"\"
    return True
```

## Klassen

```python
class ExampleClass:
    \"\"\"Beschreibung der Klasse.

    Attributes:
        attr1 (int): Beschreibung des Attributs.
    \"\"\"

    def __init__(self, attr1: int):
        \"\"\"Initialisierung.

        Args:
            attr1 (int): Wert für attr1.
        \"\"\"
        self.attr1 = attr1
```
