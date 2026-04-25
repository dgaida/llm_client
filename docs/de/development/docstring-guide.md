# Docstring-Leitfaden

Dieses Projekt verwendet den **Google-Style** für Docstrings. Dies gewährleistet eine konsistente, lesbare und automatisch extrahierbare Dokumentation.

## Standard-Format

Jeder öffentliche Member (Klassen, Funktionen, Methoden, Module) muss einen Docstring haben.

```python
def beispiel_funktion(name: str, alter: int = 0) -> str:
    """Kurze Einzeilen-Beschreibung.

    Längere Beschreibung falls nötig. Kann mehrere Absätze
    umfassen.

    Args:
        name (str): Der Name der Person.
        alter (int): Das Alter der Person. Defaults to 0.

    Returns:
        str: Eine Begrüßungsnachricht.

    Raises:
        ValueError: Wenn der Name leer ist.

    Example:
        >>> beispiel_funktion("Alice", 30)
        "Hallo Alice, du bist 30 Jahre alt."
    """
    if not name:
        raise ValueError("Name darf nicht leer sein")
    return f"Hallo {name}, du bist {alter} Jahre alt."
```

## Anforderungen

- **Typ-Hinweise**: Verwenden Sie immer Python-Typ-Hinweise in der Funktionssignatur.  
- **Vollständigkeit**: Alle Parameter müssen im `Args`-Abschnitt dokumentiert sein.  
- **Rückgabewerte**: Dokumentieren Sie den Rückgabewert im `Returns`-Abschnitt.  
- **Ausnahmen**: Dokumentieren Sie alle explizit ausgelösten Fehler im `Raises`-Abschnitt.  

## Werkzeuge

Wir verwenden [interrogate](https://interrogate.readthedocs.io/), um die Docstring-Abdeckung zu erzwingen. Die Abdeckung muss mindestens **95%** betragen.
