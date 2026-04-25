# Versionierung

Dieses Projekt verwendet **Semantic Versioning (SemVer)** und automatisiert den Release-Prozess mit `mike`.

## Mike (Versionierte Dokumentation)

Wir verwenden `mike`, um mehrere Versionen der Dokumentation gleichzeitig bereitzustellen.

### Neue Version bereitstellen

Um eine neue Version der Dokumentation zu veröffentlichen:

```bash
# Mike-Deployment für eine neue Version
mike deploy --push --update-aliases 0.5.0 latest
# Standardversion setzen
mike set-default --push latest
```

### Lokale Vorschau

Vorschau einer spezifischen Version:

```bash
mike serve
```

## Automatisierung

Die Versionierung des Pakets wird über GitHub Actions (`auto-version.yml`) gesteuert, die auf konventionellen Commits basiert.
