# Fehlerbehebung

Diese Seite listet häufige Probleme und deren Lösungen auf.

## Allgemeine Probleme

### API Key nicht gefunden

**Problem**: Die Anwendung kann Ihren API-Key nicht finden.

**Lösung**:
- Stellen Sie sicher, dass Sie eine `.env` oder `secrets.env` Datei haben.
- Überprüfen Sie die Namen der Umgebungsvariablen (z.B. `OPENAI_API_KEY`).

## Provider-spezifische Probleme

### Ollama läuft nicht

**Problem**: Verbindung zu lokalem Ollama schlägt fehl.

**Lösung**:
Prüfen Sie mit `ollama list`, ob Ollama läuft, oder starten Sie es mit `ollama serve`.
