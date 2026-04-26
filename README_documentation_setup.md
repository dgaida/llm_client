# Setup der Dokumentations-Infrastruktur

Um das vollständige Dokumentations-Ecosystem zu nutzen, führen Sie folgende Schritte aus:

1. **Abhängigkeiten installieren:**
   ```bash
   pip install interrogate mike mkdocs-material mkdocs-static-i18n mkdocstrings[python] mkdocs-mermaid2-plugin
   ```

2. **API-Abdeckung prüfen:**
   ```bash
   interrogate llm_client/
   ```

3. **Dokumentation lokal bauen:**
   ```bash
   mkdocs serve
   ```

4. **Versionierung mit mike (initial):**
   ```bash
   mike deploy --push --update-aliases 0.5.3 latest
   mike set-default --push latest
   ```
