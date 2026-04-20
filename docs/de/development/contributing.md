# Mitwirken (Contributing)

Beiträge zum LLM Client sind herzlich willkommen! Hier erfährst du, wie du dich beteiligen kannst.

## Workflow

1. **Forke** das Repository auf GitHub.  
2. **Klone** deinen Fork lokal.  
3. Erstelle einen **Feature-Branch**: `git checkout -b feature/mein-tolles-feature`.  
4. Implementiere deine Änderungen und schreibe **Tests**.  
5. Stelle sicher, dass alle Tests bestehen: `pytest`.  
6. Formatiere deinen Code: `black .` und `ruff check --fix .`.  
7. **Committe** deine Änderungen (Conventional Commits bevorzugt).  
8. **Pushe** den Branch in deinen Fork.  
9. Erstelle einen **Pull Request**.  

## Code-Stil

Wir verwenden:  
- **Black** für die Formatierung.  
- **Ruff** für Linting.  
- **Mypy** für Typprüfung.  
- **Google-Style** Docstrings.  

## Dokumentation

Wenn du neue Funktionen hinzufügst, aktualisiere bitte auch die entsprechende Dokumentation in `docs/`.
