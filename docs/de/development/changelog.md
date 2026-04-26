# Changelog Workflow

Wir verwenden [Conventional Commits](https://www.conventionalcommits.org/), um unser Changelog automatisch zu generieren.

## Commit-Nachrichten Format

Jeder Commit sollte diesem Format folgen:

\`\`\`
<typ>(<bereich>): <beschreibung>
\`\`\`

- **feat**: Ein neues Feature
- **fix**: Ein Bugfix
- **docs**: Änderungen an der Dokumentation
- **style**: Formatierungen, fehlende Semikolons, etc.
- **refactor**: Codeänderung, die weder einen Bug fixiert noch ein Feature hinzufügt
- **perf**: Codeänderung, die die Performance verbessert
- **test**: Hinzufügen von fehlenden Tests
- **chore**: Änderungen am Build-Prozess oder Hilfswerkzeugen

## Automatisierung

Bei jedem Release wird `git-cliff` ausgeführt, um die `CHANGELOG.md` basierend auf diesen Commits zu aktualisieren.
