# CLI-Nutzung

Der LLM Client verfügt über eine einfache Kommandozeilenschnittstelle (CLI).

## Installation

Die CLI ist nach der Installation des Pakets verfügbar:

```bash
pip install -e .
```

## Verwendung

### Einfache Anfrage

```bash
llm-client chat "Was ist die Hauptstadt von Frankreich?"
```

### Provider wählen

```bash
llm-client chat "Hallo" --provider groq
```

### Modell wählen

```bash
llm-client chat "Erkläre Quantenphysik" --provider openai --model gpt-4o
```

### Hilfe anzeigen

```bash
llm-client --help
```

---

## Konfiguration

Die CLI lädt API-Keys automatisch aus:  
1. Umgebungsvariablen  
2. `.env`-Datei im aktuellen Verzeichnis  
3. `secrets.env`-Datei  

## Beispiele

### Bildanalyse
```bash
llm-client chat "Was ist auf diesem Bild?" --provider gemini --file bild.jpg
```
