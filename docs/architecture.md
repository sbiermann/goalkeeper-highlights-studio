# Architektur - Goalkeeper Highlights Studio

Das System ist als modulare Pipeline aufgebaut, die Videodaten in strukturierte Torwart-Highlights umwandelt.

## Kernkomponenten

- **`cli.py`**: Einstiegspunkt der Anwendung. Verarbeitet Kommandozeilenargumente und initialisiert die Pipeline.
- **`pipeline.py`**: Orchestriert den gesamten Ablauf von der Timeline-Erstellung bis zum fertigen Video. Verwaltet den Resume-Status via SQLite und aggregiert Performance-Statistiken.
- **`sources.py`**: Verwaltet die virtuelle Timeline (`VirtualTimelineDecoder`). Ermöglicht die nahtlose Analyse mehrerer Videodateien als eine logische Einheit.
- **`detection.py`**: Führt die Objekterkennung (YOLO11) und das Tracking (ByteTrack) durch. Beinhaltet den automatischen Torwart-Bootstrap-Prozess und die Re-Identifizierung.
- **`event_engine.py`**: Analysiert Trajektorien von Ball und Torwart. Berechnet heuristische Scores und identifiziert potenzielle Aktionen (Saves, Catches, Distribution, etc.).
- **`classification.py`**: Implementiert das intelligente Routing (seit Version 0.13.3; Statistik- und Diagnosekorrekturen in 0.13.4):
    - **HIGH**: Direkte Annahme bei hoher heuristischer Sicherheit.
    - **MEDIUM**: Analyse durch Qwen-VL. Bei Unsicherheit erfolgt ein zweiter Durchlauf mit erweitertem Kontext (+2s) und mehr Frames.
    - **LOW**: Frühes Verwerfen schwacher Kandidaten zur Performance-Optimierung.
- **`models.py`**: Zentrale Datenstrukturen (`Candidate`, `Box`). Hält alle Metadaten eines Highlights, inklusive Routing-Entscheidungen und Qwen-Ergebnissen.
- **`video.py`**: Wrapper für FFmpeg/FFprobe. Zuständig für framegenaues Schneiden, Skalieren und Zusammenfügen von Clips.
- **`store.py`**: Persistenzschicht mittels SQLite. Speichert Erkennungen, Kandidaten und den Verarbeitungsfortschritt.
- **`reporting.py`**: Generiert die finalen Berichte in den Formaten JSON, CSV und als interaktives HTML-Dashboard.

## Datenfluss

1. **Input**: Video oder Verzeichnis -> `sources.py` (Timeline)
2. **Erkennung**: `detection.py` (YOLO + ByteTrack + Keeper-Selection)
3. **Ereignisse**: `event_engine.py` (Kandidatenbildung + Scoring)
4. **Filterung**: `classification.py` (Heuristisches Routing HIGH/MEDIUM/LOW)
5. **KI-Validierung**: `classification.py` (Qwen-VL 1st Pass -> optional 2nd Pass bei MEDIUM)
6. **Output**: `video.py` (Clip-Export) -> `reporting.py` (Reports)

## Konfiguration

Zentrale Einstellungen befinden sich in `config/default.yaml` (und gespiegelt in `src/goalkeeper_highlights/default.yaml`). Hier werden Schwellenwerte für das Routing, Event-Kategorien und Hardware-Parameter (NVENC, Qwen-Quantisierung) gesteuert.
