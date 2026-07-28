# Architektur

- `cli.py`: Kommandozeilenoberfläche
- `pipeline.py`: Orchestrierung und Resume
- `detection.py`: YOLO, ByteTrack, Torwartauswahl und Kandidatenbildung
- `classification.py`: optionale Qwen-VL-Klassifikation
- `video.py`: FFmpeg/FFprobe
- `store.py`: SQLite-Zwischenspeicher
- `reporting.py`: JSON, CSV und HTML
- `config/default.yaml`: zentrale Konfiguration
