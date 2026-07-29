# Goalkeeper Highlights Studio

**Version 0.13.5**

Lokale CLI-Anwendung zur automatischen Erkennung und Erstellung von Torwart-Highlights aus Fußballvideos. Die Pipeline kombiniert YOLO11, ByteTrack, eine zeitbasierte Ereignislogik, SQLite, FFmpeg und optional Qwen-VL.

## Neu in 0.13.5

- **Korrekte Routing-Statistiken**: Qwen-Aufrufe werden nur gezählt, wenn das Modell tatsächlich ausgeführt wurde.
- **Getrennte Zähler**: Direkte HIGH-Annahmen, frühe LOW-Ablehnungen, erster Qwen-Durchlauf, zweiter Qwen-Durchlauf und durch den zweiten Durchlauf gerettete Highlights werden separat ausgewiesen.
- **Gemessene Laufzeiten**: Heuristisches Scoring sowie erster und zweiter Qwen-Durchlauf erhalten eigene Laufzeitwerte.
- **Bessere Diagnose**: Kandidaten speichern, welche Qwen-Durchläufe ausgeführt wurden, ob der zweite Durchlauf ein Highlight gerettet hat und wie lange die Klassifizierung dauerte.
- **Robuster Retry-Mechanismus**: Unsichere, nicht parsebare, kurze oder aus der Recovery stammende Kandidaten werden kontrolliert behandelt, ohne Endlosschleife.
- **Mehr Tests**: Erweiterte Testabdeckung für Routing, Retry-Entscheidungen und Performance-Zähler.

Diese Version wurde als Audit- und Stabilisierungsrelease gegen die Anforderungen aus 0.13.0 bis 0.13.3 geprüft. Zusätzlich ist die gezielte Analyse nur der letzten natürlich sortierten Quelldatei verfügbar.

## Neu in 0.13.3

- **Heuristisches Routing**: Intelligente Vorfilterung von Kandidaten in die Kategorien HIGH, MEDIUM und LOW.
- **Geschwindigkeitsoptimierung**: HIGH-Kandidaten werden direkt akzeptiert und LOW-Kandidaten früh verworfen, wodurch teure Qwen-Aufrufe eingespart werden.
- **Zweiter Qwen-Durchlauf**: Bei unsicheren Ergebnissen (MEDIUM oder unklare Konfidenz) erfolgt automatisch ein zweiter Durchlauf mit erweitertem zeitlichem Kontext (+2s vor/nach) und mehr Frames zur präziseren Analyse.
- **Detaillierte Statistiken**: Erweiterter Performance-Report mit routingbezogenen Kennzahlen (HIGH/MEDIUM/LOW, eingesparte Aufrufe, gerettete Highlights).
- **Flexible Konfiguration**: Alle Schwellenwerte für das Routing und den Retry-Mechanismus sind in der `default.yaml` anpassbar.

## Neu in 0.12

- komplette Spiele können aus einem Verzeichnis mit mehreren aufeinanderfolgenden Videodateien analysiert werden
- natürliche Dateisortierung, z. B. `MVI_0540.MP4`, `MVI_0541.MP4`, `MVI_0542.MP4`
- Dateien werden nur direkt aus dem angegebenen Verzeichnis gelesen, nicht rekursiv aus Unterordnern
- gemeinsame globale Timeline für Torwarterkennung, Ereignisse und dateiübergreifende Spielsituationen
- neue `source_manifest.json` mit Quelldateien, Dauer und globalen Zeitversätzen
- Einzeldatei-Aufrufe bleiben vollständig kompatibel

## Änderungen aus 0.10

- automatische Torwarterkennung in einem konfigurierbaren Anfangsfenster
- Bewertung über Trikotfarb-Einzigartigkeit, Kameranähe, Torbereich, Bewegungsmuster, Ballkontakte und Track-Persistenz
- manuelle Auswahl nur noch als Fallback bei zu geringer Konfidenz
- stabile fachliche Identität `Keeper #1`, unabhängig von wechselnden ByteTrack-IDs
- neue Debug-Ausgaben `analysis/goalkeeper_detection.html` und `analysis/goalkeeper_detection.json`
- bestätigter Torwart-Ballkontakt verhindert False Negatives bei ruhigen Abspielen und Klärungen
- neue Kategorie `keeper_clearance`
- niedrigere, event-spezifische Schwellen für `distribution` und `keeper_clearance`
- verworfene Kandidaten werden standardmäßig immer als Video und JSON exportiert
- `--no-export-rejected` deaktiviert den Export ausdrücklich
- schnelles FFmpeg-Input-Seeking: `-ss` steht vor `-i`
- automatische NVENC-Nutzung, falls `h264_nvenc` verfügbar ist
- parallele Clip-Erstellung, standardmäßig mit zwei Jobs
- finales Highlightvideo wird normalerweise per Stream Copy zusammengefügt
- Modi `accurate` und `fast`
- Performancewerte im HTML-Report und in `analysis/performance.json`
- verworfene Clips sind direkt im HTML-Report abspielbar

## Installation unter Windows

```powershell
cd C:\goalkeeper-highlights-studio
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Versionsprüfung:

```powershell
python -c "import importlib.metadata; print(importlib.metadata.version('goalkeeper-highlights'))"
```

Erwartete Ausgabe:

```text
0.13.5
```

## Empfohlener Analyselauf

```powershell
goalkeeper-highlights analyze `
  "C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4" `
  --decoder opencv `
  --frame-stride 2 `
  --overwrite
```

Verworfene Szenen landen standardmäßig unter `rejected/`. Nur bei ausdrücklichem Verzicht:

```powershell
goalkeeper-highlights analyze "C:\video\match.mp4" --no-export-rejected --overwrite
```


## Mehrere Videodateien aus einem Verzeichnis

Version 0.12.0 akzeptiert neben einer einzelnen Videodatei auch ein Verzeichnis:

```powershell
goalkeeper-highlights analyze `
  "C:\videorohdaten\Spiel_2026-07-27" `
  --decoder opencv `
  --frame-stride 2 `
  --overwrite
```

Unterstützte Formate sind `.mp4`, `.mov`, `.mkv`, `.m4v` und `.avi`. Es werden ausschließlich Dateien direkt im angegebenen Verzeichnis verarbeitet. Unterverzeichnisse, ZIP-Dateien und bereits erzeugte Dateien mit `_goalkeeper_highlights` im Namen werden ignoriert.

Die Sortierung erfolgt natürlich nach Dateinamen:

```text
MVI_0540.MP4
MVI_0541.MP4
MVI_0542.MP4
MVI_0543.MP4
MVI_0544.MP4
```

Die Dateien werden verlustfrei zu einer internen Timeline verbunden. Dadurch bleiben automatische Torwarterkennung, globale Zeitstempel und Situationen über eine Dateigrenze hinweg konsistent. Die Kamera-Dateien müssen dafür dieselbe Auflösung, Bildrate sowie Video- und Audio-Codecs verwenden. Das ist bei fortlaufenden Segmenten derselben Kamera normalerweise der Fall.

Der Ausgabeordner heißt bei einer Verzeichnisanalyse beispielsweise:

```text
Spiel_2026-07-27_goalkeeper_highlights/
```

Zusätzlich wird `source_manifest.json` erzeugt. Es enthält die Reihenfolge, Einzeldauer und globale Startzeit jeder Quelldatei.


### Nur die letzte Quelldatei analysieren

Für eine gezielte Nachanalyse des letzten Kamerasegments:

```powershell
goalkeeper-highlights analyze `
  "C:\videorohdaten\158_0726" `
  --only-last-source `
  --decoder opencv `
  --frame-stride 2 `
  --overwrite
```

Die Auswahl erfolgt nach derselben natürlichen Sortierung wie bei der vollständigen Verzeichnisanalyse. Standardmäßig wird in `<Verzeichnis>_last_source_goalkeeper_highlights` geschrieben.

## FFmpeg-Modi

### Accurate, Standard

```powershell
goalkeeper-highlights analyze "C:\video\match.mp4" --clip-mode accurate --encoder auto --parallel-jobs 2 --overwrite
```

- schnelles Input-Seeking
- bildgenauer Schnitt durch Re-Encoding
- automatische Wahl von `h264_nvenc`, sonst `libx264`
- finaler Zusammenschnitt ohne erneutes Encoding, sofern kompatibel

### Fast

```powershell
goalkeeper-highlights analyze "C:\video\match.mp4" --clip-mode fast --parallel-jobs 2 --overwrite
```

- nutzt `-c copy`
- sehr schnell
- Schnitt beginnt am passenden Keyframe und kann etwas früher einsetzen

## Ausgabe

```text
<video>_goalkeeper_highlights/
  clips/
  rejected/
  analysis/
    goalkeeper_detection.html
    goalkeeper_detection.json
    keeper_tracks.csv
    performance.json
    score_histogram.html
    timeline.html
  analysis.sqlite3
  events.csv
  events.json
  goalkeeper_highlights.mp4
  report.html
```


## Automatische Torwarterkennung

Vor der Ereigniserkennung sammelt Version 0.10 standardmäßig in den ersten acht Sekunden mehrere Beobachtungen pro Spieler. Die Auswahl erfolgt nicht anhand eines einzelnen Bildes, sondern aus mehreren Merkmalen:

- deutlich anderes Trikot als die Feldspieler
- große Bounding Box und Nähe zur hinter dem Tor stehenden Kamera
- Aufenthalt in einem konfigurierten Torbereich
- für Torhüter typisches Bewegungsmuster
- beobachtete Ballkontakte
- Persistenz des Tracks über das Anfangsfenster

Die Auswahl wird als stabile fachliche Identität `Keeper #1` gespeichert. ByteTrack darf später intern von Track 81 auf Track 214 wechseln; im Report bleibt es dennoch `Keeper #1`. Wenn die automatische Erkennung nicht sicher genug ist, öffnet sich weiterhin die manuelle Auswahl.

Konfiguration:

```yaml
keeper:
  automatic_initial_detection: true
  bootstrap_seconds: 8.0
  bootstrap_min_score: 0.48
  bootstrap_min_confidence: 0.52
  interactive_selection: true
```

Die Begründung der Auswahl steht nach dem Lauf in `analysis/goalkeeper_detection.html`.

## Ereignislogik

Version 0.10 bewertet unter anderem:

- Ballnähe und bestätigte Kontaktframes
- Ballkonfidenz
- Torwartidentität
- Anflug und Abflug
- Richtungswechsel
- Ballbesitzdauer
- Torwartbewegung
- hohen Ball beziehungsweise Flanke

Eine Sicherheitsregel akzeptiert bestätigten Torwart-Ballkontakt auch dann, wenn Bewegungs- oder Trajektoriemerkmale schwach sind. Das betrifft besonders kontrollierte Abspiele und Klärungen.

## CUDA und NVENC

YOLO verwendet bei `device: auto` eine verfügbare CUDA-GPU. Für FFmpeg prüft `encoder: auto`, ob `h264_nvenc` vorhanden ist. Kontrolle:

```powershell
ffmpeg -hide_banner -encoders | Select-String nvenc
```

Manuelle CPU-Nutzung:

```powershell
goalkeeper-highlights analyze "C:\video\match.mp4" --encoder libx264 --overwrite
```

## Entwicklung

```powershell
python -m pytest -q
```

Architektur- und Entwicklungsregeln stehen in `AGENTS.md` und `docs/architecture.md`. Geplante Arbeiten stehen in `ROADMAP.md`.

## Category-specific and dynamic clip windows (0.10.2)

Standalone actions use category-specific context. Catches receive more lead-in so the incoming ball is visible, while distributions end shortly after the pass. A later goalkeeper event can still join and extend the same phase of play.

```yaml
clips:
  category_pre_roll_seconds:
    catch_or_control: 10.0
    keeper_clearance: 8.0
    distribution: 5.0
  category_post_roll_seconds:
    catch_or_control: 4.0
    keeper_clearance: 4.0
    distribution: 2.0
  activity_tail_seconds: 0.0
  continuation_gap_seconds: 15.0
  final_keeper_contact_tail_seconds: 8.0
  max_dynamic_clip_seconds: 40.0
```

## Console output

Version 0.10.4 uses a clean single-line terminal display without internal tqdm counters. During analysis it shows percentage, processed video minutes, ETA, candidate count and realtime factor. Separate phases are shown for candidate selection, clip creation and final video generation. A structured summary is printed at the end. Detailed detector, profiling and FFmpeg output is available with:

```powershell
goalkeeper-highlights analyze "C:\video\match.mp4" --verbose
```

Clip windows are category-specific. `catch_or_control` starts up to 10 seconds before the trigger, while a standalone `distribution` uses a shorter 5-second pre-roll and 2-second post-roll. Chained goalkeeper situations still keep the longer dynamic ending.


### Example default output (0.10.4)

```text
Analyse     ██████████████████░░░░░░░░░░░░░░  56% | 24.0/42.9 min | ETA 10:41 | Kandidaten 4 | 1.77x

================================================================
Analyse abgeschlossen
================================================================

Video
  FCWittlinge-SFETeil1.mp4

Ergebnis
  Kandidaten:          18
  Highlights:          11
  Verworfen:           0
  Zusammengeführt:     7

Torwart
  Identität:           Keeper #1
  Konfidenz:           91 %
  Re-Identifikationen: 42

Leistung
  Analysezeit:         24:51
  Geschwindigkeit:     1.73× Echtzeit
  Gesamtzeit:          25:04

FFmpeg
  Clip-Erstellung:     00:11
  Zusammenfügen:       00:01
  Encoder:             h264_nvenc
  Clip-Modus:          accurate
  Parallele Jobs:      2

Erstellt
  ✓ Einzelclips
  ✓ Rejected-Clips
  ✓ Gesamtvideo
  ✓ HTML-Report
  ✓ CSV/JSON-Auswertung
  ✓ Analyse-Datenbank

Ausgabeverzeichnis
  C:\video\match_goalkeeper_highlights
================================================================
```

Use `--verbose` only when detector profiling, individual re-identifications or FFmpeg commands are needed.

## Virtuelle Multi-Datei-Timeline (0.12.0)

Bei einem Verzeichnis werden die Originalvideos natürlich nach Dateinamen sortiert und direkt nacheinander decodiert. Es wird **keine** temporäre `source_timeline.mp4` mehr erzeugt. Das spart Startzeit, SSD-Schreibzugriffe und mehrere Gigabyte temporären Speicher. Globale Zeitstempel, Qwen-Frames und Clips über Dateigrenzen bleiben erhalten.

Beispielreihenfolge:

```text
FCWittlingen-SFETeil11.MP4
FCWittlingen-SFETeil21.MP4
FCWittlingen-SFETeil22-Tonasync.mp4
```

Die Datei `source_manifest.json` dokumentiert die Reihenfolge, Dauer und globalen Zeitbereiche aller Quelldateien. Ohne `--verbose` zeigt die CLI eigene Phasen für Quellenprüfung, virtuelle Timeline, Analyse, Clip-Erstellung und Gesamtvideo.


## Source ordering check (0.13.0)

When a directory is analyzed, the exact natural filename order is printed before probing and analysis. This makes wrong installation paths or stale editable installs immediately visible. The runtime defensively sorts the source list twice and never relies on filesystem enumeration order.

### Robust source ordering (0.13.0)

For numbered camera files, the final numeric block in the filename defines the recording order. This deliberately tolerates small spelling differences in the text prefix, for example:

```text
FCWittlingen-SFETeil1.MP4
FCWittlingen-SFETeil21.MP4
FCWittlinen-SFETeil22-Tonasync.mp4
```

The resolved order is printed before probing and analysis.


## Action-aware clip planning (0.13.0)

Version 0.13.0 no longer cuts every highlight from a fixed number of seconds around the trigger. The temporal event engine records the first meaningful ball approach and the end of the detected goalkeeper action. The clip planner adds only category-specific context around those observed boundaries.

This improves both directions: a save can start early enough to show the shot and continue until the action is complete, while a simple distribution no longer contains a long idle preparation. A following goalkeeper event within the configured continuation gap extends the same clip, for example `distribution -> turnover -> shot -> catch`.

By default clips never cross from one source file into the next:

```yaml
clips:
  allow_cross_source_clips: false
```

Enable this only when camera files are known to be consecutive chunks without a pause. File changes representing half-time remain hard boundaries.

False positives from static or incorrect ball tracks are filtered by `clips.interaction_validation`. Fast saves with only one sampled contact frame have a separate conservative recovery rule under `event_engine.single_frame_save_*`.


## Recovery and robust decoding (0.13.1)

A second, generic pass scans the complete stored detection timeline for close keeper/ball geometry and meaningful keeper movement not already covered by an event. It does not contain game-specific filenames or timestamps. OpenCV decoding uses a higher FFmpeg packet-read limit and automatically reopens the source after recoverable read failures.
