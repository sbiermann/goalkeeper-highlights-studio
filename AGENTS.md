# AGENTS.md

## Project goal
Goalkeeper Highlights Studio analyzes football video from a fixed camera behind a goal and creates local goalkeeper highlight clips.

## Supported environment
- Windows 11 is the primary platform.
- Python 3.12, NVIDIA CUDA when available, FFmpeg and FFprobe.
- The application must remain usable without cloud services.

## Pipeline
Video -> decoder -> YOLO11 -> ByteTrack -> automatic goalkeeper bootstrap -> keeper re-identification -> temporal event engine -> optional Qwen -> FFmpeg clips -> JSON/CSV/HTML/SQLite reports.

## Keeper identity
Version 0.10 first gathers multi-frame evidence during a configurable bootstrap window. Score candidates using shirt uniqueness, apparent camera proximity/box size, goal-region occupancy, low-movement pattern, ball contacts and track persistence. Do not fall back to a single-frame choice unless automatic confidence is insufficient and interactive selection is disabled.

Use a stable logical label (`Keeper #1`) instead of exposing ByteTrack IDs. Re-identification weights are: shirt-color contrast 45%, goal-region presence 30%, temporal continuity 15%, similarity to the initially selected shirt 10%. Never hardcode one goal side for an entire match. ByteTrack IDs are diagnostic implementation details and must not replace the stable domain identity. Keep `analysis/goalkeeper_detection.html` and JSON output explainable when selection logic changes.

## Event engine
Supported heuristic categories include catches, high claims/crosses, punch clearances, saves/deflections, diving saves, sweeping/one-on-one actions and distribution. Version 0.8 uses category-specific thresholds, possession bonuses and explainable score components.

## Development rules
- Keep Python type annotations and dataclasses.
- Use logging or existing progress output; avoid new ad-hoc print statements.
- Keep configuration defaults in both `config/default.yaml` and packaged `src/goalkeeper_highlights/default.yaml` synchronized.
- Add or update tests for scoring and serialization changes.
- Update README.md, AGENTS.md, ROADMAP.md and CHANGELOG.md for every release.
- Preserve backward compatibility for existing SQLite candidate payloads where practical.

## Console logging and clip tails (0.10.1)

Do not emit one console line per ByteTrack re-identification in normal mode. Aggregate re-identification counts and confidence statistics and expose detailed track switches only behind `keeper.reid_verbose_console`. Keep single-event clips concise with `activity_tail_seconds`; use `final_keeper_contact_tail_seconds` only for genuinely chained phases of play. Preserve the explainable goalkeeper report whenever confidence stabilization or score weights change.

## Console and clip-window rules (0.10.2)

- Default CLI output must stay quiet: one progress bar plus a final summary.
- New diagnostic output belongs behind `--verbose` / `runtime.verbose_console`.
- Never print FFmpeg commands or detector profiling in default mode.
- Clip windows are event-type specific. Preserve longer context before catches and keep standalone distributions short.
- Dynamic chaining may extend clips only when a later goalkeeper event belongs to the same phase of play.


## Terminal UI rules (0.10.4)

- Do not expose library-native progress counters, per-mille units, iteration rates or arbitrary totals such as `29/1000`.
- Normal mode must use one continuously refreshed line per active phase.
- User-facing analysis progress should prefer video time, percentage, ETA, candidate count and realtime factor.
- Start a new visible phase for selection, clip generation and concatenation instead of mixing phase details into one opaque status string.
- Print a structured final summary after the progress line is closed.
- Preserve detailed diagnostics exclusively behind `--verbose`.

## Multi-source timeline (0.12.0)

- `analyze` accepts a single supported video or a directory of sequential videos.
- Directory discovery must stay non-recursive unless a future explicit CLI flag enables recursion.
- Sort source files using natural filename ordering, never plain lexical ordering.
- Ignore generated videos containing `_goalkeeper_highlights` in the stem.
- Build one logical continuous timeline before detection so keeper identity and event timing do not reset per file.
- Keep `source_manifest.json` stable and backward-compatible; it is the mapping between local source timestamps and global match timestamps.
- Source concatenation is lossless stream-copy. Do not silently re-encode a full match; fail clearly when camera segments are incompatible.

## Virtual timeline rules

- Directory sources are ordered with `sources.natural_sort_key`; never rely on filesystem iteration order.
- Do not create a concatenated source video. Detection must use `VirtualTimelineDecoder`.
- Candidate timestamps are global; classification and clip extraction must map them through `SourceManifest.locate`.
- A clip may cross source boundaries and must be assembled from frame-accurate source segments.
- Keep `source_manifest.json` as the reproducibility contract.


## Source ordering invariant (0.13.0)
For multi-file camera input, the final numeric filename block is the primary recording sequence. Do not let spelling differences in filename prefixes override numeric segment order.

## Clip-boundary invariant (0.13.0)

Clip windows must be based on `Candidate.action_start` and `Candidate.action_end`, not only `trigger_time`. Category pre/post values are context margins around observed activity. Do not reintroduce one global fixed window. Source boundaries are hard by default and may only be crossed when `allow_cross_source_clips` is explicitly enabled. Interaction-validation rejections must preserve their reason for reports and rejected review clips.


## Recovery-pass invariant (0.13.1)

Missed-action recovery must operate on generic geometry over the complete virtual timeline. Never add recording-specific filenames or timestamps. Existing candidate windows are a negative mask to avoid duplicate highlights. Decoder recovery must remain transparent in normal mode and observable through summary statistics.

## Version 0.13.4

Maintenance release for corrected routing statistics, explicit Qwen pass accounting, retry diagnostics, and extended routing tests. Detection, clip planning, recovery logic, and routing thresholds remain unchanged from 0.13.3.


## Version 0.13.5
Verified audit/stabilization release with last-source analysis and an explicit feature verification matrix.

## Version 0.13.6 diagnostic contract

The primary optimization target is recall of missed goalkeeper saves. Every successful analysis must create a non-video debug ZIP containing candidate decisions, the SQLite analysis database, effective configuration, reports, profiling information, and uncovered suspicious timeline windows. Do not add hard-coded match timestamps or filenames.


## v0.13.7 invariants
- Keep the logical keeper identity independent from temporary ByteTrack IDs.
- Never hard-code match timestamps or filenames.
- Preserve all candidate lifecycle stages in the video-free debug package.

## Version 0.13.18

- Feintuning des adaptiven `catch_or_control`-Idle-Tails: LOW bleibt 3s, MEDIUM steigt auf 6s, HIGH auf 7s.
- LOW/MEDIUM/HIGH-Klassifizierungslogik bleibt unverändert; keine neuen Heuristiken.
- `catch_control_max_post_roll_seconds` bleibt unverändert bei 11s.
- Boundary-Priorität bleibt unverändert: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → dynamic idle tail → `timeout`.
- Regressionen der bestehenden Boundary-Mechanismen bleiben geschützt.

## Version 0.13.19

- Multi-Source-State-Isolation ist verpflichtend: Beim Source-Wechsel müssen transiente Tracking-/Interaktionszustände sauber zurückgesetzt werden.
- Globale Timeline bleibt führend; source-lokale Zeit darf nur explizit und konsistent zusätzlich geführt werden.
- Semantische Keeper-Identität (`Keeper #1`) bleibt von source-lokalem Tracker-State getrennt.
- Source-Diagnostik muss pro Quelle Frames, Keeper-/Ball-Frames, Candidate-Zahlen und Reset-Status nachvollziehbar ausgeben.
- Event-/Boundary-/Threshold-Logik aus 0.13.18 bleibt fachlich unverändert.

## Version 0.13.21

- Fokus ist ein isolierter FP32-vs-FP16-A/B-Test für YOLO-Inference auf CUDA; keine weiteren Performance-Refactorings in derselben Version.
- FP16 darf nur effektiv aktiv werden, wenn CUDA verfügbar ist; sonst muss ein sauberer FP32-Fallback mit Diagnosefeldern (`requested_fp16`, `effective_fp16`, `fp16_fallback_reason`) erfolgen.
- `model.track(..., half=True)` nur bei effektivem FP16 setzen; FP32-Pfad unverändert lassen.
- Benchmark-Vergleiche müssen denselben Decoderpfad und denselben Box-Converter-Modus verwenden (Packed vs. Packed).
- Event-, Candidate-, Recovery- und Boundary-Logik inklusive Thresholds bleiben unverändert.

## Version 0.13.22

- Ziel ist die isolierte Untersuchung/Optimierung des innerhalb von `model.track()` gemessenen Track-/Framework-Overheads ohne fachliche Logikänderungen.
- Technische Untersuchung umfasst Legacy-vs-Optimized-Trackpfad, `_TrackRunner`, offiziellen persistenten Predictor-Pfad und `runtime.track_execution_mode` mit `legacy`/`optimized`.
- Zusätzliches Overhead-Profiling umfasst `track_predictor_setup_ms`, `track_tracker_update_ms`, `track_result_build_ms`, `track_callbacks_ms`, `track_framework_other_ms`.
- Reale A/B-Methodik: `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4`, `start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `precision=FP32`, `boxes_from_result_mode=packed`; vier serielle Läufe (`legacy_run1`, `optimized_run1`, `legacy_run2`, `optimized_run2`).
- Finale Mediane: Legacy `analysis_seconds=133.4865`, `FPS=11.29`, `model_track_wall_ms=65.327`; Optimized `analysis_seconds=133.495`, `FPS=11.2665`, `model_track_wall_ms=66.9565`.
- Legacy→Optimized-Differenz: Analysezeit praktisch unverändert (~`-0.006%`), FPS ~`-0.208%`, `model_track_wall_ms` ~`-2.49%` (also langsamer).
- Fachliche Äquivalenz in den A/B-Läufen bestätigt: identische `candidates`, `accepted`, `rejected`, Keeper-Identität, Candidate-Timings sowie Detection-/Track-ID-Zählwerte.
- Entscheidung: persistenter Predictor-/Optimized-Pfad wird mangels belastbarem Performancevorteil nicht als Performance-Standard erklärt; die Setup-Overhead-Hypothese wurde geprüft, aber nicht als relevanter Hebel bestätigt.
- FP32 bleibt Standard; FP16 wurde bereits in 0.13.21 real getestet und wegen schlechterer Performance verworfen und ist kein 0.13.22-Optimierungsweg.
- Die Packed-Result-Conversion aus 0.13.20 bleibt unverändert bestehen.
- Finaler Teststand: `176 passed`, `0 failed`, `0 skipped`.

## Version 0.13.23

- Ziel ist die isolierte Untersuchung eines asynchronen OpenCV-Decoder-Prefetch ohne Änderungen an Event-/Candidate-/Boundary-/Recovery-/Threshold-Logik.
- Neuer Decoder-Ausführungsmodus `runtime.decoder_execution_mode` mit `legacy`/`prefetch`; konfigurierbare begrenzte Queue über `runtime.decoder_prefetch_queue_size` (Default `4`).
- Prefetch-Architektur: ein Decoder-Producer-Thread liest Frames in eine bounded Queue; Detection/Tracking (`model.track`) verbleibt strikt im Main-Thread (keine parallelen Track-Aufrufe).
- Explizite Decoder-Sentinel-Semantik für `source_end`, `global_end` und `exception`; Exceptions werden kontrolliert an den Main-Thread propagiert, inkl. sauberem Shutdown.
- Prefetch-Profiling ergänzt um `decoder_read_ms`, `decoder_queue_wait_ms`, `consumer_queue_wait_ms`, `decoder_prefetch_frames`, `decoder_queue_max_depth`.
- Reale A/B-Methodik: `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4`, `start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `precision=FP32`, `boxes_from_result_mode=packed`; vier serielle Läufe (`legacy_run1`, `prefetch_run1`, `legacy_run2`, `prefetch_run2`).
- Mediane: Legacy `analysis_seconds=106.6535`, `FPS=14.0745`, `loop_ms=60.036`, `decoder_next_ms=9.2705`; Prefetch `analysis_seconds=95.1685`, `FPS=15.772`, `loop_ms=61.4725`, `decoder_next_ms=0.1025`.
- Legacy→Prefetch-Differenz: Analysezeit `+10.77%`, FPS `+12.06%`, wahrgenommene Decoder-Wartezeit (`decoder_next_ms`) `+98.89%` verbessert; `model_track_wall_ms` blieb fachlich unbeeinflusst, aber im Median leicht höher (`-2.44%` gegenüber Legacy).
- Fachliche Äquivalenz in den A/B-Läufen bestätigt: identische `processed_frames` (`1501`), `candidates` (`2`), `accepted` (`1`), `rejected` (`1`), `merged` (`0`), Keeper-Identität (`Keeper #1`) sowie keine Unterschiede in den exportierten Candidate-Timing-/Boundary-Feldern.
- FP32 bleibt Standardpräzision, FP16 bleibt gemäß 0.13.21 verworfen; der experimentelle persistente Predictor-Pfad aus 0.13.22 bleibt nicht Performance-Default.
- Ergebnis/Entscheidung: Prefetch wird für den OpenCV-Pfad als neuer Runtime-Default aktiviert (`decoder_execution_mode: prefetch`), da A/B-Medianvorteil klar über Messrauschen liegt und keine fachlichen Abweichungen nachgewiesen wurden.
- Finaler Teststand: `180 passed`, `0 failed`, `0 skipped`.

## Version 0.13.24

- Ziel ist die isolierte, feinere Untersuchung des verbleibenden Ultralytics/`model.track()`-Framework-Overheads bei unveränderter Fachlogik.
- Decoder-Prefetch aus 0.13.23 bleibt unverändert Default (`runtime.decoder_execution_mode: prefetch`), FP32 bleibt Standard, FP16 bleibt verworfen.
- Track-Framework-Profiling wurde auf folgende Unterstufen erweitert: `track_callback_ms`, `track_predictor_pre_ms`, `track_predictor_post_ms`, `track_tracker_update_ms`, `track_result_build_ms`, `track_result_wrap_ms`, `track_ultralytics_misc_ms` (plus Restfeld `track_framework_other_ms`).
- Reale 4x-A/B-Methodik: `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4`, `start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `decoder_execution_mode=prefetch`, `precision=FP32`, `boxes_from_result_mode=packed`, `track_path=legacy|optimized`.
- Finale Mediane: Legacy `analysis_seconds=96.6955`, `FPS=15.5535`, `model_track_wall_ms=54.0835`; Optimized `analysis_seconds=90.011`, `FPS=16.703`, `model_track_wall_ms=51.0025`.
- Legacy→Optimized-Differenz: Analysezeit `+6.91%`, `model_track_wall_ms +5.70%`, `track_overhead_ms +7.94%`; damit messbarer End-to-End-Vorteil bei fachlicher Äquivalenz.
- Primärer Framework-Teilblock bleibt `track_callback_ms` (dominant innerhalb `track_overhead_ms`), insbesondere `track_predictor_pre_ms`; `track_tracker_update_ms`/`track_result_*` waren in den gemessenen Läufen nicht separat >0 messbar.
- Fachliche Äquivalenz in den A/B-Läufen bestätigt: identische `processed_frames` (`1501`), `candidates` (`2`), `accepted` (`1`), `rejected` (`1`), `merged` (`0`), Keeper-Identität (`Keeper #1`) sowie keine Candidate-Timing-/Boundary-Differenzen.
- Entscheidung: Der optimierte Track-Framework-Pfad kann als neuer Runtime-Default für `track_execution_mode` verwendet werden; Event-/Candidate-/Recovery-/Boundary-/Threshold-Logik bleibt unverändert.

## Version 0.13.25

- Ziel ist die isolierte Folgestufe auf Basis 0.13.24: verbleibenden Hotspot innerhalb `track_callback_ms` (insb. `track_predictor_pre_ms`) tiefer messbar machen, ohne fachliche Logikänderungen.
- Decoder-Prefetch aus 0.13.23 bleibt unverändert Default (`runtime.decoder_execution_mode: prefetch`), FP32 bleibt Standard, FP16 bleibt verworfen, Packed-Result-Conversion bleibt aktiv.
- Callback-/Pre-Profiling wurde verfeinert um: `track_callback_dispatch_ms`, `track_callback_predict_start_ms`, `track_callback_batch_start_ms`, `track_callback_postprocess_end_ms`, `track_callback_batch_end_ms`, `track_callback_predict_end_ms`, `track_callback_other_ms`, `track_pre_source_setup_ms`, `track_pre_batch_prepare_ms`, `track_pre_other_ms`.
- Reale 4x-A/B-Methodik: `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4`, `start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `decoder_execution_mode=prefetch`, `precision=FP32`, `boxes_from_result_mode=packed`; vier serielle Läufe (`legacy_run1`, `optimized_run1`, `legacy_run2`, `optimized_run2`).
- Finale Mediane: Legacy `analysis_seconds=92.075`, `FPS=16.3025`, `model_track_wall_ms=52.53`, `track_overhead_ms=22.338`; Optimized `analysis_seconds=90.4295`, `FPS=16.599`, `model_track_wall_ms=51.384`, `track_overhead_ms=21.6965`.
- Legacy→Optimized-Differenz: Analysezeit `+1.79%`, `model_track_wall_ms +2.18%`, `track_overhead_ms +2.87%` (kleiner, aber stabiler Gewinn über zwei serielle Paare).
- Hotspot-Befund: dominanter Anteil bleibt `track_predictor_pre_ms`, nahezu vollständig in `track_callback_predict_start_ms` (`~17.1 ms/frame`); `track_callback_batch_start_ms` ist vernachlässigbar (`~0.001 ms/frame`).
- Fachliche Äquivalenz in den A/B-Läufen bestätigt: identische `processed_frames` (`1501`), `candidates` (`2`), `accepted` (`1`), `rejected` (`1`), `merged` (`0`), Keeper-Identität (`Keeper #1`) sowie keine Candidate-Timing-/Boundary-Differenzen in den Exporten.
- Entscheidungsstand: kein riskanter semantischer Eingriff in Tracker-/Event-Pfade; 0.13.25 liefert primär tiefere Hotspot-Transparenz plus kleinen reproduzierbaren Performancegewinn innerhalb des etablierten FP32/Prefetch/Packed-Pfads.
- Finaler Teststand: `181 passed`, `0 failed`, `0 skipped`.

## Version 0.13.26

- Ziel war ein isolierter YOLO-Inference-Backend-Vergleich (PyTorch FP32 vs. TensorRT FP32) ohne Änderungen an Event-/Candidate-/Boundary-/Recovery-/Threshold-Logik.
- Basis-Konfiguration für die A/B-Läufe: `decoder=opencv`, `runtime.decoder_execution_mode=prefetch`, `precision=FP32`, `runtime.track_execution_mode=optimized`, `frame_stride=2`.
- Neuer expliziter Backend-Schalter: `yolo.backend` mit `pytorch|tensorrt|onnx`, plus Diagnosefelder `requested_backend`, `effective_backend`, `backend_fallback_reason`.
- TensorRT blieb optional: installiert wurden `tensorrt-cu12==10.13.3.9`, `tensorrt_cu12_bindings==10.13.3.9`, `tensorrt_cu12_libs==10.13.3.9`, transitive Runtime `nvidia-cuda-runtime-cu12==12.9.79`.
- Verifizierte Umgebung: Python `3.12.10`, PyTorch `2.13.0+cu126`, CUDA Runtime `12.6`, cuDNN `91002`, Ultralytics `8.4.109`, OpenCV `5.0.0`, GPU `NVIDIA GeForce RTX 3060 Ti`, Treiber `610.62`.
- TensorRT-Engine-Cache ist aktiv (`models/cache/...engine`) mit Build/Load-Trennung: `engine_build_seconds` und `backend_load_seconds` werden separat erfasst.
- Frühe TensorRT-Tests vor Installation liefen kontrolliert als Fallback (`requested_backend=tensorrt`, `effective_backend=pytorch`), keine falsche TensorRT-Klassifikation.
- Echte TensorRT-FP32-Läufe nach Installation:
  - `tmp_bench_01326_tensorrt_run1`: `analysis_seconds=215.567`, `processed_fps=6.963`, `effective_backend=tensorrt`, `engine_cached=false`, `engine_build_seconds=146.362984`.
  - `tmp_bench_01326_tensorrt_run2`: `analysis_seconds=69.135`, `processed_fps=21.711`, `effective_backend=tensorrt`, `engine_cached=true`, `engine_build_seconds=0.0`.
- PyTorch-FP32-Referenzläufe:
  - `tmp_bench_01326_pytorch_run1`: `analysis_seconds=75.688`, `processed_fps=19.831`.
  - `tmp_bench_01326_pytorch_run2`: `analysis_seconds=69.549`, `processed_fps=21.582`.
- Median auf den vier seriellen Läufen (inkl. TensorRT-Initialbuild in Run1) zeigt keinen belastbaren TensorRT-Gesamtvorteil; ohne Build-Effekt (`run2` gegen `run2`) liegt TensorRT nur nahe am PyTorch-Niveau.
- Fachlicher Hinweis aus den gemessenen Läufen: High-Level-Zähler (`candidates=2`, `accepted=1`, `rejected=1`, Keeper `Keeper #1`) bleiben gleich, aber `merged` weicht in den TensorRT-Läufen (`1`) gegenüber PyTorch (`0`) ab und verhindert aktuell eine vollständige Äquivalenzfreigabe.
- ONNX wurde nicht als Primärpfad bewertet; ONNX-Runtime wurde nur als Ultralytics-Exportabhängigkeit im TensorRT-Exportpfad nachinstalliert.
- Entscheidung 0.13.26: PyTorch bleibt sicherer Default. TensorRT bleibt optionaler Research-Pfad mit Fallback auf PyTorch.

## Version 0.13.29

- Ziel war ein credit-effizienter Boundary-Release ausschließlich für die ersten sieben chronologischen Final-Candidates (inklusive rejected) des v0.13.28-Referenzlaufs.
- Fachfokus: Clip-1/3/4/5-Boundaries korrigieren, Clip 2 explizit rejected belassen, Clip 6/7 als Regression stabil halten.
- Implementierte allgemeine Core-Regel: schwache akzeptierte Folgephasen können in den Vorgänger absorbiert werden, ohne das Clipfenster unnötig bis zum Maximal-Tail aufzublähen.
- Catch/Control-Boundary-Feintuning ergänzt um: isolierten Dynamic-Idle-Core-Tail, konservative Kappung langer gemergter Dynamic-Idle-Phasen sowie trailing-core Zuschnitt für lange gemergte Controlled-Release-Phasen.
- `analyze` akzeptiert optional `--duration` für kurze reale Validierungsläufe; ohne Parameter bleibt das bisherige Default-Verhalten (Vollanalyse) unverändert.
- Neue Boundary-Core-Defaults in `config/default.yaml` und `src/goalkeeper_highlights/default.yaml` synchronisiert; Event-/Candidate-/Recovery-/Threshold-Logik blieb unverändert.
- Debug-Paketname angehoben auf `goalkeeper_highlights_debug_v0.13.29.zip`.

## Version 0.13.30

- Credit-effizienter Boundary-Abschluss nur für den ersten chronologischen Block 1–7 bei unveränderter Event-/Candidate-/Recovery-/Threshold-Logik.
- `catch_or_control` erlaubt für kompakte isolierte Cores eine asymmetrische Context-Verschiebung (weniger Pre-Roll, mehr Post-Roll), ohne Action-Fenster abzuschneiden und ohne globale Category-Before/After-Änderung.
- Kurze isolierte `keeper_clearance`-/Distribution-artige Aktionen werden ohne echte Folgephase auf konservativen Safety-Tail begrenzt; mit Continuation bleibt längerer Kontext erlaubt.
- Stabilitätsinvariante für den Block 1–7: `raw-0001`, `raw-0004` (rejected), `raw-0006`, `raw-0012`, `raw-0015` bleiben unverändert; nur `raw-0005` und `raw-0020` erhalten Boundary-Anpassung.
- Neue Boundary-Defaults in `config/default.yaml` und `src/goalkeeper_highlights/default.yaml` synchronisiert; keine Candidate-ID-/Timestamp-Hardcodings.
- Debug-Paketname angehoben auf `goalkeeper_highlights_debug_v0.13.30.zip`.

## Version 0.13.31

- Credit-effizienter Korrektur-Release nur für die Zielklassen um Clip 12/15/17/20; keine Realvideo-Läufe durch Junie (`MAX_REAL_VIDEO_RUNS=0`) und keine Performancearbeit.
- Restart-Relevance-Guard bleibt aktiv, akzeptiert aber starke verifizierte Distribution-/Release-Phasen als konservative Ausnahme statt pauschalem `irrelevant_outside_box_restart`.
- Kontextuelle Recovery-Rescue ergänzt: kompakte, evidenzgestützte `recovery_uncovered_activity`-Fenster können trotz strengem Recovery-Interaction-Threshold akzeptiert werden; kontextlose Fälle bleiben rejected.
- Boundary-Core-Auswahl erweitert: kleine Tail-Erhaltung für restart-gerettete Distributionen, kompakter Core für lange Multi-Distribution-Phasen und kompakteres Fenster für akzeptierte Recovery-Rescues.
- Lange stark gemergte `catch_or_control`-Final-Overlap-Phasen werden konservativ auf einen relevanten Core begrenzt statt nur auf technisches Max-Limit geklemmt.
- Neue gezielte Regressionstests für Restart-/Recovery-/Boundary-Regeln ergänzt; Debug-Paketname via Version auf `goalkeeper_highlights_debug_v0.13.31.zip` angehoben.

## Version 0.13.27

- Ziel war ein credit-effizienter Low-Hanging-Fruit-Sweep auf dem bestehenden Produktionspfad (PyTorch FP32, Packed Conversion, OpenCV-Prefetch, optimierter Track-Pfad) ohne fachliche Logikänderungen.
- Untersuchte Kandidaten: `TF32`, `torch.backends.cudnn.benchmark`, `imgsz=576`, `imgsz=512`.
- Implementiert wurden nur kleine Benchmark-/Runtime-Schalter: `--tf32`, `--cudnn-benchmark`, `--imgsz`.
- 60s-Screening (`FCWittlinge-SFETeil1.MP4`, `start=0`, `duration=60`, `frame_stride=2`):
  - Baseline `42.07s`, `17.85 FPS`
  - `TF32`: `38.96s`, `19.28 FPS` (`+7.40%`)
  - `cuDNN benchmark`: `45.40s`, `16.54 FPS` (`-7.92%`, verworfen)
  - `imgsz=576`: `27.39s`, `27.42 FPS` (`+34.91%`)
  - `imgsz=512`: `25.88s`, `29.02 FPS` (`+38.48%`)
- 120s-Bestätigung (`start=0`, `duration=120`): Baseline `68.55s`, `21.90 FPS`; `TF32` `66.28s`, `22.64 FPS` (`+3.31%`), `imgsz=576` (`1/1/0`) und `imgsz=512` (`0/0/0`) mit Candidate-Abweichungen gegenüber Baseline (`2/1/1`) verworfen.
- Teil22-Fachvalidierung für verbleibenden Kandidaten `TF32` (`FCWittlinen-SFETeil22-Tonasync.mp4`, `start=540`, `duration=220`): Baseline und `TF32` beide `candidates=4`, `accepted=3`, `rejected=1`, Keeper `Keeper #1`.
- Entscheidung 0.13.27: kein Default-Wechsel, da einzig fachlich stabiler Kandidat (`TF32`) auf 120s unter der Übernahmeschwelle blieb; Baseline-`image_size` und sonstiger Produktionspfad bleiben unverändert.
- Fachliche Invarianten bestätigt: keine Änderungen an Detection-Thresholds, Tracking-Parametern, Event-/Candidate-/Boundary-/Recovery-Logik.

## Version 0.13.28

- Ziel war ein credit-effizienter Release für Benchmark-Reproduzierbarkeit plus genau ein kurzer Hotspot-Screening-Lauf ohne fachliche Logikänderungen.
- Benchmark-Läufe erzwingen standardmäßig nicht-interaktive Keeper-Auswahl über `runtime.benchmark_force_noninteractive_keeper_selection=true`, damit manuelle Klick-Wartezeit A/B-Messungen nicht verfälscht.
- Interaktive Keeper-Wartezeit wird separat erfasst (`interactive_wait_seconds`) und aus `analysis_seconds`/`realtime_factor` herausgerechnet.
- Benchmark-Metadaten wurden kompakt erweitert um `keeper_selection_mode` und `keeper_selection_timed`.
- 60s-Screening auf Produktionspfad (`FCWittlinge-SFETeil1.MP4`, `start=0`, `duration=60`, `frame_stride=2`, `decoder=opencv`, `decoder_execution_mode=prefetch`, `precision=FP32`, `boxes_from_result_mode=packed`, bestehende `image_size`) ergab `analysis_seconds=43.879` und `processed_fps=17.115`.
- Hotspot-Ranking bestätigt weiterhin `model.track`/Framework als dominanten Block; Input-Handling-nahe Blöcke (`yolo_preprocess_ms`, `boxes_from_result_ms`) zeigen in diesem Stand keinen einfachen lokal umsetzbaren >=5%-Hebel.
- Entscheidung 0.13.28: kein neuer Performance-Default; Thresholds, Tracking-Semantik sowie Event-/Candidate-/Boundary-/Recovery-Logik bleiben unverändert.

## v0.13.8 invariants

- Keeper bootstrap ranks logical identities, not isolated ByteTrack IDs.
- Automatic fallback reasons and ranking history must remain in the debug package.
- Every candidate must keep a stable id and merge ancestry.
- Debug archives must never contain video files.
- OpenCV read-attempt configuration must be applied before importing cv2.

## v0.13.10 invariants

- Merge related candidates within 2.5s or with continuous possession flow to prevent fragmented clips.
- Clips must only end after a detected restart (kick/throw) if ball control was established.
- Interaction validation must use dynamic trajectory features (interaction_score) to reduce false positives.
- Every candidate must record its clip_end_reason and merge ancestry.
- HTML reports must display interaction scores and merge reasons.


## Version 0.13.12

- Robuste Fortsetzungs-Absorption für Recovery-Kandidaten basierend auf nachfolgenden Distribution-Events.
- Behebt das Problem zu kurzer Clips bei unzuverlässiger `possession_duration` (z.B. Clip 5).
- Optionale Absorption von `rejected` Distribution-Candidates als Boundary-Evidenz.
- Zusätzliche Diagnostik für die Recovery-Distribution-Fortsetzung.
- Alle Invarianten aus v0.13.11 und v0.13.10 bleiben erhalten.

## Version 0.13.13

- Kontrollierter `recovery_window_tail`-Fallback für akzeptierte Recovery-Aktionen mit unvollständigem Timeout-Ende.
- Reihenfolge der Clip-End-Entscheidung: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → `timeout`.
- Fallback nutzt vorhandene Recovery-Boundary-Evidenz und verlängert nicht pauschal alle Recovery-Clips.
- Zusätzlicher Tail bleibt konservativ begrenzt (`recovery_window_tail_max_extension_seconds`) und respektiert weiterhin die maximale dynamische Clipdauer.
- `action_end` bleibt das Ende der erkannten Aktion; nur `clip_end` darf über Recovery-Boundary-Evidenz verlängert werden.
- Erweiterte numerische Diagnostik (`recovery_tail_*`) und merge-stabile Recovery-Window-Metadaten für das videofreie Debug-Paket.

## Version 0.13.17

- Adaptiver `catch_or_control`-Idle-Tail mit LOW/MEDIUM/HIGH-Stufen ersetzt den festen 3s-Tail.
- Einstufung basiert auf kombinierter Keeper-/Ball-Evidenz (`contact_frames`, `possession_duration`, ergänzend `interaction_score`) statt auf einem Einzelwert.
- Keeper-Motion allein bleibt unzureichend für MEDIUM/HIGH.
- Boundary-Priorität bleibt unverändert: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → dynamic idle tail → `timeout`.
- Diagnostik erweitert um numerische Idle-Level- und Schwellen-Match-Felder.

## Version 0.13.16

- Strengere Continuation-Absorption für schwache Recovery-Kandidaten: Keeper-Motion allein genügt nicht.
- Recovery-Continuation erfordert belastbare Keeper-/Ball-Evidenz (Kontakt/Ball-Dynamik/Ballkontrolle/Distribution-Signal) und nutzt den `interaction_score`.
- `catch_or_control`-Post-Roll ist dynamisch: 11s bleiben Maximalwert, bei Inaktivität wird konservativ früher beendet.
- Priorität bleibt erhalten: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → dynamic idle tail → `timeout`.
- Erweiterte numerische Diagnostik für Recovery-Continuation und dynamischen Idle-Tail.

## Version 0.13.15

- Finaler Overlap-Dedup erweitert um konservatives Context-Trimming bei zu langer Union durch äußeren Kontext.
- Vollständiges Action-Fenster bleibt erhalten; reduziert werden nur überschüssige Pre-/Post-Roll-Anteile.
- Keine pauschale Erhöhung globaler Dauergrenzen; bestehende `max_dynamic_clip_seconds` und Toleranz bleiben gültig.
- Merge-/Recovery-/Boundary-Metadaten bleiben beim finalen Merge erhalten und diagnostisch nachvollziehbar.

## Version 0.13.14

- Finaler Overlap-Dedup-Pass nach allen Clip-Boundary-Erweiterungen zur Konsolidierung nahezu identischer Highlights derselben Keeper-Phase.
- Merge nur bei kompatibler Keeper-Identität, starker Überlappung oder sehr kleiner Lücke und ohne unabhängigen Restart.
- Union-Clip respektiert weiterhin `max_dynamic_clip_seconds` inklusive vorhandener Toleranz; kein aggressives Re-Trimming.
- Merge-Ancestors (`merged_from`, `parent_candidate_ids`) und Recovery-/Boundary-Diagnostik bleiben erhalten.
- Absorbierte Kandidaten werden nicht als separater Highlight-Export ausgegeben, bleiben aber diagnostisch nachvollziehbar.

## Version 0.13.11

- Verbesserte Erkennung kontrollierter Ballfreigaben (Abschläge, Abwürfe, Pässe) nach Ballbesitz.
- Verlängerung der Clip-Fenster für kontrollierte Freigaben mit konfigurierbarem Safety-Tail.
- Zusätzliche Diagnostik für die Freigabe-Erkennung im `score_breakdown`.
- Alle Invarianten aus v0.13.10 (Phase-Merge, Interaction Validation) bleiben erhalten.

## v0.13.9 invariants

- Never hardcode match timestamps, filenames, track IDs or clip numbers.
- A source may begin after a break with the goalkeeper near midfield. Do not require goal proximity during the initial frames.
- Keep automatic selection pending while evidence is ambiguous; only use the interactive fallback after the configured deferred horizon.
- Preserve restart-context and return-to-goal evidence in the debug package.

## v0.13.10 invariants

- Merge related candidates within 2.5s or with continuous possession flow to prevent fragmented clips.
- Clips must only end after a detected restart (kick/throw) if ball control was established.
- Interaction validation must use dynamic trajectory features (interaction_score) to reduce false positives.
- Every candidate must record its clip_end_reason and merge ancestry.
- HTML reports must display interaction scores and merge reasons.
