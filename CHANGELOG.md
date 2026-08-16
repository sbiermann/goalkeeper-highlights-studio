## 0.13.28
- Benchmark-Reproduzierbarkeit verbessert: Benchmark-Läufe erzwingen nun standardmäßig eine nicht-interaktive Keeper-Auswahl (`runtime.benchmark_force_noninteractive_keeper_selection=true`), damit manuelle Klick-Wartezeit Performance-Messungen nicht verfälscht.
- Analyse-Timing bereinigt: Interaktive Keeper-Wartezeit wird separat erfasst (`interactive_wait_seconds`) und aus `analysis_seconds`/`realtime_factor` herausgerechnet.
- Kompakte Benchmark-Metadaten erweitert um `keeper_selection_mode` und `keeper_selection_timed` für reproduzierbare A/B-Dokumentation.
- Isoliertes 60s-Screening auf Produktionspfad durchgeführt (`start=0`, `duration=60`, `frame_stride=2`, OpenCV+Prefetch, FP32, Packed, bestehende `image_size`): `analysis_seconds=43.879`, `processed_fps=17.115`.
- Hotspot-Befund: dominanter verbleibender Block bleibt `model_track_wall_ms`; Input-Handling-nahe Anteile (`boxes_from_result_ms`, `yolo_preprocess_ms`) liefern in diesem Stand keinen einfachen, lokal umsetzbaren >=5%-End-to-End-Hebel.
- Entscheidung 0.13.28: keine zusätzliche Performance-Optimierung als neuer Default; fachliche Logik (Thresholds/Tracking/Event/Candidate/Boundary/Recovery) unverändert.

## 0.13.27
- Low-Hanging-Fruit Performance Sweep auf dem bestehenden Produktionspfad (PyTorch FP32 + Packed + OpenCV-Prefetch + optimierter Track-Pfad), ohne Änderungen an Event-/Candidate-/Boundary-/Recovery-/Threshold-Logik.
- Neue schlanke Benchmark-Schalter für gezielte A/B-Läufe: `--tf32/--no-tf32`, `--cudnn-benchmark/--no-cudnn-benchmark`, `--imgsz`.
- 60s-Screening (`start=0`, `duration=60`, `frame_stride=2`): `tf32` `+7.40%`, `cudnn_benchmark` `-7.92%` (verworfen), `imgsz=576` `+34.91%`, `imgsz=512` `+38.48%`.
- 120s-Bestätigung (`start=0`, `duration=120`): Baseline `68.55s`/`21.90 FPS`; `tf32` `66.28s`/`22.64 FPS` (`+3.31%`, unter Default-Schwelle); `imgsz=576` und `imgsz=512` liefern Candidate-Abweichungen und wurden trotz Speedup verworfen.
- Teil22-Fachvalidierung (`FCWittlinen-SFETeil22-Tonasync.mp4`, `start=540`, `duration=220`) für `tf32`: Baseline `4/3/1`, `tf32` `4/3/1` (candidates/accepted/rejected), Keeper `Keeper #1` identisch.
- Entscheidung 0.13.27: Kein neuer Default. PyTorch-FP32-Produktionspfad und bestehende `image_size` bleiben unverändert; TF32 bleibt optionaler Benchmark-/Runtime-Schalter.

## 0.13.26
- Isolierter Backend-Research-Release für YOLO-Inference (`yolo.backend`: `pytorch|tensorrt|onnx`) bei unveränderter Event-/Candidate-/Boundary-/Recovery-/Threshold-Logik.
- Neue Backend-Diagnostik in Benchmark/Reports: `requested_backend`, `effective_backend`, `backend_fallback_reason`, `model_format`, `engine_cached`, `engine_build_seconds`, `backend_load_seconds`, `backend_warmup_seconds`.
- Optionaler TensorRT-Pfad über offiziellen Ultralytics-Export (`format="engine"`) mit Engine-Cache und sauberem PyTorch-Fallback bei Nichtverfügbarkeit.
- Optionale Installation auf Referenzsystem (ohne Kern-Dependency-Wechsel): `tensorrt-cu12==10.13.3.9`, `tensorrt_cu12_bindings==10.13.3.9`, `tensorrt_cu12_libs==10.13.3.9`, transitive Runtime `nvidia-cuda-runtime-cu12==12.9.79`.
- Reale A/B-Serie (120s, `frame_stride=2`, OpenCV+Prefetch, FP32): PyTorch `run1=75.688s`/`run2=69.549s`; TensorRT (effektiv) `run1=215.567s` (mit Engine-Build), `run2=69.135s` (Cache-Reuse).
- Gemessener Fachstatus im Referenzausschnitt: identische Kernwerte `processed_frames=1501`, `candidates=2`, `accepted=1`, `rejected=1`, Keeper `Keeper #1`; jedoch Abweichung bei `merged` (`PyTorch=0`, `TensorRT=1`) im gemessenen Stand.
- Entscheidung für 0.13.26: Kein Backend-Default-Wechsel; PyTorch bleibt Standard, TensorRT bleibt optionaler Research-Pfad mit explizitem Fallback.

## 0.13.25
- Isolierter Performance-Release mit Fokus auf den in 0.13.24 identifizierten Ultralytics-Hotspot `track_callback_ms` (insbesondere `track_predictor_pre_ms`) bei unveränderter Fachlogik.
- Profiling erweitert um feingranulare Callback-/Preparation-Unterstufen: `track_callback_dispatch_ms`, `track_callback_predict_start_ms`, `track_callback_batch_start_ms`, `track_callback_postprocess_end_ms`, `track_callback_batch_end_ms`, `track_callback_predict_end_ms`, `track_callback_other_ms`, `track_pre_source_setup_ms`, `track_pre_batch_prepare_ms`, `track_pre_other_ms`.
- Reales 4x-A/B auf `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4` (`start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `decoder_execution_mode=prefetch`, `precision=FP32`, `boxes_from_result_mode=packed`, `track_path=legacy|optimized`).
- Medianvergleich: Legacy `analysis_seconds=92.075`, `FPS=16.3025`, `model_track_wall_ms=52.53`, `track_overhead_ms=22.338`; Optimized `analysis_seconds=90.4295`, `FPS=16.599`, `model_track_wall_ms=51.384`, `track_overhead_ms=21.6965`.
- Ergebnis: kleiner, reproduzierbarer Gewinn für Optimized (`analysis_seconds +1.79%`, `model_track_wall_ms +2.18%`, `track_overhead_ms +2.87%`) bei fachlicher Äquivalenz (`processed_frames=1501`, `candidates=2`, `accepted=1`, `rejected=1`, `merged=0`, `keeper=Keeper #1`).
- Root-Cause-Befund: `track_predictor_pre_ms` dominiert den Callback-Block weiterhin, vor allem über `track_callback_predict_start_ms`; keine Änderungen an Event-/Candidate-/Recovery-/Boundary-Logik oder fachlichen Thresholds.
- FP32 bleibt Standard, FP16 bleibt verworfen, Decoder-Prefetch bleibt Default, Packed-Result-Conversion bleibt aktiv.

## 0.13.24
- Isolierter Performance-Release zur tieferen Quantifizierung des Ultralytics/`model.track()`-Framework-Overheads bei unveränderter Fachlogik.
- Track-Overhead-Profiling um feinere Substages erweitert: `track_callback_ms`, `track_predictor_pre_ms`, `track_predictor_post_ms`, `track_tracker_update_ms`, `track_result_build_ms`, `track_result_wrap_ms`, `track_ultralytics_misc_ms` plus Restfeld `track_framework_other_ms`.
- Reales 4x-A/B auf `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4` (`start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `decoder_execution_mode=prefetch`, `precision=FP32`, `boxes_from_result_mode=packed`, `track_path=legacy|optimized`).
- Medianvergleich: Legacy `analysis_seconds=96.6955`, `FPS=15.5535`, `model_track_wall_ms=54.0835`; Optimized `analysis_seconds=90.011`, `FPS=16.703`, `model_track_wall_ms=51.0025`.
- Ergebnis: messbare Verbesserungen zugunsten Optimized (`analysis_seconds` `+6.91%`, `model_track_wall_ms` `+5.70%`, `track_overhead_ms` `+7.94%`) bei fachlicher Äquivalenz der Benchmark-Kernresultate.
- Primärer Framework-Teilblock bleibt `track_callback_ms` (insbesondere `track_predictor_pre_ms`); `track_tracker_update_ms` und `track_result_*` waren in den gemessenen Läufen nicht separat >0 messbar.
- Decoder-Prefetch aus 0.13.23 bleibt Default, FP32 bleibt Standard, FP16 bleibt verworfen; keine Änderungen an Event-/Candidate-/Recovery-/Boundary-Logik oder fachlichen Thresholds.

## 0.13.23
- Isolierter Performance-Release für asynchronen OpenCV-Decoder-Prefetch (FP32-only, keine Änderung an Event-/Candidate-/Boundary-/Recovery-Logik).
- Neuer Decoder-Modus `runtime.decoder_execution_mode` mit `legacy`/`prefetch` und bounded Queue (`runtime.decoder_prefetch_queue_size`, Default `4`).
- Klare Prefetch-Semantik für Source-Ende, Global-Ende und Decoder-Exceptions; kontrollierte Fehlerpropagation und sauberer Thread-/Decoder-Shutdown.
- Profiling um Decoder-Prefetch-Metriken erweitert: `decoder_read_ms`, `decoder_queue_wait_ms`, `consumer_queue_wait_ms`, `decoder_prefetch_frames`, `decoder_queue_max_depth`.
- Reales 4x-A/B auf `C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4` (`start=0`, `duration=120`, `frame_stride=2`, `decoder=opencv`, `precision=FP32`, `packed`): Median Legacy `analysis_seconds=106.6535`, `FPS=14.0745`; Median Prefetch `analysis_seconds=95.1685`, `FPS=15.772`.
- Ergebnis: fachlich äquivalent im Benchmarkvergleich (identische Frame-/Candidate-/Keeper-Kernwerte), mit messbarem End-to-End-Gewinn; OpenCV-Prefetch wird als Runtime-Default aktiviert.
- FP32 bleibt Standardpräzision; FP16 bleibt gemäß 0.13.21 verworfen; persistenter Predictor-Pfad aus 0.13.22 bleibt nicht Performance-Default.

## 0.13.22
- Fokus-Release zur Untersuchung und Optimierung des `model.track()`-Framework-Overheads (FP32-only, kein FP16-Einsatz in diesem Release).
- Neuer umschaltbarer Trackpfad für reproduzierbare A/B-Messungen: `runtime.track_execution_mode` bzw. CLI `benchmark --track-path {legacy,optimized}`.
- Track-Overhead-Profiling um Unterstufen erweitert: `track_predictor_setup_ms`, `track_tracker_update_ms`, `track_result_build_ms`, `track_callbacks_ms`, `track_framework_other_ms`.
- Persistenter Predictor-/Tracker-Aufrufpfad für wiederholte Frame-Inferenz eingeführt und gegen den Legacy-Pfad A/B-validiert, bei Beibehaltung der Multi-Source-Reset-Semantik.
- Ergebnis der 0.13.22-A/B-Messung: kein belastbarer Performancegewinn gegenüber Legacy, daher keine Umstellung auf einen neuen Performance-Default.
- Benchmark-/Regressionstests für Trackpfad-Auswahl und Runtime-Konfiguration ergänzt; vollständige Testsuite bleibt grün.
- Keine Änderungen an Event-/Candidate-/Boundary-Logik oder fachlichen Thresholds.

## 0.13.21
- Gezielter FP32-vs-FP16 A/B-Release für YOLO-Inference auf CUDA ohne zusätzliche Pipeline-Optimierungen.
- FP16 nur noch als kontrollierter Runtime-Zustand mit CUDA-Guard (`requested_fp16`, `effective_fp16`, `fp16_fallback_reason`) ausgewiesen.
- `model.track(..., half=...)` wird ausschließlich bei effektiv aktivem FP16 gesetzt; CPU-/CUDA-Fallback bleibt crashfrei.
- Benchmark-/Profiling-Metadaten um Precision-Diagnostik und erweiterte Vergleichskennzahlen ergänzt.
- Default-Konfiguration (`config/default.yaml` und `src/goalkeeper_highlights/default.yaml`) um explizites `yolo.half: false` synchronisiert.
- Keine Änderungen an Event-/Candidate-/Boundary-Logik oder zugehörigen Thresholds.

## 0.13.20
- Performance-Release mit detailliertem Stage-Profiling im Analyse-Loop erweitert.
- Separate Decode-Wall-Time (`decoder_next_ms`) ergänzt, um Decoder-Kosten klar von YOLO/Tracking zu trennen.
- `model.track` Wall-Time und abgeleiteter Track/Framework-Overhead (`track_overhead_ms`) als eigene Messgrößen ergänzt.
- Box-Konvertierung aus Ultralytics-Results auf kompakteren GPU→CPU-Transfer umgestellt, um Synchronisationspunkte zu reduzieren.
- Per-Source-Performance-Statistiken inkl. Loop-Percentiles und Decode-/Track-/Inference-Mittelwerten ergänzt.
- Neue Profiling-Tests für Aggregation, Decode-Trennung, Track-Overhead, Source-Stats und Non-Negative-Timings hinzugefügt.
- Keine fachliche Änderung an Event-/Candidate-/Boundary-Logik.

## 0.13.19
- Behebt einen reproduzierbaren Multi-Source-Lifecycle-Fehler, bei dem spätere Quellen trotz gültiger Szenen keine Raw-Candidates erzeugen konnten.
- Source-Transition explizit stabilisiert: transiente Tracking-/Interaktionszustände werden an Quellenwechseln sauber zurückgesetzt, die semantische Keeper-Identität bleibt erhalten.
- Zeitachsen-Semantik präzisiert: globale Timeline bleibt führend, source-lokale Zeit wird explizit mitgeführt und für Source-Diagnostik verfügbar gemacht.
- Source-Diagnostik um per-Source-Zähler und Reset-Flags erweitert, inkl. Candidate-/Keeper-/Ball-bezogener Kennzahlen.
- Neue Regressionstests für Multi-Source-State-Isolation ergänzt.
- Keine fachlichen Änderungen an Event-/Boundary-/Threshold-Logik aus 0.13.18.
- Debug-Paketversion auf `goalkeeper_highlights_debug_v0.13.19.zip` angehoben.

## 0.13.18
- Feintuning des adaptiven `catch_or_control`-Idle-Tails ohne Änderung der Klassifizierungslogik.
- `catch_control_idle_tail_medium_seconds` von 5.0 auf 6.0 erhöht.
- `catch_control_idle_tail_high_seconds` von 6.0 auf 7.0 erhöht.
- `catch_control_idle_tail_low_seconds` bleibt unverändert bei 3.0.
- Keine Änderungen an Boundary-Prioritäten, Recovery-Continuation-Validation oder anderen bestehenden Mechanismen.
- Bestehende Regressionen der Boundary-Mechanismen bleiben geschützt.

## 0.13.17
- Adaptiver `catch_or_control`-Idle-Tail ersetzt den festen 3s-Tail durch LOW/MEDIUM/HIGH-Klassifizierung auf Basis mehrerer Signale.
- Einstufung kombiniert u.a. `contact_frames`, `possession_duration` und ergänzend `interaction_score`; Keeper-Motion allein bleibt schwache Evidenz (LOW).
- Boundary-Prioritäten bleiben unverändert: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → `dynamic_idle_tail` → `timeout`.
- Dynamisches Clipende bleibt konservativ begrenzt über `catch_control_max_post_roll_seconds`, globale Dauergrenzen und ursprüngliches geplantes Clipende.
- Neue numerische Diagnostikfelder für Idle-Level, Schwellentreffer und gewählten Tail (`catch_control_idle_level`, `catch_control_*_match`, `catch_control_selected_idle_tail` etc.).
- Default-Konfiguration um adaptive Catch/Control-Schwellen und Tail-Stufen erweitert (synchron in beiden Default-YAMLs).
- Synthetische Regressionstests für LOW/MEDIUM/HIGH, Clip-9-ähnlichen MEDIUM-Fall, Motion-ohne-Ball sowie Prioritäts-/Clamp-Schutz ergänzt.

## 0.13.16
- Strengere Recovery-Continuation-Validierung: schwache Recovery-Kandidaten werden nur noch bei belastbarer Keeper-/Ball-Evidenz absorbiert.
- Keeper-Motion allein verlängert Clips nicht mehr; `interaction_score` und Ball-Dynamik fließen in die Continuation-Entscheidung ein.
- Dynamischer Post-Roll für `catch_or_control`: 11 Sekunden bleiben Maximalwert, bei ausbleibender Folgeaktivität wird früher beendet.
- Priorisierung bleibt erhalten: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → dynamic idle tail → `timeout`.
- Numerische Diagnostikfelder für Recovery-Continuation und Catch/Control-Idle-Tail ergänzt.
- Neue synthetische Regressionstests für schwache/starke Recovery-Continuation und dynamische Catch/Control-Tails ergänzt.

## 0.13.15
- Konservatives Context-Trimming im finalen Overlap-Dedup ergänzt, wenn die Union nur wegen äußerem Kontext über dem Dauerlimit liegt.
- Action-Fenster bleiben beim finalen Merge vollständig erhalten; nur überschüssiger Pre-/Post-Roll wird reduziert.
- Keine Erhöhung der globalen Clipdauergrenzen; bestehende `max_dynamic_clip_seconds` und Toleranz bleiben maßgeblich.
- Merge-/Recovery-/Boundary-Metadaten bleiben beim finalen Overlap-Merge erhalten.
- Numerische Diagnostik für Original-/effektiven Kontext und getrimmte Dauer (`final_overlap_*`) erweitert.
- Regressionstests für Trimming- und Sicherheitsfälle im finalen Overlap-Dedup ergänzt.

## 0.13.14
- Finaler Overlap-Dedup-Pass nach allen Clip-Boundary-Erweiterungen eingeführt.
- Stark überlappende finale Highlights desselben Keepers werden zu einem Clipfenster konsolidiert.
- Merge-Schutzregeln für Keeper-Identität, Restart-Kontext und maximale Clipdauer ergänzt.
- Merge-/Recovery-/Boundary-Metadaten bleiben beim finalen Zusammenführen erhalten und nachvollziehbar.
- Absorbierte Kandidaten werden nicht doppelt exportiert, bleiben aber diagnostisch sichtbar.
- Neue numerische Diagnostikfelder für finalen Overlap-Check (`final_overlap_*`) ergänzt.
- Synthetische Regressionstests für finales Overlap-Dedup und Sicherheitsfälle ergänzt.

## 0.13.13
- Implementierung eines kontrollierten `recovery_window_tail`-Fallbacks für akzeptierte Recovery-Aktionen mit zu kurzem Timeout-Ende.
- Prioritätskette für Clip-Enden explizit: `controlled_release` → `recovery_distribution_continuation` → `recovery_window_tail` → `timeout`.
- Nutzung vorhandener Recovery-Boundary-Evidenz (`recovery_window_start`/`recovery_window_end`) statt weiterer Candidate-Absorptionslogik als Hauptlösung.
- Begrenzte Verlängerung über `recovery_window_tail_max_extension_seconds` (konservativ, mit Clamping/Blockierung).
- `action_end` bleibt diagnostisch getrennt vom erweiterten `clip_end`; neues `clip_end_reason` ist `recovery_window_tail`.
- Erweiterte numerische Diagnostik (`recovery_tail_*`) und zusätzliche Nachvollziehbarkeit im Candidate-Lifecycle/Debug-Paket.
- Neue synthetische Regressionstests für Recovery-Tail-Fallback; bestehende Regressionen aus 0.13.10/0.13.11/0.13.12 bleiben unverändert erfolgreich.
- Debug-Paketversion auf `goalkeeper_highlights_debug_v0.13.13.zip` angehoben.

## 0.13.12
- Implementierung der Candidate-basierten Recovery-Distribution-Fortsetzung zur Behebung zu kurzer Clips (z.B. Clip 5).
- Ermöglicht die Clip-Verlängerung auch bei unzuverlässig erkannter `possession_duration`, sofern ein nachfolgendes Distribution-Event vorliegt.
- Unterstützung für die Absorption von rejected Distribution-Kandidaten als reine Boundary-Evidenz.
- Neue Konfigurationsparameter (`recovery_distribution_continuation_enabled`, `recovery_distribution_search_seconds` etc.).
- Erweiterte Diagnostik im `score_breakdown` für die Recovery-Distribution-Entscheidung.
- Debug-Paket in `goalkeeper_highlights_debug_v0.13.12.zip` umbenannt.

## 0.13.11
- Verbesserte Erkennung kontrollierter Ballfreigaben (Abschlag aus der Hand, Abwurf, Pass) nach Ballbesitz.
- Clips werden nun bis zum tatsächlichen Release-Zeitpunkt plus Safety-Tail verlängert, um abgeschnittene Highlights zu vermeiden.
- Neue Konfigurationsparameter für die Release-Erkennung (`controlled_release_enabled`, `controlled_release_safety_tail_seconds` etc.).
- Erweiterte Diagnostik im `score_breakdown` für die Release-Entscheidung.
- Sicherstellung der Phase-Merge-Logik und Context-Trimming aus v0.13.10.
- Debug-Paket in `goalkeeper_highlights_debug_v0.13.11.zip` umbenannt.

## 0.13.10
- Improved phase merge logic (v0.13.10): implemented intelligent context trimming (pre-roll/post-roll) to fit combined clips within the duration limit (max 45s + 8% tolerance).
- Added `phase_merge_min_pre_roll_seconds` and `phase_merge_min_post_roll_seconds` configuration parameters (default 2.0s).
- Action windows are always preserved during trimming; only context margins are reduced.
- Extended phase merge diagnostics with numeric values for original/effective rolls, action duration, and trimmed duration.
- Extended merge logic (v0.13.10): automatic merging of events within 2.5s gap or with ball possession continuity.
- Dynamic Clip End: clips now extend until a kick or throw is detected after ball control, reducing abrupt ends.
- Interaction Validator V2: significantly reduced false positives by requiring genuine interaction dynamics (direction change, approach speed, motion) instead of mere proximity.
- Expanded Debug Package: added `interaction_score`, `clip_end_reason`, and detailed merge history (`merged_from`, `merged_reason`) to every candidate.
- Updated HTML Report: now displays merge information, interaction scores, and the reason for each clip end.
- Added regression tests for 0.13.10 merge and interaction logic.
- Debug archive renamed to `goalkeeper_highlights_debug_v0.13.10.zip`.

## 0.13.9

- Reworked goalkeeper bootstrap into deferred, whole-observation evidence collection.
- Recordings that start after a break with an advanced goalkeeper no longer receive the full field-excursion penalty.
- Added restart-context detection, return-to-goal evidence and longer logical-identity gaps.
- Automatic selection can remain pending for up to 240 seconds before interactive fallback.
- Keeper diagnostics now report `start_context`, `initial_keeper_position`, `automatic_selection_deferred` and `selection_confirmed_at`.
- Debug archive/version metadata updated to 0.13.10.
- Restored the `source_selection` API used by `--only-last-source`.

## 0.13.8

- Reworked automatic goalkeeper selection around multi-stage behavioural evidence instead of a single-track startup score.
- Aggregates fragmented ByteTrack IDs into logical keeper candidates using shirt appearance and temporal continuity.
- Added central goal-axis, isolation, depth-stability and horizontal-patrol signals plus a field-excursion penalty.
- Extended the bootstrap observation window to 60 seconds and records the complete ranking history and explicit fallback reason.
- Added stable candidate ancestry and `candidate_pipeline_trace.json` for raw, recovery, validation, merge, routing and final decisions.
- Added `extended_recovery_analysis.json` to show whether each suspicious uncovered window became a recovery candidate.
- Forces `OPENCV_FFMPEG_READ_ATTEMPTS=65536` before OpenCV import; `GOALKEEPER_OPENCV_READ_ATTEMPTS` can override it.
- Debug archive is now `goalkeeper_highlights_debug_v0.13.8.zip` and still contains no video files.

## 0.13.7

- Reworked goalkeeper bootstrap with an adaptive 15-45 second observation window, repeated ranking, top-candidate output and minimum winner margin.
- Stabilized logical goalkeeper identity across ByteTrack ID changes.
- Added recovery candidates from strong uncovered diagnostic activity windows.
- Added full candidate lifecycle, keeper identity timeline, re-identification events and ball detection gaps to the video-free debug package.
- Heuristic HIGH/MEDIUM/LOW routing is now recorded even when Qwen is disabled.
- Debug archive renamed to `goalkeeper_highlights_debug_v0.13.7.zip`.

# Changelog

## 0.13.6 - Missed-save diagnostics

- Focuses the next optimization cycle on goalkeeper saves that were not detected.
- Creates `goalkeeper_highlights_debug_v0.13.6.zip` automatically after every completed analysis.
- The debug archive contains all candidate decisions, effective configuration, performance data, raw SQLite analysis data and uncovered suspicious timeline windows, but no video files.
- Adds a complete per-candidate decision path including routing, Qwen passes, recovery status, scores and rejection reasons.
- Adds a diagnostic scan for ball/keeper activity that was not covered by any final candidate.
- Sets `OPENCV_FFMPEG_READ_ATTEMPTS=65536` before OpenCV initialization.

# 0.13.5

Verified audit and stabilization release.

- Audited the implemented 0.13.0-0.13.3 feature set against code and regression tests.
- Confirmed action-aware boundaries, source clamping, interaction validation, generic recovery, OpenCV decoder recovery, HIGH/MEDIUM/LOW routing, and the controlled second Qwen pass.
- Added `--only-last-source` for targeted analysis of the naturally sorted final file in a directory.
- Added a separate default output directory for last-source analyses to protect full-match results.
- Added CLI and source-selection regression tests.
- Added `FEATURE_AUDIT_0.13.5.md` with the verification matrix and remaining runtime-validation risks.

# 0.13.4

Maintenance and quality release based on the reviewed 0.13.3 routing implementation.

- Corrected routing statistics so first-pass Qwen calls count only actual model invocations.
- Added separate counters for directly accepted HIGH candidates and early-rejected LOW candidates.
- Added measured runtimes for heuristic scoring, first-pass Qwen, and second-pass Qwen processing.
- Added explicit counters for second-pass calls and highlights rescued by the second pass.
- Added per-candidate diagnostics for first-pass calls, second-pass calls, rescue status, confidence, and runtime.
- Refactored the retry decision into a separate, testable path with loop prevention.
- Extended automated test coverage for routing, uncertain responses, malformed Qwen output, recovery candidates, and performance counters.
- Updated version metadata and documentation to distinguish this maintenance release from the original 0.13.3 implementation.
- Kept the detection pipeline, clip-boundary logic, recovery algorithms, and HIGH/MEDIUM/LOW score thresholds unchanged.
- Kept directory analysis unchanged; selecting only the final source file is not part of this release.

# 0.13.3

- Corrected routing statistics so first-pass calls count only actual Qwen invocations.
- Added separate counters for direct HIGH acceptance, early LOW rejection, second-pass calls, and second-pass rescues.
- Added measured heuristic, first-pass Qwen, and second-pass Qwen runtimes.
- Added explicit retry decision handling for uncertain, recovery, short-action, and malformed-response cases with loop prevention.
- Kept directory analysis unchanged; source-file selection is not part of the 0.13.3 routing release.
- Added heuristic candidate pre-filtering with HIGH, MEDIUM, and LOW routing categories to optimize analysis speed.
- HIGH-scoring candidates with strong dynamic features are directly accepted without Qwen analysis.
- LOW-scoring candidates with weak or static features are early-rejected, saving Qwen processing time.
- Implemented a second Qwen pass for MEDIUM or uncertain candidates, using expanded temporal context and more representative frames.
- Added detailed performance statistics for routing decisions, saved Qwen calls, and retry effectiveness.
- All routing thresholds and retry parameters are fully configurable in `default.yaml`.
- Preserved all existing clip boundary and event engine logic from 0.13.2.

# 0.13.2

- Added a constrained rescue rule for classic goalkeeper actions that narrowly miss the category threshold despite reliable dynamic keeper/ball contact.
- Reworked the missed-action recovery pass to require visible ball, approach, or goalkeeper motion, reducing false positives from static overlaps.
- Added a high-confidence single-frame recovery path for very fast saves that appear in only one analyzed frame.
- Recovery masking now ignores rejected candidates so a false rejected event cannot hide a genuine nearby save.
- Increased context before catch-to-distribution scenes so clips no longer begin only at the end of the catch.
- Extended distribution and goal-kick tails to retain the actual execution and immediate continuation.
- Prevented unrelated distributions and clearances from being chained into one clip solely because they occur close together.
- Added regression tests for classic-action rescue, static-overlap rejection, and isolated goal-kick boundaries.

# 0.13.1

- Added a generic, full-timeline recovery pass for missed goalkeeper actions; no filenames or fixed timestamps are hard-coded.
- Extended catch/save clips so the approach and the completed action remain visible.
- Shortened the lead-in for isolated distributions while retaining a longer completion tail.
- Strengthened false-positive rejection for static long contacts and irrelevant restarts in the central field.
- Increased OpenCV/FFmpeg packet-read attempts to 65536 and added automatic decoder reopen/retry.
- Suppressed OpenCV warning noise in the normal progress display and summarize decoder recoveries at the end.

# 0.13.0

- Replaces fixed trigger-centred clip windows with action-aware boundaries from the temporal event engine.
- Stores the observed action start and action end for every candidate and includes the boundary reason in JSON/CSV reports.
- Adds category-specific context margins: saves and catches keep more useful lead-in and tail, while distributions start later and end sooner.
- Chains a later goalkeeper interaction only when it belongs to the same phase of play, preserving sequences such as distribution, turnover, shot and catch.
- Treats source-file changes as hard clip boundaries by default, preventing footage from the next half leaking into the preceding clip. Set `clips.allow_cross_source_clips: true` only for genuinely seamless camera segments.
- Adds interaction validation that rejects implausibly long static ball/keeper overlaps, targeting false positives caused by a bad ball track.
- Adds a conservative single-frame save recovery path for fast shots that are visible for only one sampled frame.
- Keeps rejected review clips enabled by default and preserves the virtual multi-file timeline introduced in 0.12.
- Updates README, AGENTS, roadmap and regression tests.

# 0.13.0

- Fixed directory ordering when one camera filename has a slightly different or misspelled text prefix.
- Recording sequence is now determined primarily by the final numeric filename block.
- Added a regression test for the exact `Teil1`, `Teil21`, `Teil22-Tonasync` case with inconsistent prefixes.
- Kept the virtual timeline and visible source-order check from 0.12.x.

# 0.12.1

- Fixed directory source ordering defensively before probing and decoding.
- Print the exact source order before analysis in normal and verbose modes.
- Added regression tests for the real FCWittlingen filenames and mixed-case extensions.
- No temporary concatenated source video is created.

# 0.12.0

- Replaced the temporary concatenated source MP4 with a virtual multi-file timeline.
- Fixed deterministic natural sorting for numbered camera files and mixed-case extensions.
- Added global-to-local timestamp mapping for detection, Qwen frame extraction and clip generation.
- Added frame-accurate clips that can cross file boundaries.
- Added source/timeline progress phases in quiet console mode.
- Kept source order and offsets in `source_manifest.json`.

# Changelog

## 0.10.4

- Expanded the final terminal summary into clearly separated Video, Result, Goalkeeper, Performance, FFmpeg, Created and Output sections.
- Added raw candidate, accepted, rejected and merged-candidate counts.
- Added stabilized goalkeeper confidence and re-identification count.
- Added effective realtime factor based on source duration and analysis time.
- Added the source video name to the summary.
- Kept the quiet single-line progress UI and `--verbose` developer output unchanged.

## 0.10.3

- Replaced the raw tqdm display with a clean single-line terminal progress UI.
- Removed cryptic counters such as `29/1000`, per-mille units and percentage-per-second output.
- The default display now shows phase, percentage, processed video minutes, ETA, candidate count and realtime factor.
- Added dedicated phases for analysis, candidate selection, clip creation and final video concatenation.
- Added a structured final summary with timings, encoder, clip mode, parallel jobs and created artifacts.
- Detailed detector, profiler, ByteTrack and FFmpeg output remains available only with `--verbose`.

## 0.10.2

- Category-specific clip windows: catches start earlier, distributions end sooner.
- Quiet console mode is now the default with a single progress bar and final summary.
- Added `--verbose` for detailed detector, profiler and FFmpeg output.
- FFmpeg output is captured in quiet mode so warnings no longer disrupt progress display.

## 0.9.3 - 2026-07-28

- Increased the accepted-highlight safety tail from four to eight seconds.
- Increased the final tail after chained goalkeeper actions from four to eight seconds.
- Raised the dynamic clip safety cap from 35 to 40 seconds.
- Keeps the goalkeeper's pass after a catch and several seconds of the resulting play in the highlight.
- Rejected diagnostic clips remain unchanged.

## 0.9.2

- Added dynamic clip endings for accepted highlights.
- Adds a four-second safety tail so a clip does not end exactly on a catch or follow-up action.
- Chains accepted goalkeeper events from the same phase of play when they occur within 15 seconds.
- Keeps at least four seconds after a chained final goalkeeper action.
- Caps dynamically extended clips at 35 seconds.
- Added tests for extension, chaining and maximum-duration protection.

## 0.9.1 - 2026-07-28

- Fixed category-specific acceptance being overwritten by the global event threshold after candidate merging.
- Distribution and keeper-clearance events now remain accepted when their event score meets their own category threshold.
- Candidate merging now preserves acceptance status, threshold, rejection reason and score breakdown of the stronger event.

## 0.9.0 - 2026-07-27

### Added
- Confirmed keeper-contact safety rule for calm distributions and clearances.
- New `keeper_clearance` event category.
- Automatic rejected clip and JSON export by default.
- `--no-export-rejected` opt-out.
- `--clip-mode accurate|fast`, `--encoder`, and `--parallel-jobs`.
- Automatic NVENC detection with `libx264` fallback.
- Parallel FFmpeg clip generation.
- FFmpeg performance data in the HTML report and `analysis/performance.json`.
- Playback of rejected clips directly in the HTML report.

### Changed
- Moved `-ss` before `-i` for fast input seeking.
- Final clip concatenation uses stream copy with a re-encode fallback.
- Reduced thresholds for `distribution` and `keeper_clearance`.
- Rejected filenames include their event score.

### Fixed
- Genuine goalkeeper actions with multiple confirmed contact frames are no longer rejected solely because approach, departure, or movement scores are weak.
- The two false-negative interactions observed at approximately 24:38 and 37:38 are covered by the new strong-contact logic.

## 0.8.0 - 2026-07-27

- Category-specific event thresholds.
- Possession-duration bonuses.
- Explainable score breakdowns.
- Smarter cooldown handling.
- Analysis reports and stable keeper labels.

## 0.10.0

### Automatic Goalkeeper Detection

- Added an automatic multi-frame goalkeeper bootstrap stage for behind-goal videos.
- The first configurable seconds are scored using shirt-colour uniqueness, apparent camera proximity, goal-area occupancy, movement pattern, ball contacts, and track persistence.
- Interactive selection is now a fallback only when the automatic result is inconclusive.
- Separated the stable domain identity `Keeper #1` from temporary ByteTrack IDs.
- Re-identification messages and reports consistently use `Keeper #1` while retaining ByteTrack IDs for diagnostics.
- Added `analysis/goalkeeper_detection.html` and `analysis/goalkeeper_detection.json` explaining why a track was selected.
- Added goalkeeper-detection details to the main HTML report.
- Kept the 0.9.3 dynamic clip ending unchanged; the validated 23-second Clip 9 behavior remains intact.

## 0.10.1

- Re-identification messages are quiet by default; the console now prints one stabilization message and a compact final summary.
- Keeper confidence is stabilized after 30 seconds using the median of recent tracking matches and written to the goalkeeper detection report.
- The goalkeeper detection report now shows weighted score contributions for shirt uniqueness, camera proximity, goal-area presence, movement pattern and ball contact.
- Single-action clips use a shorter four-second idle tail. Chained goalkeeper phases retain the longer eight-second final tail so complete sequences remain intact.

## 0.12.0

- `analyze` accepts either one video file or a directory containing sequential video files.
- Directory scans are deliberately non-recursive and support MP4, MOV, MKV, M4V and AVI.
- Video files use natural filename ordering (`MVI_2` before `MVI_10`).
- Sequential camera segments are joined losslessly into one internal timeline before detection.
- Keeper identity, event timestamps and cross-file situations remain continuous across file boundaries.
- Added `source_manifest.json` with source filenames, durations and global offsets.
- Directory results use `<directory>_goalkeeper_highlights` as the output folder.
- The final terminal summary shows the number of source files.
