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
