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


## v0.13.8 invariants

- Keeper bootstrap ranks logical identities, not isolated ByteTrack IDs.
- Automatic fallback reasons and ranking history must remain in the debug package.
- Every candidate must keep a stable id and merge ancestry.
- Debug archives must never contain video files.
- OpenCV read-attempt configuration must be applied before importing cv2.

## v0.13.9 invariants

- Never hardcode match timestamps, filenames, track IDs or clip numbers.
- A source may begin after a break with the goalkeeper near midfield. Do not require goal proximity during the initial frames.
- Keep automatic selection pending while evidence is ambiguous; only use the interactive fallback after the configured deferred horizon.
- Preserve restart-context and return-to-goal evidence in the debug package.
