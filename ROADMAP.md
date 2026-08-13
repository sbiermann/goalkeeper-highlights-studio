Current stabilization release: 0.13.20, focused on performance profiling transparency and measurable analysis throughput improvements without changing event/boundary thresholds.

# Roadmap

## Current focus: Action Merging and False Positive Reduction (v0.13.10)
- Automatic merging of related events within 2.5s or via possession flow.
- Improved interaction validation to reduce non-action clips.
- Dynamic clip ends triggered by detected restarts (kick/throw).

## 0.13.x
- Version 0.13.20 extends stage-level performance profiling (decode, model.track wall time, track overhead, source-level metrics) and keeps detection/boundary semantics unchanged.
- Version 0.13.19 fixes multi-source state leaks at source boundaries, keeps semantic keeper identity stable, and adds source-level diagnostics/regression tests without threshold changes.
- Version 0.13.18 fine-tunes adaptive catch/control idle tails: LOW stays 3s, MEDIUM increases from 5s to 6s, HIGH from 6s to 7s; classification logic remains unchanged.
- Version 0.13.17 replaces the fixed 3s catch/control idle tail with adaptive LOW/MEDIUM/HIGH classification based on multi-signal evidence.
- Version 0.13.16 tightens weak recovery continuation absorption and makes catch/control post-roll dynamically idle-aware (11s stays a maximum).
- Version 0.13.15 adds conservative context trimming in final overlap dedup when only outer context violates duration limits.
- Version 0.13.14 introduced a final overlap dedup pass to consolidate near-identical final highlight windows.
- Version 0.13.6 established complete diagnostics. 
- Version 0.13.10 introduced intelligent merging and action-aware clips.
- Version 0.13.9 improved goalkeeper identity and selection logic.
- Version 0.13.10 improves highlight quality and recall through smarter merging and validation.

# Roadmap

## 0.9.0

- Confirmed-contact fallback for missed distributions and clearances
- Rejected clips exported by default
- Fast FFmpeg input seeking
- NVENC auto-detection
- Parallel clip generation
- Stream-copy concatenation
- Accurate and fast clipping modes
- FFmpeg performance report

## 0.9.x

- Tune confirmed-contact thresholds with more full-match runs
- Detect whether a distribution is a pass, throw, punt or goal kick
- Improve side-switch and halftime keeper re-identification
- Add optional per-event user labels to SQLite
- Benchmark parallel job counts on HDD, SSD and NVMe storage

## 0.10.0 completed

- Multi-frame automatic goalkeeper bootstrap
- Shirt-colour uniqueness scoring in HSV space
- Camera-proximity and bounding-box-size scoring
- Goal-region occupancy and movement-pattern scoring
- Ball-contact and track-persistence evidence
- Interactive fallback for inconclusive selections
- Stable `Keeper #1` identity across ByteTrack changes
- Explainable HTML and JSON goalkeeper-detection report
- Preserve the validated 0.9.3 dynamic clip length behavior

## 0.10.x

- Tune bootstrap weights with additional full matches and different camera heights
- Detect the near-goal orientation instead of assuming lower-image proximity
- Add optional jersey-colour swatches and bootstrap thumbnails to the report
- Use later confirmed hand contacts to retroactively strengthen keeper identity

## 1.0

- Stable end-user release
- Reproducible model installation
- Windows installer
- Validated presets for common behind-goal camera positions

## 0.9.2 completed

- Dynamic clip endings and continuation chaining
- Safety tail after the final goalkeeper action
- Maximum clip duration guard

## 0.10.4 completed

- Expanded final terminal summary with source, event statistics, goalkeeper quality and performance.
- Added raw candidate and merge statistics.
- Added stabilized keeper confidence, re-identification count and realtime factor.

## 0.10.3 completed

- Clean single-line terminal progress UI without raw tqdm counters.
- Separate analysis, selection, clip-generation and concatenation phases.
- Structured final run summary.
- Detailed diagnostics remain available through `--verbose`.

## Completed in 0.12.0

- Non-recursive directory input for sequential camera files
- Natural source ordering
- Continuous multi-source timeline
- Cross-file highlight situations
- Source manifest with global offsets

## 0.13 validation

Validate action-aware clip boundaries on complete matches, especially set pieces, long goalkeeper distributions, chained attacks, source/half-time transitions, false-positive static ball tracks and the single-frame save recovery pass.


## 0.13.1

- Generic missed-action recovery pass
- Stronger false-positive validation
- Robust OpenCV packet reading and automatic reopen
- Refined action-aware clip margins

## 0.13.4 completed

- Correct routing and Qwen-pass accounting
- Add measured routing and Qwen runtimes
- Improve per-candidate retry diagnostics
- Extend tests without changing detection thresholds or clip behavior


## Version 0.13.5
Verified audit/stabilization release with last-source analysis and an explicit feature verification matrix.
