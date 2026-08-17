Current stabilization release: 0.13.31, focused on credit-efficient false-negative rescue and core-boundary correction for clips 12/15/17/20 with unchanged performance/runtime defaults.

# Roadmap

## Current focus: Action Merging and False Positive Reduction (v0.13.10)
- Automatic merging of related events within 2.5s or via possession flow.
- Improved interaction validation to reduce non-action clips.
- Dynamic clip ends triggered by detected restarts (kick/throw).

## 0.13.x
- Version 0.13.31 is completed as a credit-efficient logic/boundary release for clip classes 12/15/17/20 with strict regression protection for already approved clips.
- 0.13.31 outcome: strong merged distribution phases can survive outside-box restart rejection when multi-signal release evidence is present; weak isolated restart situations remain rejected.
- 0.13.31 adds conservative contextual recovery rescue for compact uncovered-activity windows while preserving rejection of unsupported recovery windows.
- 0.13.31 boundary outcome: rescued distribution/recovery clips use compact core-focused windows and long multi-catch final-overlap phases are trimmed by relevance instead of pure max-duration clamping.
- 0.13.31 validation scope for this implementation: targeted unit/regression tests plus one full pytest suite run; no real-video runs performed by Junie (`MAX_REAL_VIDEO_RUNS=0`).
- Version 0.13.30 is completed as a credit-efficient boundary-fix release for the first block (clips 1–7) with unchanged event/candidate/recovery threshold semantics.
- 0.13.30 outcome: clip 3 (`raw-0005`) is shifted to later start and longer post-context; clip 7 (`raw-0020`) is shortened to a conservative isolated safety tail; clip 2 remains rejected; clips 1/4/5/6 remain unchanged.
- 0.13.30 validation was executed on the bounded real run window (`--duration 625`) and confirmed stable candidate mapping/invariants for clips 1–7.
- Decision for 0.13.30: no performance optimization work; PyTorch FP32 + prefetch + packed + current track path remain unchanged.
- Recommendation for 0.13.31: only continue with broader quality validation after locking this 0.13.30 baseline.
- Version 0.13.29 is completed as a credit-efficient boundary-fix release for clips 1–7 with a short real-run validation window (`--duration 625`).
- 0.13.29 outcome: clip boundaries for the first block were corrected via generalizable core-selection rules; clip 2 remains rejected; clips 6/7 remain regression-stable.
- `analyze` now accepts optional `--duration` for bounded real validations; default full-length analysis remains unchanged.
- Decision for 0.13.29: no performance optimization work; PyTorch FP32 + prefetch + packed + current track path remain unchanged.
- Recommendation for 0.13.30: continue quality-focused boundary validation on broader real samples only after locking the new 0.13.29 baseline.
- Version 0.13.28 is completed as a credit-efficient reproducibility and hotspot-screening release.
- Benchmark runs now force non-interactive keeper selection to avoid user-click waiting-time skew; keeper selection metadata is exposed via `keeper_selection_mode` and `keeper_selection_timed`.
- Analysis timing now excludes interactive keeper waiting-time via `interactive_wait_seconds`, so `analysis_seconds`/`realtime_factor` remain benchmark-clean.
- 60s production-path screening (`start=0`, `duration=60`, `frame_stride=2`, OpenCV+Prefetch, FP32, packed, baseline `image_size`) measured `analysis_seconds=43.879`, `processed_fps=17.115`.
- Decision for 0.13.28: no new optimization default. Remaining dominant hotspot is still inside `model.track` framework cost; input-handling-related blocks did not show a simple local >=5% low-hanging-fruit path under current constraints.
- Recommendation for 0.13.29: isolate and measure safe reductions around Ultralytics callback/predictor-pre overhead with strict functional equivalence guards.
- Version 0.13.27 is completed as a constrained low-effort sweep on the established FP32/Packed/Prefetch path.
- Screening/confirmation outcome: `tf32` gave a small speedup but remained below the 5% default threshold on 120s; `cudnn_benchmark` was slower; `imgsz=576` and `imgsz=512` were faster but not functionally equivalent (candidate-count deltas).
- Event-rich Teil22 validation (`start=540`, `duration=220`) confirms functional equivalence for `tf32` (`4/3/1` vs `4/3/1`, keeper unchanged), but default criteria were still not met.
- Decision for 0.13.27: keep current production defaults unchanged (PyTorch FP32, existing baseline `image_size`, packed conversion, OpenCV prefetch, optimized track path).
- Recommendation for 0.13.28: target detection-quality-preserving acceleration around model input handling with stricter ball-sensitivity guards before considering any imgsz default change.
- Version 0.13.26 is completed as an isolated backend research release: `yolo.backend` (`pytorch|tensorrt|onnx`) was added with explicit `requested_backend`/`effective_backend`/`backend_fallback_reason` diagnostics.
- Version 0.13.26 completed real serial FP32 runs on the 120s reference segment (`pytorch_run1`, `tensorrt_run1`, `pytorch_run2`, `tensorrt_run2`); TensorRT was only counted as TensorRT where `effective_backend=tensorrt`.
- Version 0.13.26 confirms engine-cache separation (`engine_build_seconds` vs `analysis_seconds`) and stable PyTorch fallback behavior when backend requirements are unavailable.
- Version 0.13.26 does not promote a backend default switch: PyTorch remains default, TensorRT remains optional due measured variability and observed candidate-merge divergence in the reference runs.
- Recommendation for 0.13.27: focus on strict backend-equivalence instrumentation (candidate/detection/track deltas incl. IoU/confidence/track semantics) and validate on a second event-rich excerpt before any default-change decision.
- Version 0.13.25 is completed: callback/preparation hotspot profiling was refined (`track_callback_dispatch_ms`, per-event callback timings, `track_pre_*` substages) and benchmarked in real 120s FP32 A/B runs.
- Version 0.13.25 confirms functional equivalence in the A/B runs (same processed frames/candidates/accepted/rejected/merged/keeper and no candidate timing/boundary diffs in exported artifacts).
- Version 0.13.25 observed a small but reproducible median gain for the optimized track path (`analysis_seconds` ~`+1.79%`, `model_track_wall_ms` ~`+2.18%`, `track_overhead_ms` ~`+2.87%` improvement) while preserving track semantics and event/candidate/boundary logic.
- Primary remaining framework hotspot is still callback-related predictor-pre work (`track_predictor_pre_ms`, mainly `track_callback_predict_start_ms`).
- Next isolated performance candidate for 0.13.26 is targeted reduction of unavoidable predictor-pre callback/setup cost, with strict tracking-equivalence checks.
- Version 0.13.24 is completed: track/framework-overhead profiling was refined (`track_callback_ms`, `track_predictor_pre_ms`, `track_predictor_post_ms`, `track_tracker_update_ms`, `track_result_build_ms`, `track_result_wrap_ms`, `track_ultralytics_misc_ms`) and benchmarked in real 120s FP32 A/B runs.
- Version 0.13.24 confirms functional equivalence in the A/B runs (same processed frames/candidates/accepted/rejected/merged/keeper and no candidate timing/boundary diffs in exported artifacts).
- Version 0.13.24 observed a measurable median gain for the optimized track path (`analysis_seconds` ~`+6.91%`, `model_track_wall_ms` ~`+5.70%`, `track_overhead_ms` ~`+7.94%` improvement), while preserving track semantics and event/candidate/boundary logic.
- OpenCV decoder prefetch from 0.13.23 remains runtime default; 0.13.24 does not change decoder semantics.
- FP32 remains default precision; FP16 stays rejected based on 0.13.21 real benchmarks.
- Version 0.13.22 is completed: track/framework overhead was decomposed further, a persistent predictor track path (`legacy` vs `optimized`) was implemented and A/B-tested, functional equivalence was confirmed, but no measurable performance gain was observed; therefore this path is not promoted as a performance default.
- Version 0.13.22 keeps FP32 as default runtime precision and does not use FP16 as an optimization path (FP16 remained rejected based on 0.13.21 real benchmarks).
- Next isolated performance candidate after 0.13.22 is decoder prefetch/decode-processing overlap (not implemented or validated in 0.13.22, and not a default decision).
- Version 0.13.21 adds isolated FP16/CUDA precision gating, fallback diagnostics and benchmark comparison metrics while keeping detection/candidate/boundary logic unchanged.
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
