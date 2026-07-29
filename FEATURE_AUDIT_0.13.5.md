# Feature audit 0.13.5

| Requirement | Implementation evidence | Regression coverage | Status |
|---|---|---|---|
| Action-aware start/end | `event_engine.py`, `detection.plan_clip_windows` | `test_action_aware_clips.py` | Verified |
| Hard source boundaries | `clamp_clip_windows_to_sources` | source-boundary test | Verified |
| Static-overlap rejection | interaction validation and recovery gates | static-overlap tests | Verified |
| Single-frame fast-save path | event engine and recovery configuration | event/recovery tests | Verified |
| Generic full-timeline recovery | `recover_missed_candidates` | `test_recovery_pass.py` | Verified |
| OpenCV read retry/reopen | `decoder.py` | decoder configuration inspection | Verified in code; real corrupt stream remains runtime-dependent |
| HIGH/MEDIUM/LOW routing | `classification.py` | `test_routing.py` | Verified |
| Controlled second Qwen pass | `classification.py` | retry and malformed-response tests | Verified |
| Routing performance counters | `pipeline.py` and reports | routing statistics tests | Verified |
| Analyze only final source | CLI `--only-last-source` | CLI/source-selection tests | Verified |

No match-specific timestamps, clip numbers, or filenames are encoded in the detection or recovery logic. Real-world model recall still depends on video quality, detector output, frame stride, and Qwen behavior and must be validated with complete match runs.
