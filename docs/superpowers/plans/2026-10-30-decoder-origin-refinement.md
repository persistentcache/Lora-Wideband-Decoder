# Decoder Origin Refinement Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this focused task. Keep repository changes uncommitted for the coordinating agent's review.

**Goal:** Restore the useful preamble grid lost when FFT padding moves the capture-origin coarse window outside the top three candidates.

**Architecture:** Keep the padded FFT and complete original coarse, unrefined-origin, and global search sequence. Save its top six `(run, phase)` pairs before optional work. If the coarse searches yielded no runs, origin was not already searched, and the existing deadline permits it, refine origin once and append up to six origin pairs for candidate scoring. Evaluate baseline pairs first; optional scoring also checks the existing deadline. Preserve baseline diagnostics and candidate tie order.

**Tech Stack:** Python, NumPy/SciPy, pytest.

**Spec:** `/home/david/Desktop/LoRa-Regression-Repair-20261030/decoder_diagnosis.txt`, section A; coordinating agent approved the narrow behavior on 2026-10-30.

## Global Constraints

- Preserve FFT padding, exact BW sample clock, adjacent-frame masks, long-frame drift tracking, and successful coarse-search paths.
- Keep the existing CPU deadline; no additional origin search after expiry and no deadline reset.
- Do not implement the separate truncated-frame rescue change or make commits.
- Coordinate CPU-heavy tests and retained-IQ replays with the root agent.

## Review Focus

- Coarse searches already succeed: no new origin refinement or changed candidate state.
- Origin was among the ranked searches: no duplicate refinement.
- Refined origin yields nothing or returns zero: retain the unrefined fallback without duplicate scanning.
- No local run exists: retain global xcorr fallback.
- An optional local run exists but original local paths found none: still run the original global fallback.
- Six original candidates already exist: retain all six through actual SFD evaluation, even if the optional origin run is longer.
- Deadline expires before or during refinement: do not start extra work afterward.

## Task 1: Conditional Origin Refinement

**Files:** Modify `src/decoder.py`; create `tests/test_decoder_origin.py`.

**Interfaces:** No public signature changes. The existing `_decode_attempt` search keeps `(run, phase)` entries in `all_runs`, plus `all_syms` and `best_phase` diagnostic state.

- [x] Write focused tests that execute the production search-control block with controlled expensive ranking/refinement/scan boundaries. The missing refined-origin run must fail before the production change.
- [x] Run `/usr/bin/python3 -m pytest tests/test_decoder_origin.py -q`; confirm the expected missing-run failure.
- [x] Add one conditional origin xcorr refinement and nonzero-grid scan after failed coarse searches, checking the existing deadline before refinement and before scanning. Preserve both existing fallback paths.
- [x] Run the focused tests and existing FFT, decoded-span, preamble-fit, tracker-horizon, primary-DD, and deadline tests.
- [x] Replay retained indices064,140,147 serially using the same10-second base budget and single-thread environment. Verify exact packet identity/payload and retain first-arrival/total timings;064 must use padding, and both SHORT_TURBO gains must remain.
- [x] Hand off the diff, test evidence, and replay results to root for the full repository suite and final review. Make no commit.

## Evidence

- RED: missing refined-origin candidate,1 failed/8 passed.
- GREEN:113 focused decoder tests passed.
- Padded retained064: first0.993s,total1.198s, exact expected packet. Gains140 and147 retained exact expected output at5.793s and6.224s total.
- Artifacts: `/home/david/Desktop/LoRa-Regression-Repair-20261030/decoder_origin_replay.json` and matching logs. Root owns the combined full-suite run and final review.

## Review correction

The optional refined-origin run must not seed the existing unrefined-origin bin exclusion set. Review supplied a concrete collision: refined phase13 has four symbols at bin20/start0; unrefined phase0 has eight symbols at bin20/start16. The implementation now snapshots only the original coarse runs for the existing exclusion rule, preserving both runs and diagnostic phase handling.

- RED: both exact-bin and same20-bin-bucket collision cases failed (2 failed, 9 passed).
- GREEN: all11 origin tests passed; combined decoder coverage later passed135 tests after the separately approved capacity change.
- Retained064 after this correction: first0.911s,total1.113s,one exact expected packet with FFT padding still enabled. Evidence: `origin_collision_replay.json` and `064_origin_collision.log/.time` in the diagnosis directory.

## Baseline search and evaluation preservation

Review found two downstream interactions: adding the optional run before global admission suppressed the original global search, and sorting the union before the six-run limit could evict the sixth original candidate. The revised architecture above replaces the earlier insertion strategy, retaining the complete original pipeline and its six candidate slots before adding the optional bounded set.

- Regression tests use real synthetic upchirps/SFD chirps and production candidate scoring: six original runs include F as the only valid SFD, and longer optional G must not evict it. A second case expires the deadline before optional scoring but still evaluates all original runs. The ordinary successful-coarse path keeps its six-run limit.
- Global fallback regression confirms an optional local run does not suppress the original global candidate or alter baseline diagnostic phase.
- RED: three new regression failures against the exact previous source SHA25633a6dc4911e1c7eab2bf7fac43279bdfd8bdcec27c42efad49ad269cd916b586. An external exact copy was used after correcting the test fixture, without reverting production source.
- GREEN:139 focused decoder tests passed in1.91s, including15 origin tests and20 capacity tests. Retained064/140/147 follow-up runs remain under root's CPU coordination.
