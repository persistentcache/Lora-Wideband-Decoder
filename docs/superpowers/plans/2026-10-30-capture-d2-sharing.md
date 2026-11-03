# D2 Capture Sharing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve simultaneous nearby 5 Msps captures from the 10 Msps input within the configured 3 GiB aggregate capture budget.

**Architecture:** For target decimation two, retain one unfiltered full-rate parent and apply the existing D2 mixer/FIR separately when each view is rendered. All other factors, including the exact-center D5 path, retain their existing plan. The existing reference retirement and byte/outbox accounting continue to own the complete admitted spans.

**Tech Stack:** Python, NumPy, the existing streaming FIR, pytest.

**Spec:** The root agent's approved D2-only task in this session; supporting diagnosis is `/home/david/Desktop/LoRa-Regression-Repair-20261030/capture_diagnosis.txt`.

## Global Constraints

- Modify only `src/capture_timeline.py`, `tests/test_capture_d2_sharing.py`, and this plan for the implementation.
- Keep `LORA_SHARED_CAPTURE_MB=3072`; retain the 671088640-byte header workspace and bounded prefix reservations.
- Preserve every admitted full span and each view's requested frequency, sample rate, integer start, and real FIR right context.
- Leave admission failure policy, retune handling, and workspace shrinking for the root agent's separate repair.
- Wait for the root agent's GO before running tests; another agent owns the CPU benchmark slot.
- No hardware, no benchmarks, no commits; the root agent owns final integration and broad verification.

## Review Focus

- Four or more nearby D2 centers must not allocate one retained band per frequency.
- An odd first raw sample must retain the per-view mixer phase and output grid.
- Arbitrary hop seams must match a direct single-stage D2 FIR within floating-point precision.
- Input-edge signals remain available even when their oversampled export crosses the input edge.
- Writer-held samples remain charged until release, and D5 retains its existing bounded exact-center behavior.

---

### Task 1: Share the raw parent of D2 views

**Files:**
- Create: `tests/test_capture_d2_sharing.py`
- Modify: `src/capture_timeline.py`, `SharedCapture._spec`

**Interfaces:**
- Consumes: `SharedCapture.register(...)`, `SharedSave.stream(index)`, `SharedSave.release()`, and `CaptureBudget.stats()`.
- Produces: unchanged public interfaces; `_spec(2, frequency)` selects the full-rate input parent.

- [x] Write a metadata-only regression admitting four and eight SF12/500 candidates, each with a 40000000-sample base and endpoint 75082240, plus the actual header workspace and two maximum prefixes. Assert all admissions, one parent, unchanged ranges/rates, budget bound, and complete release.
- [x] Write small-array FIR regressions with odd and even starts, three hop sizes, four distinct centers including the input edge, and direct `_StreamDecimator(..., D=2)` reference outputs. Assert full lengths, correct offsets, and retirement/release.
- [x] Write a right-context regression proving a D2 view is withheld until real trailing samples arrive.
- [x] After GO, run `python3 -m pytest -q tests/test_capture_d2_sharing.py`; verify the new capacity test fails at the fourth admission because of the current per-center bands.
- [x] Special-case target factor two alongside factor one in `_spec`; document why D2 shares full-rate input while larger prime factors retain exact-center storage.
- [x] Run the focused D2 tests and existing capture/header tests with numerical-library thread counts set to one. Expected: all pass, including the existing three-center D5 test.
- [x] Inspect the final diff, record RED/GREEN evidence below, and hand it to the root agent for integration without committing.

## Execution Record

- Design chosen by root: full-rate D2 parent; the alternatives of a larger budget or dropping nearby frequency candidates do not meet the repair requirement.
- Existing isolated worktree verified: `fix/decode-throughput-regression` at `b8a3456`.
- RED: `tests/test_capture_d2_sharing.py` produced five failures and one pass in 0.40s. The budget tests failed at the fourth admission; the sample cases reported four retained bands instead of one.
- GREEN: the six D2 tests passed in 0.52s; the existing capture/header regression set passed 174 tests in 6.29s, including D5 coverage. Logs: `capture_d2_red.log`, `capture_d2_green.log`, and `capture_header_green.log` under `/home/david/Desktop/LoRa-Regression-Repair-20261030`.
- No hardware, benchmark, full-suite run, or commit was performed in this delegated slice. The root agent owns final integration and broader verification.

## Delegated detector-main retune integration

The root agent separately assigned `src/detector.py` main-loop integration and `tests/test_detector_retune.py` after implementing the capture/recorder rollover APIs. This follow-on leaves those APIs and `capture_timeline.py` untouched.

- The control handler invokes the recorder rollover and invalidates live history exactly once. It records the shared capture's received raw boundary in detector-clock coordinates, preserving monotonic total and sequence counters.
- Sliding window, carry, pre-hop, slice cache, slow assemblies/holds, and bin-based slow trigger state reset. Buffered pre-boundary samples remain consumable but cannot enter a new detection window or lookback.
- In-flight pool slots and the existing slow worker keep their ownership until completion. Old-generation results are discarded before publication, trigger bookkeeping, or capture registration.
- The existing independently polled hardware control cannot establish a precise acquisition-side retune timestamp. This slice records the published receive boundary and does not add a native hardware handshake.
- RED: eight failures and three passes in 0.21s. GREEN: eleven focused tests passed in 0.18s; the related shared-recorder, slow-capture, seam, airtime, published-history, and D2 regression set passed 189 tests in 2.77s. Logs: `detector_retune_red.log`, `detector_retune_green.log`, `detector_retune_regression.log` in the repair evidence directory.
- The first GREEN attempt exposed a test-fixture assumption: production CS16 uses the B200 12-bit scale of 1/2048, not 1/32768. Correcting the independent expected sample values resolved that test-only mismatch.
