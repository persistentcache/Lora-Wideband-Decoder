# Detector edge-window correction

> **For agentic workers:** Use superpowers:executing-plans inline. The independent decoder/capture investigations do not modify this task's files.

**Goal:** Prevent the reproduced carrier-width exception from discarding a complete detector window.
**Architecture:** Clip the carrier-local peak-search interval to the actual PSD array. Keep the existing edge-contour abstention, detection thresholds, family arbitration and all ordinary searches unchanged.
**Tech Stack:** Python, NumPy, pytest; retained real IQ for integration verification.
**Spec:** User instruction to continue repairing confirmed regressions; evidence in Desktop/LoRa-AB-Validation-20261029/worker_reproduction. Both frozen versions fail naturally at the same window, at the negative-start PSD slice. Implementation choices already delegated by the user.

## Global Constraints
- Existing isolated fix/decode-throughput-regression worktree; preserve unrelated dirty documentation and immutable old/new evidence.
- No Pi changes or pushes. Hardware remains exclusive and serial. No benchmarks alongside tests.
- Do not suppress the exception at worker level or return an empty detection set as the fix.

## Review Focus
- Negative slice starts must not wrap to opposite-band energy.
- Positive upper edges must keep ordinary NumPy-clipped behavior.
- A contour touching a PSD edge must abstain from width relabeling.
- Ordinary interior widths must remain identical.
- Actual retained detector calls must complete and preserve existing successful detections.

### Task 1: Bound carrier-width peak search

Files: src/detector.py; tests/test_carrier_width.py. Interface remains detect_preamble and its existing internal carrier-width branch.

- [x] Add failing real-branch tests with 4096-bin PSDs: left/right/interior plateaus, opposite-edge distractor, and edge-touching abstention. Execute the actual branch and edge walker, not a duplicated algorithm.
- [x] Run /usr/bin/python3 -m pytest -q tests/test_carrier_width.py; expect the reproduced empty-argmax failure on left-edge cases.
- [x] Clip search start/end to PSD extent and add the clipped start when recovering the absolute peak index.
- [x] Run focused tests and the full pytest suite; all pass, preserving ordinary widths and abstention.
- [x] Run the retained natural and targeted-shoulder detector reproductions against the corrected worktree, compare existing detections, retain diagnostics and hashes.
- [ ] Commit only this correction/test/plan after review and verification. Follow-on capture and decoder fixes have separate root-cause evidence and validation gates.

Evidence: RED 3 failed / 6 passed; focused GREEN 9 passed; first full suite 704 passed. Retained IQ: 40 detector calls, zero errors, seven previously failing calls now complete, and all 33 previously successful calls unchanged. Independent review found no blocker. The integrated repair suite subsequently passed 772 tests in 19.68 seconds. Root will commit the reviewed repairs together for an immutable native/live comparison baseline; the decoder and capture changes retain separate plans and evidence.
