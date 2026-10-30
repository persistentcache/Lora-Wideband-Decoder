# Shared Slow Admission Latency Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. The root agent owns the reviewed baseline, CPU schedule, source freezes, and final retention decision.

**Goal:** Remove the extra slow-result hold before shared capture admission while preserving the original sample span, full fallback tail, history identity, legacy routing, and packet recovery.

**Architecture:** At the existing slow-result recorder branch in `detector.main`, preserve the original `_slow_record_immediately` partition for both recording paths. Shared recording sends each nonempty group through `_record_slow_capture` separately in the same turn; legacy recording retains its existing due-tick hold for the later group. Separate calls preserve the recorder's existing per-group label arbitration, full tail, and export geometry. The shared helper registers the original span without copying a delayed raw tail. Header and full fallback views retain their existing independent ownership and save policy.

**Tech Stack:** Python, pytest, NumPy metadata-only fixtures, existing native raw replay and live validation harnesses.

**Spec:** Root's delegated latency experiment in this session. Diagnosis: `/home/david/Desktop/LoRa-Regression-Repair-20261030/capture_diagnosis.txt`. Historical evidence: `/home/david/Desktop/LoRa-AB-Validation-20261029`.

## Constraints and decision gates

- Preparation changes only this plan and the pending routing test outside repository test discovery. No production edit or test execution before GO.
- Root first completes review, full verification, and a frozen source commit for the repair baseline. Do not let the intentionally failing latency test enter the base suite.
- After GO, copy `/home/david/Desktop/LoRa-Regression-Repair-20261030/test_shared_slow_latency.pending.py` to `tests/test_shared_slow_latency.py`, prove RED, and make the smallest shared-only routing change. No permanent experiment flag is needed: baseline and experiment use separate frozen source commits.
- Preserve the 3 GiB capture budget, existing CRC and full fallback behavior, all admitted spans, legacy hold timing, and early/full keep-N isolation. Do not broaden this slice into dedup redesign, worker scheduling, or hardware retune handshakes.
- Preserve both original registration groups: merging them can let `update_deferred` discard a weaker conflicting label within 6 kHz and lets `_register_shared` select a common strongest-label export geometry and maximum tail.
- Tests and replays run sequentially under root's CPU grant. No commit or hardware action by this delegated agent.
- Treat the change as an experiment until paired replay and the exact affected live modes establish its recovery, latency, and added-work effects.

## Why this slice needs measurement

At the default four-window commit lag, wider slow results wait `min(_LAG + 6, 8)` half-second ticks before registration. Shared header planning cannot begin before that registration. The saved `03_new` sequence 7 SF11/500 case took 7.601 s from transmit-completion receipt to submission, then 1.103 s to its decoder JOB; `06_new_extended` sequence 2 SHORT_FAST took 7.038 s before submission and 0.066 s after decoder start. These logs identify delay before decoding, but do not isolate the hold from detection and crop time.

Immediate shared registration preserves sample geometry, but can increase work. Early captures deliberately bypass full-save keep-N reads and writes. More admitted views can therefore produce more header attempts, early files, decoder submissions, and competing full saves even when later full-save dedup suppresses some copies. The current shared full-save decoder-cap refusal is fatal, so added pressure must be measured before retaining the change. Existing raw replay inputs do not include the two exact delayed live cases.

## Task 1: Prove the routing contract

**Files:** Create `tests/test_shared_slow_latency.py` after GO; then minimally modify the slow-result recorder branch in `src/detector.py`.

- [x] Stage a test outside discovery that compiles and executes the actual production routing and legacy hold blocks, rather than a duplicate routing model.
- [x] Cover the observed SF11/500 and SHORT_FAST SF7/250 labels at the real 10 Msps geometry. Use a broadcast base so the test does not allocate a multi-second IQ array.
- [x] Assert shared admission occurs in the same turn through the real `_record_slow_capture`, with the identical base object, detector/raw start, base endpoint, full fallback tail, and history anchor. Assert no shared hold entry remains.
- [x] Assert legacy results remain held until the existing due tick for zero, default, and larger commit lag; preserve the same base, anchor, and helper arguments when released.
- [x] Exercise real `SignalRecorder.update_deferred`, `_register_shared`, and `SharedCapture.register` for a mixed SF12/62.5 and stronger SF11/125 batch. Preserve both labels at 0 and 5 kHz separation, and distinct endpoints/export rates at 20 kHz separation.
- [x] Root freezes the reviewed repair baseline and grants CPU; copy the pending test into `tests/`, run the focused command, and retain RED output. Expected: two shared tests fail for the legacy hold, six legacy controls pass.
- [x] Implement the shared-only immediate branch and update its comment. Preserve the legacy partition unchanged.
- [x] Update the existing retune slow-result test's observation if necessary: it currently treats a hold entry as evidence of accepted shared work. Observe admission instead while retaining rejection before old-generation bookkeeping/publication.
- [x] Run focused GREEN plus `tests/test_detector_retune.py`, `tests/test_slow_capture.py`, and `tests/test_header_recorder.py`. The latter already proves early-save keep-N isolation and independent full fallback.
- [ ] Root reviews and freezes the experiment source. Release CPU promptly and record exact source IDs and test logs.

## Task 2: Compare the same three retained raw inputs

Run both frozen sources with identical settings, pacing, and complete inputs, with fresh capture/cache/output directories and no competing tests or benchmarks. Use the existing harness and the hashes in `native_replay_summary.json`:

- `/home/david/Desktop/LoRa-Local-Repair-20261028/raw-diagnostic/raw.cs16`
- `/home/david/Desktop/LoRa-Local-Repair-20261028/raw-sf10-20261029/raw.cs16`
- `/home/david/Desktop/LoRa-Local-Repair-20261028/raw-sf12-250-20261029/raw.cs16`

- [ ] Record source IDs, input hashes, full commands/environment, pacing, fresh output paths, and machine/thermal conditions for both arms. Keep `detect_workers=2`, shared capture/header checkpoint enabled, and `LORA_SHARED_CAPTURE_MB=3072` fixed.
- [ ] Compare exact independently CRC-validated packet/hop recovery and full fallback coverage, first useful admission/prefix/submission/publication times, header attempts, early/full files and submissions, total decoder JOBs, worker CPU, peak charged capture/decode bytes, backlog, refusals, skips, and errors.
- [ ] Review additional save/decode work explicitly. A lower first-packet latency does not establish acceptable aggregate throughput or bounded-capacity behavior.

## Task 3: Reproduce the exact delayed live modes before retention

- [ ] Root uses the saved `03_new` sequence 7 CUSTOM_SF11_500 and `06_new_extended` sequence 2 SHORT_FAST settings, payload lengths, and packet identities from their receipts as the controlled-case template. Pair the frozen baseline and experiment with the same authorized RX/TX settings and payload structure.
- [ ] Correlate transmit-completion receipt, detection-result availability, capture admission, prefix/header planning, submit/start, and publication using existing logs where available. If an event is unlogged, record that limit rather than infer its timestamp from another event. Any added instrumentation is a separate reviewed change applied equally to both arms.
- [ ] Validate exact decoded payload/CRC and sample-span attribution to the intended transmission; distinguish a delayed first attempt from a later repeat.
- [ ] Root retains the routing change only after recovery and fallback remain intact, no new pressure failures or admitted-span loss appear, and measured latency benefit justifies added work. Otherwise revert the experiment source change and preserve the evidence.

## Execution record

- Root froze baseline `ba9206b01946767bb77d80b1f563a93a77bace3b` after its base repair verification, then granted the latency RED/GREEN CPU slot. The pending test entered `tests/` only after that gate.
- RED: `tests/test_shared_slow_latency.py` produced two failures and six passes in 0.18 s. Both shared labels waited in the legacy hold; every legacy due-tick control passed. Evidence: `/home/david/Desktop/LoRa-Regression-Repair-20261030/shared_slow_latency_red.log`.
- The production edit is confined to the existing slow-result routing branch: shared results go to the real helper immediately; the legacy partition and hold calculation are unchanged. The retune regression now observes accepted admission instead of a hold entry and still rejects old generations before bookkeeping/publication.
- GREEN: the new routing tests plus detector retune, slow capture, and header recorder tests passed all 57 cases in 1.60 s, with numerical-library thread counts set to one. Evidence: `/home/david/Desktop/LoRa-Regression-Repair-20261030/shared_slow_latency_green.log`.
- Review found that the initial shared route merged the two original registration groups. The new real-admission regressions reproduced the resulting label loss and changed geometry: three failures and eight passes in 0.23 s. Evidence: `/home/david/Desktop/LoRa-Regression-Repair-20261030/shared_slow_mixed_red.log`.
- The correction restores the original group construction exactly and admits each nonempty shared group separately in the same turn. Legacy hold timing stays unchanged. Focused GREEN, including the actual shared-recorder regressions, passed 70 cases in 1.91 s. Evidence: `/home/david/Desktop/LoRa-Regression-Repair-20261030/shared_slow_mixed_green.log`.
- CPU released to root after focused GREEN. No delegated commit, benchmark, replay, or hardware run. Root review, experiment source freeze, paired replay, and exact live-case validation remain outstanding; no measured latency improvement is claimed yet.
