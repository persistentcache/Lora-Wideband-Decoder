# Decoder Payload Capacity Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this focused task. Finish the origin-search review correction first. Keep all changes uncommitted for root review.

**Goal:** Avoid payload recovery trials which cannot supply the required bytes from the recorded samples in their current FEC/LDRO mode.

**Architecture:** Add `_payload_fits_samples(sample_count, data_start, N, ppm, ppm_pay, cr, need_bytes, min_offset=0)` using existing `_frame_data_symbols`. Move the unchanged timing-offset enumeration ahead of PRIMARY-DD and evaluate capacity at its earliest offset. Gate only PRIMARY-DD, the timing sweep, and late curved-DD. This is an attempt-local capacity bound, not a claim that the header or entire capture is correct or exhausted.

**Tech Stack:** Python, NumPy/SciPy, pytest.

**Spec:** `/home/david/Desktop/LoRa-Regression-Repair-20261030/decoder_diagnosis.txt`, section B, and root's approved narrow pruning scope on2026-10-30.

## Global Constraints

- Preserve primary decoding, try-both drift, alternate LDRO, frac-park, partial payload/margins, header diagnostics, no_rescue flags, and outer header/recenter/carrier-rescue behavior.
- Preserve exact successful masks, complete adjacent frames, overlapping traffic, CRC-absent protocols, both SHORT_TURBO gains, and long SF12 recovery.
- Do not infer missing CRC bytes or claim that a short selected mode excludes a viable opposite LDRO mode.
- Preserve all timing offsets and current CPU deadlines. No commits or hardware activity.
- Coordinate every CPU test/replay with root.

## Review Focus

- Index78 geometry: LDRO-on requires68 symbols while LDRO-off requires58; the alternate primary still runs before pruning.
- A negative timing offset can make a barely short nominal frame complete; the bound must honor the actual earliest searched offset, including dec greater than8.
- Header overflow and whole FEC blocks determine the required symbol count, including optional CRC bytes.
- Short and long complete frames still reach existing recovery; invalid current geometry must not suppress outer attempts with a different header.
- Frac-park changes diagnostic margins and no_rescue policy, so retain its execution and state even for a short payload.

## Task 1: Current-Mode Capacity Bound

**Files:** Modify `src/decoder.py`; create `tests/test_decoder_capacity.py`; add only complete-frame metadata to `tests/test_primary_dd.py`'s isolated fixture.

**Interfaces:** `_payload_fits_samples(...) -> bool` compares available complete symbols at `min_offset` against `_frame_data_symbols(...)`. `_decode_attempt` computes `_payload_recovery_fits` from the current mode and the existing timing-offset list; no return/status/diagnostic interfaces change.

- [x] Add helper tests with independently counted68/58/168/144/392-symbol frames, CRC-present/absent sizes, exact-end and one-sample-short boundaries, and earliest timing offsets.
- [x] Execute the production primary/alternate-LDRO blocks and the three recovery guards in focused tests. Demonstrate alternate-LDRO success before pruning, failed alternate restores the original mode, complete geometry keeps recovery, and impossible geometry adds no recovery trials.
- [x] Run the focused tests against unchanged capacity behavior and confirm expected missing-helper/extra-trial failures.
- [x] Add the helper, move the existing timing-offset construction without changing its output, and guard PRIMARY-DD, timing sweep, and late curved-DD with the shared capacity result. Do not change frac-park or outer rescue behavior.
- [x] Run focused capacity/origin/FFT/span/preamble-fit/tracker/PRIMARY-DD/deadline checks. Keep `test_primary_dd.py`'s prior tests in a complete-frame context via `_payload_recovery_fits=True`.
- [x] Replay68/72/78 and complete adjacent0037 sibling073 under the existing budget, checking exact output and separate first-arrival/total CPU metrics. Retain original078's missing-CRC behavior.
- [x] Hand root the diff, timing evidence, preserved outputs, and any remaining duplicate-work limitation for full-suite/corpus review. Do not call the broader rescue regression solved unless its measured total cost is actually resolved.

## Evidence

- RED:14 failed,6 passed; missing capacity helper and the three expected extra-recovery assertions.
- GREEN:135 focused decoder tests passed in1.85s, including20 capacity cases and11 origin cases.
- Single retained replays, exact packet semantics preserved:068 first1.202s,total9.776s,CPU9.75s;072 first1.376s,total8.657s,CPU8.61s;078 first2.595s,total13.128s,CPU13.10s.
- Complete sibling073 still verifies0037 (pktid0x288de646), first1.236s,total1.555s,CPU1.50s. Original078 still emits only0036; the unavailable CRC byte is not inferred.
- Total wall time improves approximately14%,21%,32% against prior three-repeat confirmation medians. These are single repair replays, not repeated benchmark estimates. Duplicate emissions remain7,7,5, so target identity in outer rescue is still unresolved.
- Source SHA256:33a6dc4911e1c7eab2bf7fac43279bdfd8bdcec27c42efad49ad269cd916b586. Evidence: `/home/david/Desktop/LoRa-Regression-Repair-20261030/capacity_repair_replay.json` and matching log/time files.
- Final-source preservation checks:064 first0.952s,total1.152s;140 first3.954s,total5.711s;147 first4.076s,total5.752s. All exact expected packet semantics retained. Evidence: `final_decoder_replay.json` in the same directory.
- CPU reservation released to root after the serial replays. No full suite, radio work, or commits were performed by this agent.
