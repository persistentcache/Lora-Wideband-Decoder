# Bound shared capture save workspace

> **For agentic workers:** Use superpowers:executing-plans for this focused correction and request an independent review before the next hardware regression.

**Goal:** Preserve complete 8x IQ captures without exhausting the existing 3 GiB shared/header budget on overlapping SF12/250 requests.

**Architecture:** Keep capture spans, filter centers, decimation, detection, decoder validation, and the memory limit unchanged. Recenter the shared renderer's private output in bounded tiles using the exact existing arithmetic. Reserve two full output arrays plus the existing bounded workspace instead of eight full arrays.

**Tech stack:** Existing Python/NumPy recorder and pytest.

**Spec and authorization:** The local reliability plan and the user's instruction to continue until low latency, no misses, and no sample drops are validated. The user delegated implementation choices. This remains an opt-in local experiment; no Pi deployment or main push follows from a unit test.

## Evidence and alternatives

The original-charger live run `e64_original_charger_live_full_01` recovered the first five controlled messages in 1.146–3.385 seconds after first RF completion. It then failed admission before controlled message six, with `band_retention growth exceeds capture byte budget`; cleanup restored Heltec and stopped the receiver. The run is incomplete/failed, never acceptance evidence.

A metadata-only reproduction uses rate 10 Msps, hop 5M samples, 671088640 bytes header workspace, SF12/250 tail 70164480 samples, a 4-second base, and three 100-Hz-separated D5 centers. The third reservation exceeds 3 GiB. One normal 1.5-second-base view reserves 2505603072 bytes, matching the live log exactly. No retention leak is required. A fresh native recording is being replayed with exact admission snapshots.

Increasing the limit would hide the workload cost; merging frequency hints could alter requested IQ. Bounded recentering preserves every requested sample and reduces actual transient owners. The lifetime maximum writer lease stays conservative: shrinking it requires accounting for header holds that can create early views after the parent finishes, which is a separate lifecycle change.

## Constraints and review focus

- Keep the shared renderer's source chunks immutable; only its private complex64 output may be mixed in place.
- Keep absolute float64 sample indices, coefficient evaluation, complex128 exponent/product, and final complex64 conversion identical to the existing expression.
- Bound each tile to `min(65536, hop_n)`; the fixed temporary allowance must also cover small-hop configurations.
- Preserve two complete outputs in the lease for the previous preamble during the next render and the current same-dtype `.astype()` write copy.
- Do not reduce the existing second-stage FIR/phase workspace, lifetime maximum lease, capture endpoints, optional-header behavior, or memory limit.
- Review identified an8192-sample fixed FIR mixer block even when hops are smaller. Floor both crop/save workspace-hop terms at8192; real allocation probes must remain below their leases.

## Implementation and verification

- [x] Add failing tests for the real three-band admission, exact sample bits across positive/negative/fractional shifts and tile seams, and the actual save worker's bounded temporary arrays and unchanged output.
- [x] Implement `_mix_capture_inplace(iq, offset_hz, fs, chunk_n)` in `src/detector.py`; use it only for shared capture saves. Keep the legacy path intact.
- [x] Change only the full-output coefficient in `src/capture_timeline.py` from eight to two, retaining the fixed workspace term and ownership rules.
- [x] Run focused tests, then the complete suite while no timed replay or live capture is running. Request independent source/owner-accounting review.
- [x] Freeze the candidate; replay the complete new SF12/250 recording. Require all input, complete views/jobs, exact identity/received CRC, and no fatal/loss/truncation. Repeat the existing long and baseline latency controls if the capture changes require it.
- [x] Run a new complete live matrix on the original charger; independently verify first-transmission latency, identities, received CRC, native sample continuity, downstream queues, exact source, and cleanup. Preserve all failed runs separately.

Validation before freezing: seven primary regressions failed against e64, then passed; review exposed and reproduced both small-hop allocation gaps before the workspace floors. Final full suite: 695 passed in18.87s. Independent review found no remaining blockers; real small-hop writer peak520538 bytes is below749824 reserved. Native replay and live acceptance remain pending.

Final validation: b8a3456 full186.5613312s SF12/250 replay completed all53jobs with first-RF3.0798s and fresh58-byte CRC47ce. The live10-case matrix completed cleanly:10/10 first-transmission identities/PHY,1.1264–3.9458s publication,10/10 fresh independent receivedCRC bindings,214/214 jobs with0timeouts,15,170,732,032 native samples with0timestamp gaps/errors,377health snapshots with0reported ringdrops. All7nodeinfo and1unknownraw supplemental frames also verify numericalCRC. Heltec/channel controls restored, no ownedprocesses remain, frozen35sourcefiles unchanged. Independent finalreview accepts onlythis measured desktop scope. Existing gates shed170candidate peaks;2out-of-band footprints notadmitted. Forcedshutdowntail, phone rendering, everyPHYcombination/interference condition, defaultshared/retune rollout andPi behavior are outside this proof. Evidence: Desktop/LoRa-Local-Repair-20261028/live_validation/b8a3456_original_charger_live_10_01/VALIDATION.txt andvalidation_summary.json.
