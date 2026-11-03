# Bounded wide-scan resolver cache implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** Remove repeated narrowband FFT extraction within one energy peak without changing detection decisions.

**Architecture:** Give the global resolver a private per-peak narrowband cache. Preserve its existing whole-buffer, zero-offset extraction, sampling clocks, candidate order and position-specific quality calculation. Reuse only already computed arrays, with at most 16 MiB of retained arrays in this new cache per peak; oversized/overflow entries follow the existing uncached path.

**Tech stack:** Existing Python/NumPy/SciPy detector and pytest.

**Spec:** This document's architecture and constraints; the larger acceptance contract remains `2026-10-28-local-reliability.md`.

## Evidence and decision

Exact SF10 first-frame and noise windows under the actual `LORA_SCAN_FULL=1` configuration spend about 24% of isolated wall time repeating identical extractions (61 calls/21 unique and 36/17). The earlier preset-mode profiles did not reproduce the full-scan resolver; retain them as separate diagnostics, not evidence for full-scan timings. Single-core placement halves throughput and is rejected. Global resolver caching is a smaller, directly measurable change than changing queue admission or allowing more memory backlog.

## Constraints and review focus

- Do not change energy candidates, thresholds, lag arbitration, positions, rates, filters or CRC validation.
- Do not seed this cache from the chunked matched-filter arrays: those are different numerical inputs.
- Scope the cache to the same immutable per-peak 1 Msps input; never reuse it across peaks/windows/recentering.
- Bound retained array bytes even on long slow-pass windows; overflow must compute normally.
- The global resolver only reads its cached arrays. Preserve uncached callers and preset mode.
- No Pi deployment or Git push; no RF until isolated evidence justifies a live trial.

## Task: reuse global resolver extractions

Files: `src/detector.py`, `tests/test_resolver_cache.py`.

Interface: `_resolve_sf_bw_global(..., nb_cache=None)` accepts an optional caller-owned dictionary of bandwidth to narrowband array, scoped to one `iq_1m`. The detector constructs a fresh dictionary for each peak, separate from the existing matched/preset cache.

- [x] Write real-transform tests showing repeated harmonic votes reuse narrowband extractions while keeping exact results, separate position votes, source immutability and the byte bound. Observe the expected missing-argument failure before implementation.
- [x] Add the optional cache and per-peak wiring, without changing the extraction or quality algorithms.
- [x] Run the focused tests and full repository suite with radio/replays idle.
- [x] Compare exact recorded-input tasks against e64a05a under full-scan settings: complete detection dictionaries must match; record transform counts, timings and hashes.
- [ ] Export the exact candidate and run the full paced SF10 input with original 8x capture settings. Audit complete feeding, capture/job completion, strict identity, fresh received-byte CRC and first-RF latency. A latency failure is not acceptance.
- [ ] Review and retain only a verified improvement. Long-input/live regression remains necessary for overall acceptance.

## Execution ledger

2026-10-29: User already delegated implementation choices and requested continued testing. Execute inline in the existing isolated repair worktree. No approval round or main-branch mutation is needed.

Independent read-only review found no critical/important issues. Clarified that the new cache bound does not bound total peak memory (the existing dechirp memo also pins arrays). Direct resolver tests do not independently cover all recenter/thread wiring; current scope was reviewed and two full detector-window comparisons match exactly. Full replay remains pending.

Full8x replay rejects integration: baseline completed132.54s with14.053s latency (8jobs), but candidate failed after31s when raw145–150M was overwritten. Faster detection admitted7independent D5 filter bands. Only1,243,131,072/5,301,600,256input bytes fed, exit-6; early exact packet emissions do not excuse downstream loss. Application source restored exactly to e64a05a. Candidate source, patch, tests and plan remain frozen under the external resolver_cache evidence directory. Next experiment uses the existing4x capture setting to share first-stage filtering; this changes archived IQ rate and is not unchanged-configuration acceptance.

## Bounded worker FFT execution diagnostic

The original two-worker queue is a compute deficit; four workers with the same native/cache/serial candidate completes but worsens firstRF latency to14.675s. Isolated single-thread FFTW ESTIMATE with explicit input/output copies improves the costly1D transforms by~1.5–2.5x; MEASURE adds~0.36–1.02s startup for little extra benefit. Batched2D regresses and remains SciPy. This is a test-only external prototype, not production integration.

Architecture: install a direct FFTW1D backend only inside already-forked detector workers. Eligible complex64 one-dimensional transforms use identical lengths, orientation and normalization, copy caller input into aligned private buffers and return independent output. Use ESTIMATE (no timed measurement), an LRU bounded to64MiB of owned input/output arrays, and the original backend for unsupported/oversized inputs or allocation failure. FFTW internal workspace is additional and must not be described as bounded by this array budget. No interface cache thread, home wisdom file, parent scanner, decoder, thresholds or rate changes.

- [ ] Write failing tests for numerical DFT/FFT agreement, inverse normalization, irregular strides/input immutability, output ownership across reuse/eviction, exact buffer bound, fallback and worker-only installation.
- [ ] Implement external helper and pass focused real-transform tests; inspect child stats and no thread/fork side effects.
- [ ] Compare exact retained worker tasks against original SciPy results; record numeric differences and complete detection outputs. Reject lost/added identities before timed replay.
- [ ] Freeze candidate and run full132.54s original8x SF10 replay with unchanged2worker controls. Audit input completion, allviews/jobs, receivedCRC and firstRF latency.
- [ ] Review and validate long-input/matrix/live before any integration claim.

FFT diagnostic ledger:16initial tests pass0.53s; exact retainedseq1/21/22/24 full detection dictionaries match SciPy/FFTW/SciPy with unchangedIQ/PSD. Isolatedwall1.436/1.216/1.357s (seq1),0.450/0.409/0.445s (seq21),0.595/0.563/0.582s (seq22),0.355/0.323/0.351s (seq24). Backend difference is floating-point; these matching decisions are four controls only.

Review found output-copy MemoryError, planner-NULL RuntimeError, exception-held partial allocations, and aligned backing-padding accounting gaps. Five targeted assertions first fail, then20totaltests pass0.54s after the single fix pass. Fallback now runs outside the exception context, releases plans first, catches only the documented planner-NULL RuntimeError, and accounts both alignment allowances. Initial huge-ramp pointwise comparison was mis-scaled; relativeRMS1.18e-7 confirmed; test uses the same normalized error criterion as other large transforms.

Ruling: keep the FFT prototype limited to the local SciPy-backed detector. Hook now sets original child environment/module path before import, rejects a non-SciPy baseline explicitly, and records parent/child import/thread receipts. Missing-library installation and forced termination can still prevent final stats; this is a local diagnostic prerequisite, not portable production behavior. Original short3s preflight completes317FFT calls across two workers, no failure. Revised preflight checks cold child import and normal cleanup before full replay. No global/full memory bound is claimed beyond owned aligned arrays; internal FFTW workspace remains additional.
