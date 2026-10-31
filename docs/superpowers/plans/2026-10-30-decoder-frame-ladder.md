# Decoder frame ladder: truncated frames, grid aliases, grid phase

**Goal:** Remove the per-recording slowdowns the 2026-10-30 corpus comparison still showed against old main (921afce) without losing any output, and without changing the decode path of an attempt the old code would have decoded.

**Scope:** `src/decoder.py` `_decode_attempt` / `_process_file_label`; `tests/test_decoder_frame_ladder.py`. No detector, capture or UI changes.

## Evidence that drove each change

Corpus = the 197 retained recordings listed in `Desktop/LoRa-Regression-Repair-20261030/corpus_final/timing_comparison.csv`, replayed serially old vs candidate with `LORA_DECODE_BUDGET_S=10 LORA_UNKNOWN=1` and single-threaded BLAS (the same environment as the 2026-10-30 comparison). Runner and per-case logs: `corpus_ab.py`, `corpusA..D/` under the session scratchpad; the impairment runner is `impair_ab.py` over the 80 stored manifests under `Desktop/LoRa-AB-Validation-20261029`.

1. **Truncated relay tail (corpus 068/072/078: 9.4/8.2/13.0 s vs old 1.9/1.9/3.2 s).** Each capture holds the decoded frame and, ~200 ms later, a second RF burst with the identical PHY header (a relay rebroadcast) cut off by the capture end 20-66 symbols in. Cross-correlation shows it is a separate transmission, not a sample copy. The old oversized success mask erased it; the exact mask (8e9b008) exposed it, and every retry then read its header, failed its payload, recentered (re-decoding the already-verified frame), armed the carrier rescue (a full recursive ladder) and blocked the fast-path exit — 7 duplicate emissions.
2. **Grid aliases (corpus 010: 2.8 s vs 1.4 s; 052: 13.3 vs 9.8 s).** The FFT padding (c858040) surfaces one preamble as several candidates on different symbol grids; delaying the grid by g chips lifts the dechirped tone g bins, so `(bin - start) mod N` is identical for all of them. The score prefers late grids (the SFD metric cannot see a grid up to a quarter symbol late) and in 010 picked a 24%-late alias whose header still checked but whose payload never CRC'd; old had only the 6%-early alias and decoded on it.
3. **Grid phase is a coin flip on marginal frames.** A forced-offset probe (`gridprobe/`) on SF11 frames that old decoded at the first attempt: one passed at every offset from -512 to +256 chips except -256, another failed at +64..+256 and passed at 0, a third passed attempt 0 only at +256 while its recenter passed at most offsets. A strong frame (068) passed at every offset. Any re-phasing of a grid whose header already checks therefore randomises the outcome against old's result; two corpus passes with an unconditional header-sharpness descent each lost 3-6 first-attempt decodes (found later, slower) while gaining others.

## Changes

- **Truncated frame** (`_decode_attempt`, before the payload decode): a valid header whose frame does not fit the remaining samples in either LDRO mode — or fits only in the opposite mode while the dechirped PMR past that frame's end shows the burst still on — returns FAIL at once with a mask to the capture end, `no_rescue`, and `truncated`. The ladder bans its bin, masks immediately, counts it as a no-header attempt and never recenters for it. The live header checkpoint (`header_only`) is unchanged so capture extension still works.
- **Alias merge** (before the skip-bin filter, so a frame keeps one candidate per attempt and the retry ladder does not spend a full attempt per grid alias of a frame that will not decode): candidates with the same grid-invariant CFO (±3 bins) within two symbols are one preamble; keep the score winner unless another alias reads a checksum-valid header with ≥1.3x its all-codeword min margin (reduced-rate header symbols are forgiving: the 24%-late grid in 010 read only 1.39x below the 6%-early one). The recenter's runner-up attempt is where a frame's next alias gets its turn.
- **Header grid descent, rescue-only** (after the SFD re-anchor): binary descent on summed header peak/mean over ±0.48 symbol; applied only when the current grid reads no valid header, the proposed grid does, and sharpness gains ≥1.5x (fluke headers on noise gained +25..45%). The crop is rotated by -g bins so the preamble reference stays at bin 0.
- **Runner-up alias in a recenter rung**: when the recentered crop's winner read a valid header but its payload failed and another candidate exists, that rung makes one more attempt with the winner banned (the merge then keeps the next alias). Case 010 decodes there (1.9 s) instead of in Pass 2 (2.8 s).
- Candidate list print (`cand:` lines, top 4) and `ALIAS:` decisions are logged for every multi-candidate attempt.

Tried and reverted: skipping the timing sweep on repeats of a frame, and a per-frame budget of four failed payload attempts that then suppressed sweep/recenter/rescue (corpus pass E: it made 172 15.5→5.5 s but lost the weak gained frame 140, which decodes in the carrier rescue after seven failures, and 101 needs thirteen; the 30 s live job's packet was lost the same way); unconditional and margin-gated valid-to-valid descents (corpus passes A/B/C); a 1.3x alias margin bar (flipped 057 the wrong way at 1.37x).

## Results (corpus pass G, final build, 2026-10-31, serial, same machine)

197 recordings, old 921afce vs candidate:
- Outputs: nothing lost; 5 recordings gain a verified packet (101, 125, 127, 140, 147). 188 recordings decode at least one packet vs 183.
- Wall: total 459.8 s → 280.5 s (−39.0%); median per-recording ratio 0.84; p50 1.32 → 1.08 s; p95 10.7 → 3.5 s; max 33.2 → 14.3 s; 24 recordings faster by more than 2 s.
- No recording is slower by both >1.25x and >0.5 s. Slower by >0.3 s: 010 (1.40 → 1.78 s: the score-best alias fails first, the runner-up decodes), 075 (9.97 → 12.37 s: a junk capture with no packet in either version, where the padded crop surfaces more fluke candidates), 095 (3.54 → 3.92 s, junk).
- Duplicate emissions: old 4 recordings, candidate 1.
- The 30.4 s live job from the 2026-10-30 run (SF11/500k shared capture) replays offline in 23.7 s (old) and 6.9 s (candidate), same packet.

80 stored noise/drift impairments (`Desktop/LoRa-AB-Validation-20261029`), old vs candidate: targets recovered 78/80 → 79/80 (gain at weak-boundary noise −20 dB seed 17, sequence 5); nothing lost; wall 151.1 → 115.1 s (−23.8%); no case slower by >0.3 s. The one remaining miss (sequence 9, noise −20 dB) fails in both versions.

Unit tests: 909 passed + 16 subtests (`tests/test_decoder_frame_ladder.py` adds 17: truncated-frame geometry and tail check, ladder handling of a truncated frame, header-descent leave-alone/noise/rescue rules, alias merge, margin preference, runner-up handover).

## Live run 2026-10-31 (ladder build 3501fb4) and what it exposed

Same 13-message USRP/Heltec matrix as the d3eb682 reference run (`Desktop/LoRa-Local-Repair-20261028/live_validation/ladder_final_3501fb4`, run with the plain runner; Codex's `launch.py` wrapper was not used, so the host sampler was started separately and the harness-hash/detector-stdout receipts are missing — procedural, redone in the re-run below). 13/13 messages recovered, drain clean, Heltec restored, zero ring drops, 212 decode jobs with no timeout. Publication latency median 3.16 s (reference 2.84 s), max 5.38 s on the SF11/500k message (reference 3.61 s); both runs sat at a 95-105 °C package temperature with heavy thermal throttling, so single-message latencies are not a paired comparison. Two findings were real:

1. **Long SF12/62.5k crops ran 37-59 s** (reference max 30.4 s; the SF12/62.5k job time summed to 317 s against 108 s). The corpus has no SF12/62.5k recording at all. Offline replay of the live crops: the runner-up attempt of the recenter rung was repeating the winner's attempt verbatim — it banned the bin the attempt *returned* (refined by the SFD re-anchor / grid descent, which moved it by far more than the 20-bin ban radius on these crops: a -2560-sample re-phase) instead of the finder's chosen bin, so the merge handed nothing over and the all-filtered fallback re-picked the winner, sweep and curved-drift retries included: +2.5 s per rung, three rungs per crop. Fix: the attempt records `chosen_bin` and `alt_candidates`; the runner-up bans the chosen bin and runs only when another finder candidate lies outside the ban radius. With the ban working, the takers on these crops (aliases that read no header, or a weaker one, at the finder grid) still read a valid header in a full attempt and then failed the payload exactly like the winner: 0 decodes in 12 such attempts. So the rung is also gated on `runner_up_ratio` ≥ 1: the absorbed alias must have read its header at least as well as the winner at the finder grid — the near-tie the 1.5x merge bar deliberately left in score order, which is the case the rung was built for (010 decodes on its 1.39x alias; 074: 33.0 → 29.9 s = d3eb682's 29.3 s; 075: 56.3 → 47.2 s, old 41.8 s).
2. **One published `candidate`-confidence Meshtastic record** (message 7, 21 wrong bytes, CRC-16 matched: a sweep collision at one of ~170 timing×chase combos, then AES failed so it was published unverified). Not a regression: the old decoder emits the identical record from the same crop (072) offline, as does d3eb682. The sweep's 16-bit acceptance is the pre-existing exposure (≈0.26 % per failing sweep); left as is and documented.

The alias merge's finder-grid header read is unreliable on these crops (margins marked invalid for grids whose full-attempt header checks) — it is used only as a relative discriminator, never to drop a candidate.

Re-run: see the ledger.

## Ledger

2026-10-31: corpus passes A–G and the forced-offset probe are under the session scratchpad (`corpus_ab.py`, `impair_ab.py`, `gridprobe.sh`); per-case logs and `summary.json` in `corpusG/` and `impairG/`. Committed as the frame-ladder change on `fix/no-performance-regressions`; nothing pushed.
