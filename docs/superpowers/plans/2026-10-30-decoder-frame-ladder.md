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

## Live run 2 (2b7d58f, Codex's launcher) and the two fixes it drove

Same matrix through `prepare_live.py` + `launch.py` (`live_validation/ladder_live_2b7d58f`, metrics in `live_metrics_ladder_live_2b7d58f.json`): 13/13 recovered, `all_messages_verified`, zero unverified records, independent CRC audit 20/20, harness receipts valid; the only `live_metrics` acceptance reason left is one message whose firmware log shows two completions for one start (an attribution limit of the harness, not a decode result). Publication latency median 3.01 s (reference 2.84 s), max 5.50 s on the SF11/500k message. That message's first crop (pwr17, `early_214`) replays in 3.1 s on old and took 8.1 s with three duplicate emissions on 2b7d58f:

3. **Alias-merge window was too wide.** The crop's finder reported the same preamble at starts 0, +0.95, +1.93 and +4.5 symbols. The old code tried each as its own candidate and decoded on the +1.93-symbol lock's first sweep; with the merge window at two symbols that lock was absorbed into the winner, whose attempt and recenter both read a valid header and never CRC'd — the pre-existing "confident demod, skip sweep" rule fired there (the fractional timing estimate rounded to −0.44 bins instead of old's −0.50, so the frac-park check that disarms the skip did not run), and without the redundant alias the frame surfaced only through Pass 2. Fix: aliases merge within ONE symbol (the padding's grid aliases); a lock a symbol or more later stays a candidate. 128: 8.1 → 2.3 s, one emission. Tried and rejected: handing a banned group to its next alias in the main ladder (made 052 and 172 slower than old and did not fix 128 — the second alias hits the same skip), and lifting the runner-up ratio gate (no decode).
4. **Truncated recenter attempts left their span unmasked.** A CRC-absent fluke header (PL=203 on a 149-symbol junk SF12/250k capture, corpus 075) decoded to garbage in the old code's recenter rung, returned OK and masked its span; the truncated FAIL returned nothing to the caller, so the junk preamble survived for two more attempts and their recenters (10.6 → 14.1 s). The rung now hands the truncated span back and the caller masks it exactly as the main ladder does (075: 9.9 s).

## Results (final build, 2026-10-31, serial, same machine, old main 921afce vs candidate)

- Corpus 197: nothing lost; 101, 125, 127, 140, 147 gained (188 vs 183 recordings with a packet); total wall 471.1 → 280.3 s (−40.5 %); median per-recording ratio 0.84; p95 10.45 → 2.21 s; max 34.3 → 13.3 s; 25 recordings faster by more than 2 s; no recording slower by both >1.25x and >0.5 s (095, a junk capture, +0.4 s); duplicate-emission recordings 4 → 1.
- 80 impairments: 79/80 vs 78/80 targets, nothing lost, wall −14 %.
- 72 SF12/62.5k live crops (both live runs, `sf12_62_list.txt`): 54 → 68 crops decode a packet, nothing lost, wall 1049.6 → 384.9 s (−63 %), no crop slower by the bar; the one unverified record is the same CRC-16 collision old emits.
- The five junk SF12/500k early crops of live run 2: both versions decode nothing; wall 25.2 → 12.9 s; one crop +0.85 s (2.1 → 3.0 s: the cheap truncated exits let more Pass-2 rungs run before the budget ends).
- Unit tests: 914 passed + 16 subtests.

## Live run 3 (a1dc44a, final build)

`live_validation/ladder_live2_a1dc44a`, same 13-message matrix through `prepare_live.py` + `launch.py`: 13/13 recovered, `all_messages_verified`, `live_metrics --require-complete` **passes with no outstanding reason** (the first build to do so since d3eb682), drain clean, Heltec restored, zero ring drops, no timeouts. Publication latency median 2.84 s (reference 2.84 s), max 3.51 s (reference 3.61 s), none over 5 s. The message-9 first crop decoded on its early crop as in the reference.

Independent received-byte CRC audit: 20 of 22 unique outputs pass; the two that do not are `unknown`-protocol structural candidates (LoRaWAN-shaped, CRC absent) from two weak off-frequency junk crops at 916.6 MHz and 912.6 MHz — an external SF9/500k source that was active only during this run (61 gate firings, none in the reference or in run 2) and cost 625 s of decode time across its crops. The old decoder emits nothing on crop 175 (5.9 s) where this build emits one such candidate (2.2 s): the emission rule is the old code's own structural LoRaWAN check on a CRC-absent payload, and which garbage grid gets read is the coin flip; across the corpus and the SF12 set the candidate-tier counts are equal (0/0 and 1/1). Not a test-message result; reported, not changed. On the six junk crops of that source replayed offline, old 75.9 s → new 65.9 s in total, with one crop slower (032: 10.1 → 16.9 s — a fluke PL=10 header on a different junk grid armed the carrier rescue, the same rule as old's; the five header grid descents on that crop were all rejected) and one faster by 17 s. Weak external junk is where fluke headers decide the ladder's expensive branches in both versions; left alone.

## Where the remaining time goes, and what was tried after the final build

Live run 3 decode time, 1055 s: 380 s productive (of which 306 s are redundant re-decodes of already-verified test messages: each transmission is captured by two detections (`LORA_DEDUP_KEEP=2`) × early and full crop × a plain and a relay-isolation job, and Meshtastic retransmits each message three times — 10 to 24 jobs per message); 675 s on crops with no packet at all (the external weak SF9/500k source: 361 s plain jobs + 314 s relay-isolation twin jobs). The reference run had 62 s of no-packet time. Old and new spend the same on a junk crop within noise.

Tried after a1dc44a and rejected with measurements:
- **Header-quality gate for fluke headers.** Per-symbol header peak-to-mean does not separate: attempts on captures that never decode have a median of 14.9 dB and 308 of 356 sit above the weakest real decode (9.1 dB). The failing attempts on junk SF12 crops are real frames whose payload is unrecoverable in that crop, not flukes.
- **Mini sweep (±1 chip × 5 taus) where the confident-demod rule skips the sweep.** Zero decodes attributable across corpus, SF12 set and live crops, and one LOSS: the ten extra trials fed the cross-attempt soft combiner, which then produced a CRC-16 match with two chase flips on a wrong stream (source address one nibble off), the attempt counted as decoded, the span was masked and the real frame never got its retry (SF12 crop 001). Reverted. Without those trials the soft combiner's matches were all verified packets (corpus 8, old 9; SF12 old 6).
- Curved-drift retries stay: they decode 9 (new) / 18 (old) SF12 crops and 9 impairment targets (old).

Next levers are pipeline policy, not decoder code, and need a decision on latency trade-offs: defer the relay-isolation twin job until the plain job of the same crop has verified a packet (saves all of its no-packet time, 314 s here, 16 s in the reference; delays relay-hop decodes by one job); skip the full crop of a detection whose early crop already verified the frame; and bound the number of crops per transmission once one has verified.

## Decode pool: the time-isolated twin waits for the plain job (19a846c, 793418d)

The first pipeline lever above, implemented in `BackgroundDecoder`: a PASS-2 time-isolated twin (fast SF only) is held until the plain job on its file has left the system; it is skipped when that job decoded a CRC-verified frame (admitted or a publication-level duplicate — the first version counted only admitted packets and released 36 of 40 twins), released to its original tier otherwise. Every plain exit path reports; a REQUEUE-SLOW retry keeps the hold; a 180 s stale valve and `drain()` release anything left; `pending()` counts holds. One-sided relay searches are unchanged. `tests/test_iso_hold.py` (8 tests).

Live run 4 (19a846c): 13/13 verified, 40 twins held, 4 skipped, 36 released (the duplicate gap). Live run 5 (793418d): 13/13 verified, `live_metrics --require-complete` passes, publication latency median 2.88 s (reference 2.84 s), max 4.27 s; 65 twins held, 36 skipped, 29 released — 25 on junk files with no packet in either twin (the external SF9/500k source again, 616 s of its 1251 s) and 4 whose packet came from the released twin itself. Twin time on files with a packet: 52 s in run 3 → 24 s in run 5. Both runs' independent CRC audits pass for every Meshtastic output; the unverified records are again the structural `unknown`/LoRaWAN-shaped candidates on the external junk (4 in run 5).

Acquisition outlier, recurring: run 4's message 3 published at 8.0 s because its first crop was *submitted* 7.9 s after RF end (decode 0.1 s) — the same acquisition-side delay Codex recorded (7.77 s) and left unresolved; it occurs in 2 of 5 runs, never in the decoder or the pool.

## Missed first transmissions under interference, and the patience region rule

Across the six live runs (78 messages) the acquisition latency from RF end to decode job is median 2.29 s, p90 2.83 s, with two outliers, both in run 4: messages 3 (7.9 s) and 4 (5.0 s). Header-checkpoint timestamps show the gate never planned a capture for the first transmission of message 3 and planned message 4's 4.84 s after RF end (normal 2.3-2.9 s); both were recovered through the Meshtastic retransmission 5.2 s later. At those moments the external SF9/500k source at 916.3-917.4 MHz had all four patience slots: a carrier was promoted, burned 12 futile windows, was retired with a per-bin cooldown (±29 kHz), and a neighbour 50-100 kHz away was promoted in its place — 155 promotions in that run — while the budget cap fell from 24 to 10 peaks per window with 10-30 peaks shed per window. Run 4 also had the highest skipped-main dechirp count. Codex's 7.77 s outlier (SF12/500k, prefix planned 5.0 s after RF end) is the same acquisition-side class.

Rule added in the patience promotion scan: a candidate bin with two or more futility cooldowns within ±500 kHz (the neighbourhood the scan already uses for "being detected normally") is not promoted — a wandering junk source makes its region junk until those cooldowns lapse. Decode confirms still lift nearby cooldowns; main-path detection is untouched.

Evidence: a fresh 187 s raw 10 Msps recording with the source present and one SF12/62.5k message (`Desktop/LoRa-Raw-Junk-20261031/raw.cs16`), replayed through the live profile (`run_replay.py`, shared capture + header checkpoints, 2+2 workers): HEAD and the rule both detect 10 frames and decode the same 2 packets in 19 events; promotions 28 → 25, futility retires 24 → 21, gate throttle events 38 → 15 (HEAD spent 30 of 38 windows at caps 8-10). The miss itself is not in this recording (one message, detected by both); the rule removes the load that accompanied it. The three earlier clean recordings (`raw-diagnostic`, `raw-sf10-20261029`, `raw-sf12-250-20261029`) replay to identical detections and identical distinct packets under both builds, with promotions 29 → 22, 16 → 14, 25 → 21 and throttle events 1 → 1, 246 → 245, 30 → 16 (`Desktop/LoRa-Raw-Junk-20261031/clean_replays/`).

## Off-grid LoRaWAN candidates and the full-crop hold

**LoRaWAN-shaped candidates off the channel grid.** The structural LoRaWAN gate alone (MHDR RFU/Major bits zero plus a length rule) passes about 8/256 of random first bytes, so a CRC-absent fluke header on a weak junk crop occasionally decodes to garbage that "looks like" LoRaWAN: six such `unknown`/`lorawan`-hint records were published across live runs 3 and 5, all from the external source at 916.6 and 912.6 MHz (SF9/500k, SF7/250k), where no US915 plan has a channel. `parse_lorawan_packet` now returns nothing when the frame is clearly off every grid point for its SF/BW in the configured region (`_on_lorawan_grid` False); unknown region or missing RF context keeps the hint. Replay of the 19 junk crops that carried those records: 0 candidate records under both old main and this build (5 off-grid rejections), and the two crops where the false record used to end the capture early now take old main's time (2.2 → 4.2 s vs old 5.6; 3.9 → 20.9 vs old 21.4). The CRC-policy test fixture moved from 915.0104 to 914.9004 MHz (channel 63) because it exercises dispatch, not the grid.

**Full crop waits for its early sibling.** The header checkpoint exports an early crop when a frame's header reads and the full crop of the same detection when the frame completes; both were decoded. The early view records its parent view number, so both crops now carry that frame key into `BackgroundDecoder.submit`: a `shared_full` plain job whose early sibling is still open is held and skipped once the early job decoded a CRC-verified frame (its own file then counts as verified so its isolation twin skips too); released otherwise; a full crop with no early sibling known at submit time runs at once, so a header abstain or a fast-SF frame whose full crop lands first is unchanged. Same stale valve and drain as the twin hold; `pending()` counts the holds. `tests/test_iso_hold.py` adds six cases. Wideband replays through the live profile, both identical packets to the previous build: junk recording jobs 19 → 10 and decode time 36 → 13 s (9 full crops skipped); SF12/250k recording jobs 53 → 39 and 354 → 307 s (14 skipped, 1 held then skipped).

Still open: the Meshtastic encrypted-only candidate from a sweep CRC-16 collision (one in ~400 captures; old main emits it too). The sweep-derived encrypted-only emission exists to surface PKI direct messages the sweep recovers, and the collision cannot be told apart without channel-key bookkeeping the decoder does not have; left as is.

## Small-host emulation (4 cores at Pi-class clocks, 2026-10-31 evening)

Every A/B above ran on the 23-core laptop. To check the branch on the class of host it will also run on (Raspberry Pi 5, 4 cores, HackRF at 5 Msps), the same recordings were replayed through old main (921afce, the frozen `ab_validation_01_old` copy) and the branch head on an emulated small host, since no Pi was reachable: each arm pinned with `taskset` to four E-cores capped at 1.6 GHz (first pass 2.0 GHz) via `scaling_max_freq`, every interpreter in the pipeline told `os.cpu_count()==4` through a `sitecustomize` on `PYTHONPATH` so the gate takes the ≤4-core pyfftw path, the serial per-peak path, the bounded slow scan and the Pi's auto pool sizes (2–3 detect workers, 3 decode workers sharing the cores), deployed defaults otherwise (10 s budget, patience on). Three arms ran concurrently on disjoint core sets: old main plain capture, branch plain capture (what `run/web.py` deploys — the shared recorder and header checkpoints are env-only), branch with `LORA_SHARED_CAPTURE=1 LORA_HEADER_CHECKPOINT=1` (the live-validation mode). Recordings: the four raw captures (SF12/62.5k long message with the external junk source, SF12/62.5k clean, SF10/250k, SF12/250k), at their native 10 Msps and decimated to 5 Msps (127-tap half-band, time base kept). Harness, per-run outputs and the host sampler are under `Desktop/LoRa-PiEmul-20261031/` (`run_replay_emul.py`, `run_arm.sh`, `replay_summary.py`, `pass_totals.py`, `host_health.csv`).

Caveat on the first 10 Msps pass: the laptop's package reached 99–105 °C from 17:28:51 to 17:34:37 (concurrent start-up hashing on the P-cores plus a decimation job) and PROCHOT dropped every core to 400 MHz in 15 s bursts, so that pass is kept only as a stress sample. The 5 Msps pass and the clean 10 Msps pass ran with the harness pinned to frequency-capped P-cores: package 51–61 °C, PROCHOT counter flat.

**5 Msps (the Pi's recommended rate), four recordings, whole-pipeline totals:**

| arm | unique packets | transmitted messages decoded | decode jobs | pipeline CPU s | sample-loss lines | aborts |
|---|---|---|---|---|---|---|
| old main, plain | 2 | 2/4 | 194 | 3456 | 9 | 0 |
| branch, plain | 6 | 4/4 | 30 | 2477 | 3 | 0 |
| branch, shared + checkpoints | 6 | 4/4 | 40 | 2496 | 0 | 0 |

Publication latency after transmission, per message (old / branch plain / branch shared): SF10/250k 16.1 / 15.2 / 9.7 s; SF12/250k 64.9 / 49.1 / 15.8 s; SF12/62.5k clean — / 46.9 / 27.8 s; SF12/62.5k with the junk source — / 82.1 / 37.0 s.

What the emulation showed:

1. **Old main never detects SF12/62.5k frames.** Its gate logged no preamble for either SF12/62.5k recording at 5 or 10 Msps. This is not an emulation artefact: old main's own live validation on this laptop (`ab_validation_01_old`, 2026-10-29) recovered 6 of 10 messages and created no job for any of its four SF12/62.5k transmissions. The branch's slow-pass detection (Codex's 9045c40, 3b4f79d, 8f3dd51, 078d076) finds them on the small host too (`DETECTED ... [slow-pass]`).
2. **Old main thrashes on the external junk source at 5 Msps:** 73 submissions and 135 decode jobs of alias-labelled ghosts (SF9/125k and SF11/250k at 914.2–915.5 MHz), decode queue up to 62, six `[NB-CROP] queue full — early finalize` sample losses and one `SLOW-GAP`, 1217 CPU s for zero packets. The branch: 10 submissions, 11 jobs, no loss, and the message.
3. **28 % less pipeline CPU** for the branch over the four recordings while decoding three times the packets; decode jobs 194 → 30.
4. **Latency is lower on every message both decode**, and plain-mode slow-SF latency on this host is dominated by the fixed worst-case capture window (32 s for SF12/62.5k, 3.6 s for SF10/250k) plus decode time on a 1.6 GHz core; the shared-capture/header-checkpoint mode roughly halves it (early crop decoded as soon as the header reads).
5. **Observed, not changed:** (a) same-frame sibling captures (keep-2 copies and alias peaks 66–115 kHz apart) are decoded concurrently on the branch where old main's sibling-wait serialised them — the bucket skip does not match them because their carrier estimates differ by more than the bucket, so after an SF10 frame three cores were busy for ~40 s re-decoding an already-published frame (same total CPU as old main, which decoded them one after another); (b) the branch's plain-mode SF10/250k crops are 3647 ms against old main's 2025 ms (long-frame retention), so each of those redundant decodes costs about 1.6× more; (c) in shared-capture mode the pipeline aborted once, in the PROCHOT-perturbed 10 Msps pass, with `[SHARED-CAPTURE FATAL] raw history/crop interval ... ring history unavailable or overwritten` (capture debt 10.5 s against the 16 s ring) — Codex's fail-fast in `capture_timeline.py`, already present in four of their earlier laptop logs; the web supervisor restarts the detector. Not seen at 5 Msps (max debt 7.9 s). On ≤4-core hosts the deployed default (plain capture, 5 Msps) avoids it.
6. **Decoder alone on one capped E-core** (10-recording corpus subset, serial): old 156.6 s → branch 82.4 s (−47 %), the same three gains (101, 140, 147), nothing lost. Recording 010 is +1.6 s (4.10 → 5.68 s; +0.4 s on a P-core): the alias merge keeps bin 312 whose finder-grid header margin is 1.39× below the absorbed bin 0, and the runner-up rung recovers the frame one attempt later. The full 197-recording corpus on the P-core has no recording more than 1.25× slower.

**Clean 10 Msps pass** (same cores at 1.6 GHz, harness pinned, no throttle events — twice the Pi's recommended rate, so every gate ran at its budget cap):

| arm | unique packets | transmitted messages decoded | decode jobs | pipeline CPU s | sample-loss lines | aborts |
|---|---|---|---|---|---|---|
| old main, plain | 3 | 2/4 | 66 | 3047 | 6 | 0 |
| branch, plain | 7 | 4/4 | 28 | 2712 | 6 | 0 |
| branch, shared + checkpoints | 5 | 3/4 | 18 | 2001 (one recording cut short) | 0 | 1 |

Per message (old / branch plain / branch shared): SF10/250k 31.4 / 31.4 / 23.9 s; SF12/250k 126.1 / 57.7 / aborted; SF12/62.5k clean — / 51.0 / 21.4 s; SF12/62.5k with the junk source — / 94.5 / 29.5 s. The shared-capture abort on the SF12/250k recording reproduced without any throttling (2 of 2 at 10 Msps, 0 of 2 at 5 Msps): while the SF12/250k frame is on air the capture writer on this host falls 11 s behind real time (`debt=11.016s`), the 16 s ring laps the raw interval it still has to crop, and Codex's writer latched the whole service fatal. That is fixed below.

**Ring lap is no longer fatal (capture_timeline.py).** The writer's history read distinguishes two failures: a lost stream identity (drops, retune — `'stream skipped or retuned'`) and a lapped ring (`'ring history unavailable or overwritten'`). The first still fails the service. For the second the writer now seals only the band whose interval was overwritten, ends the views still collecting from it the way a retune already did (an optional early/header view is cancelled; a full view keeps the samples produced before the lap as a `shared_partial` save with `metadata['interruption']='ring_lap'`, or is cancelled when there are none), reports one `[SHARED-CAPTURE] raw history lapped ...` notice, counts `lapped_bands` in `stats()`, and keeps every other band running; a sealed band is never reused by a later registration (it would never produce again and would trip the writer-stall fatal 30 s later). Nothing after the gap is ever spliced onto a capture, which was the property Codex's fatal protected. Tests: `test_lost_history_never_splices_future_samples` (drop and retune stay fatal; a lap yields one partial of pre-lap samples) and `test_ring_lap_drops_only_the_lapped_capture_and_keeps_the_service_alive` (two collecting views on the lapped band become partials, the service admits and completes a new capture afterwards). Reproducer after the fix (SF12/250k recording, 10 Msps, shared mode, same emulated host): no abort, five laps each costing one partial capture, both packets decoded, the message published 25.3 s after transmission (old main 126.1 s, branch plain 57.7 s), zero sample-loss lines; `sf12_250_new_shared_10m_lapfix/` under the emulation directory. Full suite 933 tests.

**Same-frame copies after a duplicate (detector.py, `BackgroundDecoder._compact`).** The emulation's SF10 timelines showed the sibling wait doing its job (the second-window copy of a transmission requeued behind the first) and the copy then decoding anyway once the first finished, 30 s on the 1.6 GHz core, because the first's result was a `[DUP]` of a frame an earlier transmission had already published and only an *admitted* publication marked the RF bucket as decoded. A CRC-verified decode now marks its bucket before the publication dedup, on both the packet-ID and the payload-fingerprint paths, exactly as `_note_verified` already did for the twin hold. Copies of the same instant (carrier within the per-frame tolerance, start within the per-SF window) skip at dispatch; a retransmission seconds later is its own bucket and still decodes. Test: `test_a_duplicate_of_a_published_frame_still_covers_its_own_bucket`. Emulated 4-core replays at 5 Msps (plain capture, same packets): SF10 recording 5 → 3 decode jobs, SF12/250k recording 10 → 7, the dropped jobs being second-window copies skipped while waiting on their verified twin (`SKIP-INFLIGHT-ADMIT`). Commit 404ca1d.

**Alias choice (recording 010) left as is.** The runner-up rung fires in 5 of 197 corpus recordings: it decodes 010 and 147 (one of the gained recordings) and finds nothing in 097, 168 and 172, each of which is still faster than old main. Choosing the alias with the higher finder-grid header margin instead would fix 010 (alias 1.39× the winner decodes) and break 057 (alias 1.37× the winner fails, the winner decodes directly in 1.16 s). The two cases are not separable on that ratio, and the alias that decodes in 010 also needs the fractional park and sweep, so a cheap first pass of the runner-up before the winner's sweep would not help either. The finder's winner stays first; the runner-up stays a rescue.

**Plain-mode crop length left as is.** The branch's plain recorder keeps the worst-case frame duration for the SF/BW (3.6 s for SF10/250k against old main's 2.0 s) so long frames are not truncated. The cost fell on redundant decodes of the same frame, which the bucket fix above removes; the header-checkpoint mode cuts the crop to the header's declared length.

## Live run 8 (404ca1d) and the acquisition miss, investigated to the end of the evidence

Live run 8 (bucket fix + ring-lap fix, Codex's launcher, 13 messages): 13/13 recovered, CRC audit 20/20, `all_messages_verified`, 127 decode jobs (reference 243, run 7 166), 2880 CPU s against the reference's 3752, publication median 2.91 s. Acceptance failed on one item only: message 4 (the third SF7/125k CR4/8 message) published 13.8 s after transmission, so "first transmission versus retry attribution remains unresolved". The Heltec's firmware log shows three transmissions of that message, at +0.21, +5.56 and +11.01 s, each 0.19 s long; the gate captured only the third. The same signature appeared in run 4 (first transmissions of its second and third SF7 messages missed, retries 5.4 s later caught) and in Codex's reference (one 7.77 s outlier): always an SF7/125k message that follows other SF7/125k messages at the same carrier, always the first transmission, never anything logged.

What was ruled out for message 4, from the run's own records: the decode pool (no skip, hold or bucket decision touched that window; the first capture of the message is the third transmission); the gate budget (no `GATE-BUDGET`, shedding or `TRIAL-DEFER` line in the first 243 s of the run); the pool's emergency window skip (needs an emergency level that was never raised); the prescreen (never learned anything near 915 MHz); the patience cooldown (only gates promotion); the channelizer (its first live apply came seven minutes later); AGC (off); receiver gain (fixed); stream integrity (native telemetry: no overflow, no timeout, no timestamp anomaly); the pipeline's own health (backlog 0.02 s, pipe 4 ms, 10.0 Msps, no drops at every 4 s sample across the window); the host (same package temperature and throttle rate before, during and after). Between message 3's last capture and message 4's late capture the gate wrote 43 lines, all about message 3's decodes.

Offline reproduction attempt: `Desktop/LoRa-Raw-SF7x3-20261031/` records 231 s of raw IQ while the Heltec sends three SF7/125k CR4/8 messages at the matrix cadence (`run_raw3.py`, firmware transmit offsets in `finished.json`); replayed through the branch at full speed in plain and in shared-capture mode, all nine transmissions are detected and saved (`tx_check.py replay_plain replay_shared`). The live miss therefore does not come from the sample processing on its own; it is an intermittent live-path event (about 4 of 45 SF7 first-or-retry transmissions across today's eight runs) that leaves no trace because the gate logs nothing for a strong peak it fails to confirm.

**Diagnostic added (detector.py, `_note_unconfirmed`).** When a gate peak at least 30 dB over the window's PSD floor (ambient spurs on these recordings sit at 20–25 dB, the missed bursts at about 50 dB) reaches the preamble detector and no bandwidth class confirms it, the detector logs once per 20 kHz carrier bucket per 30 s: `[GATE-UNCONFIRMED] peak 915.3003MHz +40dB over floor, width 51 bins: SC best lag=256 score=0.26 < 0.55` (or "no bandwidth class reached the SC stage" when the width gates skipped every class). Debug level 1 and above, which the web deployment uses; no decision changes. Tests in `test_dechirp_coverage_dsp.py`: a strong band-limited noise peak is reported exactly once and then rate-limited, a weak one and debug 0 are silent, a real SF7 preamble is never reported. Volume check on the two raw recordings replayed at full speed: with the bar at 20 dB the three-message recording produced 8 lines, all ambient spurs at exactly +20 dB (912.2, 919.4, 911.0, 911.6 MHz, width 13 bins, SC 0.09–0.14) and the junk recording none; at the committed 30 dB bar neither recording produces a line, and all nine transmissions are still detected. Commit 55a60ee.

## Two mechanisms rewritten rather than patched

**Sibling wait → sibling park (detector.py, `BackgroundDecoder`).** A capture whose RF bucket was already decoding used to be dispatched to a worker that slept in a 0.1 s loop for up to 2 s, then put the job back on the queue and took the next one, repeating until its twin finished or a per-SF deadline passed. On a three-worker host that is one worker spent waiting while its twin decodes; the emulation showed 30 such requeues on the junk recording and 63 in old main. The job is now parked off the worker, keyed by the in-flight bucket it waits on, and the in-flight job's exit releases it: skipped when the bucket verified (`SKIP-PARKED`), otherwise re-queued at its own priority or to the slow tier it came from (`RELEASE-PARKED`). Its deadline (unchanged formula) is enforced by a flush in the dispatcher loop; a job released by deadline runs even if its twin is still in flight, as before, and the drain releases every parked job. `pending()` counts parked jobs. Tests: `tests/test_sibling_park.py`. Replays with the same packets everywhere: emulated 4-core SF10 recording, the two second-window copies are parked and skipped (`PARK-SIBLING` 2 → `SKIP-PARKED` 2, 3 jobs as before, no worker time waiting); emulated junk recording, `REQUEUE-SIBLING` 30 → 0, three copies parked and released when their junk twins failed, 7 jobs; SF12/250k recording neutral (no sibling in flight at dispatch time); the three-message recording at full speed, 9 of 9 transmissions, 10 jobs as before, one copy parked and skipped. Like for like on the junk recording (previous build and this one, same uncapped cores): 7 jobs each, 481 against 482 CPU s, identical publication times, 21 requeues replaced by 3 parks. The gain is not CPU (a sleeping worker burns none) but worker availability: a parked job no longer occupies one of a small host's two or three decode slots while its twin runs, which is what matters when the queue is deep.

**Shared-capture budget exhaustion → non-admission (capture_timeline.py).** A registration that did not fit the byte or outbox budget failed the whole service (Codex's logs show `capture admission failed: band_retention growth exceeds capture byte budget` twice, each a receiver restart). The limit is a resource bound, not an invariant, so that registration is now refused like the existing out-of-band and retune-boundary cases: counted in `non_admitted`, reported once, every other view and band untouched. The two tests that asserted the fatal (`test_full_outbox_slot_limit_is_reserved_before_capture_completion`, `test_retention_capacity_failure_is_explicit_main_loop_state`) now assert the refusal, that nothing leaked from the budget, and that the admitted capture still completes.

## Frame-length-exact capture is the default (20ea5ff)

Plain capture records a frame for the worst-case duration of its SF/BW and decode cost scales with the crop; the shared recorder with header checkpoints ends the capture where the header says the frame ends. It was env-only. On the emulated 4-core host it halved slow-SF publication latency (SF12/250k 49 → 16 s, SF12/62.5k 47 → 28 s) with fewer decode jobs, and its two fatal paths are degradations now. `shared_capture_plan()` (detector.py) decides the mode once at start and the recorder logs it as a `[CAPTURE]` line:

| `LORA_SHARED_CAPTURE` | result |
|---|---|
| unset | exact capture when MemAvailable ≥ 2 × budget + 2 GiB, budget = the validated 2048 MiB plan + header workspace (about 2.8 GiB), header checkpoints on unless `LORA_HEADER_CHECKPOINT=0`; otherwise plain capture with the reason. A 16 GB Pi 5 or any desktop qualifies; an 8 GB Pi keeps plain capture unless forced. |
| `0` | plain capture (`LORA_HEADER_CHECKPOINT=1` with it is still an error) |
| `1` | the explicit diagnostic form, unchanged: `LORA_SHARED_CAPTURE_MB` sizes the budget, header checkpoints only with `LORA_HEADER_CHECKPOINT=1` and an explicit budget of at least plan + workspace |

Why 2 × budget + 2 GiB: the budget is a cap on charged bytes (peaks in the emulation 1.5–2.2 GiB at 5 Msps, 1.6–3.0 GiB at 10 Msps) and the rest of the pipeline, its decode workers and the web app need their own several GiB; an 8 GB Pi 5 ran plain capture at 6 GB RSS before and froze once, so it is not enabled there by default. In auto mode a source without validated ring history or `LORA_NO_DEFER=1` falls back to plain capture with a `[CAPTURE]` line; the explicit form still refuses. Tests: `tests/test_capture_default.py`.

Validation of the default path (no environment set, so the detector chose the mode itself: `[CAPTURE] frame-exact capture: budget 2816 MiB of 50107 MiB available, header checkpoints on`): emulated 4-core host at 5 Msps, same packets as the plain arm on the same cores, message published 23.5 s after transmission against 36.2 s plain on the junk recording and 4.8 s against 12.1 s on the SF12/250k recording, charged peaks 1.46 and 2.19 GiB under the 2.75 GiB budget, no non-admission, no lap; the three-message recording at full speed, 9 of 9 transmissions. Live run 9 (`ladder_live8_20ea5ff`, Codex's launcher, which sets the explicit form): 13/13 recovered, CRC audit 20/20, `all_messages_verified`, publication median 2.81 s (reference 2.84 s), max 4.17 s, nothing over 5 s, 135 decode jobs (reference 243), 61 full crops skipped for their verified early sibling. The acceptance gate's attribution item stayed open for message 6 only because the Heltec's serial log carried one "Started Tx" and two "Completed sending" lines for that packet; the message itself was published 2.18 s after transmission.

## The gate's cap ranked spur teeth above real preambles (66e3ed2)

What the `[GATE-UNCONFIRMED]` diagnostic showed on its first live outing (run 9, `ladder_live8_20ea5ff`): 3742 lines in 25 minutes, 2271 of them "no bandwidth class reached the SC stage". The carriers sit on a 50 kHz grid offset 37.3 kHz from 915.000 (915.0369, 916.2371, 917.4874 ... MHz), about 90 teeth between 915 and 919 MHz, each 4-5 bins wide (10-12 kHz at 10 Msps) and 40 to 52 dB over the floor, present for the whole run and absent from the raw recordings made the same day with the same radio and gain, so it is an intermittent environmental comb, not the receiver. The SC stage refuses such a peak at every class by design: at 15 dB or more over its floor the width is trusted, and a chirp of bandwidth B cannot measure narrower than B/2.5. But the per-window cap runs first and sheds weakest by raw power, so with cap 12 to 24 (run 9 saw 172 sheds, up to 10 peaks a window) a +50 dB tooth kept its slot and an SF7/125k preamble at +48 dB competed with the teeth for the rest. On a Pi-class host, where the budget clamp holds the cap at 2 to 5, the comb would have taken every slot.

The fix moves the width rule in front of the cap through one shared helper (`_sc_width_admits`; the SC stage's inline test now calls it too, so the two sites cannot drift): a peak no class at or above 31.25 kHz admits leaves before the sort. The slow-pass, prescreen and patience snapshots are taken before that point and still see every peak; channelizer and patience peaks are forced after the cap as before. Drops are counted and reported once a minute at debug 1 as `[GATE] dropped N spur-width peak(s) over M window(s) before the cap`. Tests in `tests/test_gate_doomed.py` execute the real block from `main()`: the comb plus one real preamble at cap 12 keeps only the preamble, the same window without the rule would have shed it, LoRa-shaped peaks still shed weakest-first, weak narrow peaks are left to the SC stage, and the gate drops exactly the peaks the SC stage evaluates no class for.

Run 8's message 4 miss re-read against this: its window (227.5 to 232.5 s) carries no shed, throttle or budget line and 11 ms of gate time, so capacity was not that miss's cause; no PREAMBLE line exists for the first transmission at all, which puts it upstream of decode. Still open.

Replays through the new build (default capture mode): three-message recording 9 of 9, identical detections and saves to the previous build, 1 shed as before, 71 to 82 spur-width peaks dropped a minute (the recording's ambient narrow spurs); junk recording the same 10 detections and 2 packets, 2 sheds against 2, 73 to 115 dropped a minute. 959 tests.

## Gate decisions as data, and a capture plan that fits the host (67d565e)

Live run 10 (`ladder_live11_66e3ed2`, 2026-11-02) was the first in the shipped default capture mode, with a harness profile (`auto-default-workers2`) that proves no capture variable reached the detector and that the three startup receipt lines agree with each other. It got through 6 of 13 messages before the mains-power guard stopped it (the charger connection flapped at t+490 s); all six were recovered on their first transmission, publication 1.49 to 3.85 s, median 2.73 s, CRC audit 9 of 9. The host ran at a package temperature of 97 °C median with a sampled clock median of 0.8 GHz, which is the laptop's thermal limit and not the pipeline's.

**The plan was too small for this host.** During the SF12/62.5k long-frame phase, with three bands and ten views in flight, charged bytes peaked at 2.90 GB against the 2.95 GB default budget and four captures were refused (`band_retention growth exceeds capture byte budget`), on a machine with 57 GB available. Retention scales with sample rate and concurrent frames, so the plan now scales with the memory that is there: MemAvailable/8 clamped to [2048 MiB, 8 GiB], plus the header workspace. A 16 GB Pi 5 lands exactly on the validated floor (its behaviour is unchanged); this host gets 7.9 GiB. The explicit form (`LORA_SHARED_CAPTURE=1`) still takes its budget from the environment only. Tests in `tests/test_capture_default.py`.

**Gate decisions as data** (`src/gate_ledger.py`). The missed first transmissions of 2026-10-31 could not be explained because the gate's decisions existed only as rate-limited stderr lines. Every window now leaves one JSON record: `t0`/`t1` (the span, so a firmware timestamp finds its window), `floor`, the counts `scan`, `prescreen`, `doomed`, `shed`, `forced`, the effective `cap`, the `route` (inline, pool, skipped, quiet) and, for each peak 20 dB or more over the floor, its `fate` and the detect stage's verdict: `r` in detected, slow-wide, no-crop, short, no-class, sc-below, unresolved; `sc`/`lag`/`thr` where the SC stage ran; `det` as (sf, bw) pairs. `detect_preamble(outcomes=[])` collects the verdicts at the per-peak exit points; pool workers return them beside the result (`DetectPool.outcomes`), with the emergency cap they applied. The ledger is on at debug 1 (the web deployment's level); `LORA_GATE_LEDGER` names the file and `''` disables it; the web app keeps it in `lora_web/gate_ledger.jsonl`, rotating at 20 MB with one previous file kept. Cost measured on the three-message recording: 461 records for 231 s, about 1 KB each, no change in detections (9 of 9) or packets. Tests in `tests/test_gate_ledger.py` include a whole-program file-mode run that reads its own ledger back.

## Live runs 11 and 12 in the default mode, and what the ledger found (076e1ac, 30094f3)

Run 11 (`ladder_live13_67d565e`) was the first complete ladder in the shipped default capture mode: 13 of 13 recovered on their first transmission, median publication 2.72 s, charged bytes peaked at 3.67 GB with no non-admission under the memory-scaled plan (the old fixed 2.95 GB budget had refused captures at 2.90 GB the run before). It also cost 247 decode jobs and 1575 s of decoder time against run 9's 135 and 357 s, and the independent CRC audit passed 20 of 21 records. Three defects, each with its evidence in the run directory:

**Placeholders on dropped spur bins.** 106 slow-tier jobs were 29.6 s SF12/62.5k full captures of noise at 911.0, 913.0, 916.98, 917.98 MHz and the like (power 1 to 3 dB over the floor), none of which run 9 had. The spur-width drop had removed the one thing that kept the patience and channelizer placeholders off those carriers: a placeholder is planted only where no gate peak sits within 15 bins, and with the spur peak gone a 125 kHz-width placeholder ran the dechirp trial on the CW tone. The bins of a window's dropped peaks now block both placeholders.

**A verified message with no text.** Message 7's CRC-16 coincidence with three flipped bits decrypted to portnum 1 and a damaged payload length; the old rule (known portnum, UTF-8 checked only when a payload exists) called it verified and published a TEXT_MESSAGE with null text beside the real one. `data_message_valid()` now requires the whole decrypted buffer to parse as well-formed Data fields with known numbers and wire types, a known portnum, and for text portnums a non-empty UTF-8 payload (`tests/test_meshtastic_validity.py` replays the exact live case). Twenty other null-text records in the stream were NODEINFO packets, which have no text.

Run 12 (`ladder_live14_076e1ac`) with those two fixes: 13 of 13, CRC audit 20 of 20, `evidence_complete`, median 3.09 s, 200 jobs, 1147 s of decoder time, still 47 junk full captures and a 24 s publication for message 7 while the workers decoded them. The gate ledger placed the first junk capture in a window with no forced placeholder and no dropped peak: an ordinary weak gate peak at 911.123 MHz confirmed by a same-slope SF9/250k dechirp trial (`CHAN-DECHIRP SF9 q=16.3dB`) at the moment a +51 dB SF7/125k frame started at 915.002 MHz. The junk capture's own IQ settles it: a chirp 25 dB below the real frame, time-aligned with it, 3.88 MHz away on the receiver's 1 MHz spur grid, i.e. an intermod copy that keeps the chirp slope. **Intermod-copy rule** (30094f3): the window arbitration, which already drops mirror images and splatter shoulders, now drops a detection that starts when a kept one starts, shares its slope family (bw²/2^sf, so a preset and its aliases) and is at least 20 dB weaker, whichever the quality sort placed first. Measured copies sit 25 to 40 dB down. `tests/test_intermod_copy.py` builds the strong frame and its copy synthetically; a comparable-power second transmitter and a different-slope weak frame survive.

Open from run 12: one decode job logged `ERROR: 'bool' object is not subscriptable` on a junk capture (`SF12_62.50k_918.0012MHz..._089`); the same file decodes cleanly offline, so the path is load- or budget-dependent and is not yet located.

## Live run 13 and the on-air registry (b24ac93)

Run 13 (`ladder_live15_30094f3`, the in-window copy rule): 13 of 13, CRC audit 20 of 20, evidence complete, median publication 2.75 s and a maximum of 3.91 s, the lowest maximum of any run. The copy rule fired 15 times, all on copies starting in the same window as their frame, and the run still produced 65 junk full captures and 84 slow-tier jobs (1393 s of decoder time). Read against the detector's own output, 22 of the 65 coincided with a strong same-slope detection and only 18 of those in the same window; 43 had no strong detection anywhere near them in time, because the frame they copied, an SF12/62.5k long frame, had been detected once at its preamble, by the slow pass at its nominal 20 dB, and then kept leaving copies on the spur carriers for the next 30 s of its body.

`OnAirFrames` (detector.py) remembers every detection at least 30 dB over the window floor for its worst-case duration (the same bound that sizes captures). Every detection path, the pooled commit, the inline detect and the slow pass, first drops a same-slope detection at another carrier that falls inside a remembered frame's span and is at least 20 dB weaker, then registers the strong survivors. A detection's elevation is the larger of its own power and the latest window's PSD at its carrier, so a frame the slow pass reported at 20 dB is remembered at its real strength. Tests in `tests/test_intermod_copy.py` (registry semantics, purge, alias copies, the frame's own carrier exempt, all three paths wired). Replays of the three-message and junk recordings: see the ledger entry.

## Live run 14: the default mode holds (1a514d9)

Run 14 (`ladder_live16_b24ac93`, the on-air registry): 13 of 13 recovered on their first transmission, CRC audit 20 of 20, evidence complete, median publication 2.78 s, mean 2.57 s (the lowest of any run), maximum 4.17 s. 152 decode jobs and 541 s of decoder time against 236 and 1393 s the run before (run 9's figures were 135 and 357 s); the on-air registry dropped 39 copies, all in the pooled commit path and all SF12/62.5k copies of the long frames at 911.99, 913.02, 916.67, 916.99, 917.02 and 917.99 MHz, 1 to 17 dB over the floor under frames at +61 to +63 dB; the in-window rule dropped 10 more. Junk full captures fell from 156 to 8 (three are the SF10/125k frame's own weak late captures at its carrier, 5.8 s each), slow-tier jobs from 84 to 8, and no decoder error occurred. The acceptance gate flagged only message 6's attribution, where the Heltec's serial log carried one "Started Tx" and two "Completed sending" lines, as in run 9. A few of the remaining copies had committed in the same batch as their frame but ahead of it; `OnAirFrames.filter()` (1a514d9) now registers a batch's strong detections before judging any of them.

Where this leaves the day's open item: the acquisition miss of 2026-10-31 did not recur in five ladders today (15 SF7/125k first transmissions, all recovered on the first attempt); the gate ledger is in place for when it does.

## SF7 acquisition stress and the decoder's budget bool (49867ab)

Twenty SF7/125k CR4/8 first transmissions through the live pipeline in the default mode (`Desktop/LoRa-SF7-Stress-20261102/sf7_stress.py`, the ladder arm's frozen app and logged Heltec shim, 52 s cadence, Heltec restored afterwards): 20 of 20 published from their first transmission, firmware TX start to publication 2.58 to 3.38 s, 251 decode jobs for 165 s of decoder time, 8 slow-tier jobs, 17 copies dropped by the on-air rule, no non-admission, no decoder error. With the day's five ladders that is 35 SF7 first transmissions recovered first-time; the 2026-10-31 miss has not recurred since the gate changes, and the ledger (2347 windows recorded in this run) is in place if it does. `analyze_stress.py RUNDIR` prints the per-transmission table and, for a miss, the ledger windows around its firmware timestamp.

The decoder error seen once each in runs 12 and 13 is located and fixed: `_recenter_and_decode` returned `False` when the CPU budget ran out between recenter rungs, and its caller indexed the result as a span tuple. Under the pool's small slow-tier budgets that aborted the job; in run 12 the job was message 7's early capture, which is why that message waited 24 s for its full capture. It returns None now, and decoder tracebacks also go to stdout so the raw job log keeps them (the pool discards worker stderr).

## Early-expect hold and the merge gate (b52d20e)

The 5 Msps four-core emulation of the SF10/250k recording was the one case where the
branch published later than old main (24.2 s default / 21.1 s plain against 14.3 s).
The decode pool's FULL-HOLD only knew about an early crop once that crop was enqueued;
on a slow host the full crop arrived first, was decoded in full, and the early crop then
found a verified sibling and was skipped. `HeaderCaptureController` now reports
`on_early_planned(frame)` the moment an early crop is planned, `DecodePool.expect_early`
records it, and a full crop of that frame waits (bounded by `_EARLY_EXPECT_S = 10 s`, and
released by the stale valve if the early crop never comes). Re-emulated with the same
caps: default arm 13.2 s (old main 14.3 s). Four tests in `tests/test_iso_hold.py`.

The merge gate itself (evidence under `Desktop/LoRa-Merge-Gate-20261102/`,
`merge_gate_report.txt` assembled by `report.py`), old main 921afce against HEAD:

- Corpus, 197 stored recordings, serial, alternating order: decode wall
  min/median/mean/p95/max old 0.81/1.37/2.41/10.91/33.91 s, new 0.67/1.13/1.45/3.59/13.66 s;
  total 476 -> 286 s (-40 %); recordings with a packet 183 -> 188; no lost output, five
  gained; nothing slower by more than 1.25x and 0.5 s (two cases slower by ~0.4 s).
- Impairments, 80 stored noise/drift captures: targets 78/80 -> 79/80, no lost output,
  wall -14.6 %.
- Emulated four-core Pi at 1.6 GHz, 5 and 10 Msps: old main missed the junk and diag
  recordings at both rates and published SF12/250k at 77 s and 128 s; the branch's
  default mode published every message (5 Msps 13.2-42.5 s, 10 Msps 19.8-46.9 s) with
  zero sample loss. The branch's *plain* mode on that host is the one place a message
  publishes later than old main: SF10/250k 21-28 s against 14.3 s. In isolation on one
  1.6 GHz core the new decoder takes 2.9 s on the old 2025 ms plain capture and 3.5 s on
  the new 3647 ms one (the worst-case frame bound from ebdbccd), so the gap is
  scheduling contention when three copies of the frame decode together on four slow
  cores, not decoder cost; plain mode is only chosen when the host cannot carry the
  frame-exact budget.
- Live, same evening, USRP + Heltec, 13-message ladder, two detector workers each:
  old main decoded 10/13 and never saw the three SF12/62.5k long frames (its captures
  of them were 4 s long); RF-end to decode-job 3.58/5.34/13.53 s min/median/max, 219
  jobs totalling 1048 s, 13 worker errors, 51 crop-queue-full, dec_q up to 58, RSS up to
  25.7 GiB. HEAD decoded 13/13, every message faster than old main's on the same
  message: 1.58/2.74/3.65 s (publication 1.27/2.73/3.47 s), 114 jobs totalling 307 s,
  no worker errors, dec_q max 5, RSS max 9.9 GiB, CRC audit 20/20.

## Round 2: the decode pool stops decoding the same frame twice (b11eecb .. fb3a2a8)

After v3.0 the last live ladder still spent 307 s of decode on 13 messages:
114 jobs, of which 83 published a packet the pool had already published and
two spent 52 s and 44 s on one SF12/62.5k frame whose payload never passed
its CRC.  Five pool changes, each from a measured pattern:

- **A failed early crop retires its full sibling (b11eecb).**  Across every
  live run since 2026-10-31, 21 full crops were released after their early
  crop had read the header and failed every retry; none recovered a packet;
  686 s of decode.  The early crop is planned from that header to end at the
  frame's end, so its full sibling holds the same samples.  The pool now
  classifies the finished early job (header read, no packet, not a budget
  bail, timeout or error) and skips the full crop; budget bails, timeouts and
  header-less early crops still release it.
- **Early crops own their RF bucket (e754d13).**  In frame-exact capture every
  capture is an early or shared view, and both roles were exempt from the RF
  bucket dedup.  An early crop is one frame, so the exemption only cost: the
  gate labels one burst at carriers a few hundred Hz apart, two header
  checkpoints follow, and both early crops were decoded (78 such jobs in one
  ladder).  Early crops now park behind a same-frame sibling in flight and skip
  once it verified; shared views are unchanged.
- **A skipped early crop retires its held full (b7a90b2)**, found in the first
  emulation of e754d13: PARK then SKIP-PARKED early, then a 4 s FULL-RELEASE
  of the same frame.
- **An early crop's bucket uses the per-frame airtime window (f3c4812)**, never
  the 1.5 s forced-dechirp cooldown, so a different packet a second later on a
  channelised carrier can never be taken for a copy.
- **A clean CRC after a failed first result is the verdict (fb3a2a8)**, found
  in the ladder of f3c4812: message 12's early crop failed its first read,
  the recentre pass decoded it clean and published it, but the admission
  parser kept the first CRC line, skipped `_note_verified`, and the pool
  released the held full crop and a parked sibling (two per ladder, 3-8 s).

Validation (evidence under `Desktop/LoRa-PiEmul-20261031/*_dd3`, `*_10m_pi2`,
`*_5m_pi2`, `sf10_*_5m_rep[123]`, live arms `ladder_live22_dedup4`,
`ladder_live23_main`, `ladder_live24_dedup5`; `report_dedup.py` in the merge
gate directory assembles it).  decoder.py is untouched, so the 197-recording
corpus and the 80 impairment captures are identical by construction; the
pool is exercised by replays and live runs:

- Uncapped replay of the four wideband recordings, main vs f3c4812, two arms
  concurrently: identical packet sets, identical publication latencies to
  0.1 s, 44 -> 40 jobs.
- Emulated four-core Pi at 1.6 GHz, three arms concurrently: at 10 Msps the
  branch publishes every recording sooner than main (35.5/25.6/27.3/48.6 s
  against 40.5/26.2/28.7/51.0 s), identical packets, zero loss, 1028 -> 821 s
  of decode.  At 5 Msps three recordings are within a second and SF10/250k
  looked 6 s slower in one run; three paired repeats gave 13.3/20.2/13.4 s
  for the branch against 19.2/21.0/23.2 s for main, so that was host noise.
- A first five-arm pass (two uncapped and three capped at once) overloaded
  the laptop: the branch's capped SF12/250k arm died on a 12 s catch-up
  discontinuity and its uncapped SF10 arm stalled 12 s at start-up and never
  saw the first transmission.  Per-window detector profiling was identical
  across arms, and every clean rerun above was unaffected; emulation arms
  are run one set at a time from now on.
- Live, same evening, USRP + Heltec, two detector workers: main 07d4655
  decoded 13/13 with RF-end-to-decode 1.69/3.03/52.03 s min/median/max (its
  first transmission of message 7 was never detected and the firmware's own
  retransmission 46 s later was decoded; an acquisition miss of the kind
  seen before, unrelated to the pool), 102 jobs, 279 s, RSS 10.9 GiB.  The
  branch at f3c4812 decoded 13/13 at 1.61/2.62/6.37 s, 63 jobs, 246 s, RSS
  7.7 GiB; message 12 at 6.37 s was the parser defect fixed in fb3a2a8.
- The fb3a2a8 ladder decoded 13/13 but taught two more things.  Message 6
  took 115 s: the gate saw nothing during its first transmission and the
  firmware's retransmission was decoded, the same acquisition miss main had
  on message 7 above (one per ladder tonight, either arm; next target, with
  the gate ledger enabled in live runs).  Message 11 took 11 s and that one
  was the parking rule: a 3 dB receiver image of the frame arrived first and
  took 16 s to fail while the 64 dB real copy waited behind it (7 of 31 parks
  that run put a clearly stronger crop behind a weaker one), and three early
  crops of one corrupted frame each burned ~30 s because the parked copies
  were released when the first failed.
- **bcb42eb**: the bucket key carries the detection power and a sibling more
  than 6 dB stronger than the job in flight decodes alongside it; an early
  crop that exhausted its retries marks its bucket so parked copies skip and
  their fulls retire.  Validation of bcb42eb: uncapped replay identical
  packets, 50 -> 33 jobs, 421 -> 266 s, latencies within 0.4 s; capped
  10 Msps identical packets, zero loss, latencies within the capped host's
  run-to-run spread (main alone gave 40.5/46.0/61.1 s on the junk recording
  across three runs); live ladder 13/13, every message from its first
  transmission, 1.66/2.85/5.19 s min/median/max, 64 jobs totalling 120 s
  against main's 102 jobs and 279 s the same evening, RSS 7.6 GiB, zero
  full-crop releases, zero parked releases.

## Ledger

2026-11-03 (round 2, perf/decode-dedup, final bcb42eb): see the section above; evidence `report_dedup.py` output in `Desktop/LoRa-Merge-Gate-20261102/round2_report.txt`.

2026-11-03 (b52d20e + merge gate): early-expect hold; SF10/250k 5 Msps emulation 24.2 -> 13.2 s (old 14.3). Old-main live baseline `oldmain_live_20261103` (the `oldmain_live_20261102` attempt failed the harness receipt because `prepare.py` only pinned two detector workers when a profile was given; fixed in the harness) 10/13, misses the SF12/62.5k long frames; `ladder_live19_b52d20e` 13/13, 1.58-3.65 s. Full numbers in the section above and `Desktop/LoRa-Merge-Gate-20261102/merge_gate_report.txt`.

2026-11-02 (49867ab): recenter budget returns None (`tests/test_decoder_recenter_budget.py`); SF7 stress run `Desktop/LoRa-SF7-Stress-20261102/run_20261102_212142` 20/20 first-attempt, 2.58-3.38 s; 984 tests.


2026-11-02 (1a514d9): batch-ordered on-air filter; live run 14 `ladder_live16_b24ac93` 13/13, CRC 20/20, evidence complete, 541 s of decoder time, 8 junk captures; 982 tests.


2026-11-02 (b24ac93): on-air frame registry for cross-window copies; live run 13 `ladder_live15_30094f3` (13/13, CRC 20/20, max 3.91 s, 65 junk captures remaining); replays `replay_onair` of the three-message (9/9, 38 publications) and junk (10 detections, 2 packets) recordings identical to the previous build with zero copy drops; 981 tests.


2026-11-02 (076e1ac, 30094f3): placeholders off dropped spur bins, strict Data-message validity, intermod-copy arbitration; live runs 11 `ladder_live13_67d565e` (13/13, four defects' evidence) and 12 `ladder_live14_076e1ac` (13/13, CRC 20/20, evidence complete); 978 tests.


2026-11-02 (67d565e): gate ledger (`src/gate_ledger.py`, `tests/test_gate_ledger.py`) and the memory-scaled capture plan; live run 10 `ladder_live11_66e3ed2` 6/6 recovered before the mains guard stopped it, four non-admissions under the old fixed budget; replay `Desktop/LoRa-Raw-SF7x3-20261031/replay_ledger` 9/9 with the ledger on; 968 tests.


2026-11-02 (66e3ed2): spur-width peaks dropped before the gate cap (`tests/test_gate_doomed.py`); replays `Desktop/LoRa-Raw-SF7x3-20261031/replay_doomed_66e3ed2` (9/9) and `Desktop/LoRa-Raw-Junk-20261031/replay_doomed_66e3ed2` unchanged in output; live harness gained the `auto-default-workers2` profile (no capture variables, default plan receipt); the Heltec helper the harness runs was recovered from the 2026-10-26 transcript to `Desktop/LoRa-Heltec-Helper/` after its scratchpad copy vanished (run `ladder_live9_66e3ed2` aborted before any transmission for that reason).


2026-10-31 (night, 20ea5ff): frame-length-exact capture is the default where MemAvailable allows (`shared_capture_plan`, `tests/test_capture_default.py`); default-mode emulation replays `Desktop/LoRa-PiEmul-20261031/{junk,sf12_250}_new_default_5m`, full-speed `Desktop/LoRa-Raw-SF7x3-20261031/replay_default` (9/9), live run 9 `ladder_live8_20ea5ff` 13/13, CRC 20/20, median 2.81 s, max 4.17 s, 135 jobs; 951 tests.

2026-10-31 (night, b020438): sibling wait rewritten as a park off the worker (`tests/test_sibling_park.py`), shared-capture budget exhaustion made a non-admission instead of a fatal; 942 tests; emulated and full-speed replays identical in packets (details in the section above); evidence under `Desktop/LoRa-PiEmul-20261031/*_park*` and `Desktop/LoRa-Raw-SF7x3-20261031/replay_plain_park/`.

2026-10-31 (late evening, 404ca1d + 55a60ee): a verified [DUP] now covers its RF bucket (emulated SF10 recording 5 → 3 jobs, SF12/250k 10 → 7, same packets); alias choice (010 vs 057) and plain crop length left as is with the data above; live run 8 (`ladder_live7_404ca1d`): 13/13, CRC 20/20, 127 jobs, 2880 CPU s, median 2.91 s, one first-transmission miss (message 4, +13.8 s) investigated to the end of the evidence and not reproduced by the three-message raw recording (`Desktop/LoRa-Raw-SF7x3-20261031/`, 9/9 in both modes); `[GATE-UNCONFIRMED]` diagnostic added so the next live miss explains itself; 936 tests.

2026-10-31 (evening, small-host emulation + ring-lap fix): three concurrent arms (old main 921afce plain, branch plain, branch shared+checkpoints) on four E-cores each, `os.cpu_count()==4`, 1.6 GHz (first 10 Msps pass 2.0 GHz and PROCHOT-perturbed), four raw recordings at 5 Msps and 10 Msps; harness, outputs, `host_health.csv` and `pass_totals.py` under `Desktop/LoRa-PiEmul-20261031/`. 5 Msps: old main 2/4 messages, 194 jobs, 3456 CPU s, 9 loss lines; branch plain 4/4, 30 jobs, 2477 CPU s; branch shared 4/4, 40 jobs, 2496 CPU s, no abort. Clean 10 Msps: old 2/4, 66 jobs, 3047 CPU s; branch plain 4/4, 28 jobs, 2712 CPU s; branch shared aborted on the SF12/250k recording (ring lap at 11 s capture debt) → `capture_timeline.py` now seals the lapped band and saves partials instead of failing the service (tests in `test_capture_timeline.py`); reproducer afterwards: 2/2 packets, +25.3 s, no abort. Decoder-only subset on one capped E-core: 156.6 → 82.4 s, no lost output, 010 +1.6 s.

2026-10-31: corpus passes A–G and the forced-offset probe are under the session scratchpad (`corpus_ab.py`, `impair_ab.py`, `gridprobe.sh`); per-case logs and `summary.json` in `corpusG/` and `impairG/`. Committed as the frame-ladder change (3501fb4) on `fix/no-performance-regressions`.

2026-10-31 (live run 7, b27726e, grid gate + full-crop hold on top of everything above): 13/13 verified, `live_metrics --require-complete` passes, CRC audit 20/20, zero candidate-tier records, every message attributed to its first transmission, acquisition latency 0.47-3.04 s, publication median 2.87 s, max 4.38 s (an SF12/62.5k long frame whose early crop took 3.9 s to decode). Full crops: 62 skipped, 3 held, 2 released; twins: 50 held, 31 skipped, 19 released; no twin ran on a file with a packet. Jobs 166 (run 3: 310); decode time 707 s of which 159 s productive, 115 s redundant re-decodes of verified messages (run 3: 306 s) and 547 s on the external junk source, which was active again (38 SF9 junk jobs). `live_validation/ladder_live6_b27726e`, `live_metrics_ladder_live6_b27726e.json`.

2026-10-31 (live run 6, 61a312f, region rule + twin hold): 13/13 verified, `live_metrics --require-complete` passes, independent CRC audit 20/20 with no unverified record, every message attributed to its first transmission, acquisition latency 0.55-2.99 s, publication latency median 2.95 s and max 3.35 s (the lowest maximum of any run; reference 3.61 s). Patience promotions 45 (run 4: 155; other runs 109-133); 60 of 88 throttle events at the uncapped 24 peaks (run 4: 71 of 155). Twins: 47 held, 39 skipped, 8 released. `live_validation/ladder_live5_61a312f`, `live_metrics_ladder_live5_61a312f.json`.

2026-10-31 (later): live runs 1–3, the SF12/62.5k crop set, the runner-up ban/gate (2b7d58f), the one-symbol merge window and truncated-span hand-back (a1dc44a). Evidence under `Desktop/LoRa-Regression-Repair-20261031-ladder/`: `corpusP_final/`, `impairP_final/`, `sf12_62/` (old) + `sf12_62_final/` (new), `msg9_early_final/`, `live2_trunc_final/`, the runners (`files_ab.py`, `files_cmp.py`, `live_compare.py`) and `live_metrics_*.json` for all three live runs; live run directories under `LoRa-Local-Repair-20261028/live_validation/ladder_*`. Branch pushed to `origin/fix/no-performance-regressions`; not merged to main.
