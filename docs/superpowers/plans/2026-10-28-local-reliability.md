# Local receiver reliability implementation plan

> **For agentic workers:** Use systematic debugging and test-driven development for each change. Root owns live hardware and integration; delegated workers have explicit method ownership.

**Goal:** On the local USRP B200mini/Heltec rig, recover every controlled test message with zero sample drops and low publication latency.

**Architecture:** Keep the existing receiver and PHY decoder. Repair the measured capture service deficit and blocking queue completion, preserve complete contiguous frames, and publish verified packets while further decoder searches continue. Measure acquisition, capture, decode, and publication separately.

**Tech stack:** Python, NumPy/SciPy, SoapyUHD, existing Flask UI and native CS16/IQ capture paths.

## Authority and scope

- The user explicitly rejected the previous result and requested low latency, no misses, and no sample drops. This authorizes fixes and continued local live testing.
- Pi is disconnected and must not be contacted. Main and previous frozen evidence are preserved. No publication/deployment is part of the current development loop.
- Work in the existing isolated `fix/decode-throughput-regression` worktree. Starting source is `3b4f79d`; fresh baseline is 369 passing tests.
- Keep 915 MHz center, 10 Msps / 10 MHz receiver bandwidth, RX2, gain40, full scan, all protocols, unknown reporting, and capture retention. Do not obtain passing results by reducing coverage or discarding difficult packets.
- Working latency target: at most5 seconds from the first firmware RF-completion event for a test ID to its first verified published result. User clarification is pending; this target is an engineering assumption. Report capture/queue/decode/publication timing separately.
- Restore Heltec configuration after each live test. Retain original IQ, identifiers, radio readbacks, RF-completion logs and all negative evidence.

## Evidence and design choices

The prior paired run is `/home/david/Desktop/LoRa-Local-USRP-20261028`: original4/8 versus modified5/8, queue maxima60/4, original zero sample skips versus modified16.56seconds. Both missed the three long SF12/62.5 CR4/8 PL97 test messages. Both source trees and original Heltec configuration were restored/verified.

Current failed long candidates contain valid headers and partial controlled packet identities but lack the end of their transmitted frames. Repeated `NB-CROP queue full` warnings finalize contiguous prefixes early. On queue saturation, finalization then blocks putting `FINAL` onto the same full queue, potentially for30seconds. Real recorder reproduction shows both truncation and blocking. The single crop worker also has a measured service deficit with three concurrent streams.

Increasing buffers/timeouts alone cannot satisfy latency and sustained capture requirements. A wholesale decoder rewrite is not justified by this evidence. The selected approach repairs these demonstrated boundaries while preserving numerical filtering, frame validation and existing protocol behavior. Independent received-CRC verification remains mandatory for claimed recoveries.

## Tasks

- [ ] Reproduce and pin crop saturation using real recorder tests: accepted tails remain complete and contiguous; finalization cannot wait on its own full data queue. Establish numerical and timing baselines for actual streaming decimation.
- [x] Optimize FIR execution without changing taps, phase, output grid, sample count or mixer behavior. Serial real/imaginary FIR is bit-identical across tested chunk seams and a received PL97 packet. Actual three-stream cost fell from0.74–0.81s to0.29–0.38s per0.5s input. Production change and regression tests independently reviewed; live acceptance remains pending.
- [ ] Repair ordered finalization and accepted-tail retention. Address both full queues and the three-pending-job early-finalization limit; choose bounded capacity from measured demand, not an unbounded queue. Add real failure-path tests before implementation.
- [x] Extract `BackgroundDecoder._publish_packet_line(line, fpath, *, received_at=None)` and call it as complete worker lines arrive. Removed deferred duplicate publication. Real-manager tests prove publication before job completion, no double counts, preserved later frames, safe malformed/fragmented lines, and preserved output after worker timeout/EOF. Root independently ran33 publication/deadline tests; all passed.
- [ ] Record continuous native CS16 from a controlled long transmission, retaining complete input independently of the detector. Check hardware identity/settings, zero stream errors, exact sample continuity, full RF events and Heltec restoration.
- [ ] Replay the same complete input through frozen checkpoints. Trace any remaining pre-submission delay; if capture length dominates, implement a tested frame-completion/early-decode mechanism that preserves later frames rather than shortening captures blindly.
- [x] Update the test-only audit to use actual packet publication timestamps and the first RF completion for latency. Match exact sender, ID, text, PHY and received CRC; automatic announcements and repeats do not inflate controlled recovery. Agent verified24 timing tests; old evidence correctly has no eligible explicit publication timestamps. Final result audit still requires independent root review.
- [ ] After each coherent fix, review source diff and run relevant regression tests plus the full suite. Freeze a source checkpoint for live testing; do not edit a running source tree.
- [ ] Run focused long-packet validation, then repeat the full known-failure matrix and broader preset/SF regression. Require100% controlled recovery, zero input/catch-up skips, no recorder truncations, bounded queues, and the agreed latency target. Do not mark complete on an improved percentage alone.
- [ ] Re-run the retained197-capture corpus to check existing decode recall. Review final live results independently; record any unresolved failure and continue diagnosis before claiming acceptance.

## Review focus

- Packet received before the worker stalls or times out must remain published exactly once.
- Multiple overlapping long captures must not truncate one another or block input consumption.
- Arbitrary input chunk sizes must preserve FIR phase/sample alignment and frame CRC.
- Distinct frames in one recording, including repeated/relayed frames, must remain discoverable.
- A zero-drop counter alone does not prove detection coverage; audit every controlled ID and full received-frame evidence.

Evidence and scratch experiments for this iteration live under `/home/david/Desktop/LoRa-Local-Repair-20261028`. Do not put raw IQ or credentials in Git.

## Additional confirmed boundaries

- The pending-count3 eviction independently truncates a12.337s frame even with a fully drained crop queue. Removing that count limit alone would create roughly12–14 overlapping FIRs for one long carrier. The proposed correction separates retained capture spans from shared filtered timelines.
- Worst-case237-byte capture retention inherently delays a normal97-byte frame. A header-only checkpoint may schedule an early decode view at the measured frame boundary, while the shared timeline retains the full continuation. Header checksum alone must never authorize discarding later IQ or publishing a packet.
- EOF currently joins saves before finalizing pending crops. A real-recorder reproduction leaves an unsubmitted save when decoder.pending()==0. Repair ordering to finalize pending, join crop/save, then drain decoding.
- Native paced replay uses the live stdin path and a dedicated process group. An independent failure test found and verified a cleanup correction for children surviving their parent, including a TERM-resistant child. Replays are diagnostic until pacing and settings parity are verified.
