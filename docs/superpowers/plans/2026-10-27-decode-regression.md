# Decode throughput and live regression plan

Goal: resolve the four captured-but-unreported Heltec messages and verify the decoder on the Pi with retained IQ and transmitter evidence.

The user authorized investigation, fixes, deployment and full live regression. Preserve captures and user settings; do not change the Pi's boot services. During tests, enable all supported/unknown decoding and retain IQ. Respect the transmitter helper's 60-second configuration hold and 20-second send guard. Restore the original radio and receiver configuration. Count live reception separately from offline recovery, and CLI acceptance separately from confirmed radio transmission.

## Implementation and evidence

- [x] Efficient, decimation-aligned FFT sizes without losing original IQ duration; fair, pressure-aware access to deferred decoding. Commit c858040.
- [x] Replay all 197 original IQ captures: all 42 numbered messages recovered, including the four missing messages; all 47 previously reported unique packet variants preserved.
- [x] Live preset/custom sweep and independent TX/packet/health audit: 36 of 37 texts received. LONG_SLOW message 0016 remains unresolved because RF completion and corresponding IQ were not established. A later three-message LONG_SLOW retest passed, but does not resolve that historical send.
- [x] Verify actual modem readback: this Heltec firmware silently substitutes unsupported preset values. Test SF11/500 explicitly; baseline plus three texts received.
- [x] Scale slow acquisition seed width at low input rates, preserving decisions at 20 Msps and above. Commit 078d076.
- [x] Preserve contiguous slow tails through delayed scans with validated, nonconsuming ring history; stop honestly on discontinuity or queue failure. Commit c327cba.
- [x] Use current effective capture budget for slow admission while retaining pressure and spacing guards. Commit 8f3dd51. Three narrow SF12/62.5 texts and baseline received.
- [x] Retain supported long narrow frames using one shared bound for collection and trimming; retain explicit overrides and bounded wideband fallbacks. Commit ebdbccd.
- [x] Prevent repeated samples when registering current/pool tails; test partial overlap, gaps and 25/50/75% overlap. Commits 243ad59 and b716263.
- [x] Complete the 17-message live matrix on b716263: 12 received; five misses investigated from preserved IQ. This run exposed further centering, noise-scoring and processing defects; it was not a passing final result.
- [x] Prevent save-stage center estimation from selecting a weak region after rejecting dominant broad energy. Commit 13ac181; retained-IQ diagnostic controls independently validated.
- [x] Require learned-channel preamble energy to persist at a common dechirped bin instead of averaging unrelated per-symbol maxima. Commit 41bedaa. Noise controls rejected and real/weak/fractional-bin/SFO controls retained. Scanner-only CPU cost increased and was checked on the Pi.
- [x] Complete eight-message live retest on 41bedaa: five received; all eight had TX-completion evidence and corresponding decodable IQ. Configured-channel quiet probe no longer produced the prior continuing queue flood. Queue drained and settings restored.
- [x] Respect the worker's actual CPU budget in its supervisor wall deadline. Commit 0f58f3b; preserve finite deadlines and document them in START records.
- [x] Give a wall-preempted primary decode one slow retry when a plausible explicit header was parsed. Commit 383459c. A failed header checksum is search progress, not packet validation; header-like noise can qualify once. Empty jobs, worker death, isolated jobs and repeat slow attempts remain excluded.
- [x] Correct successful-frame masks using measured header position and actual payload block accounting (8e9b008). Add one bounded later-frame retry using mean CFO and zero preamble slope after a verified first frame and a valid, non-fluke header with failed payload CRC (53f06e0). Preserve earlier skip bins, five-attempt limit and CPU deadline. Ordinary unchanged capture006 now yields both exact PL97 frames with independent received CRCs a8e4/6c8a. Final source suite: 316 tests.
- [x] Replay all 197 original captures against 53f06e0: all exit successfully, all 42 expected messages match, all 47 original-live and 54 prior-replay exact variants are preserved, with no additions or identity/plaintext conflicts. Separate numerical CRC checks confirm the two-frame regression; original corpus comparison retains the original four-worker / 10-second CPU-base / 150-second per-file wall configuration.
- [x] Deploy the reviewed detector/decoder pair with backups and independently checked rollback guards (13 deployment fault tests, 68 harness tests, 34 audit tests). Verify the Pi source hashes and 10 Msps input.
- [x] Complete the Recovery matrix on 53f06e0 (06:38–06:59 EDT): 4/8 texts received; all eight have RF-completion evidence. Baseline, SF12/250, SF11/500 and SHORT_TURBO passed; four narrow texts did not. Queue drained and settings restored. The complete narrow burst in capture002 was later identified as automatic NodeInfo, not an intended text. The 180-second configured-channel quiet probe produced no new submissions.
- [x] Correct the 96-symbol limit in both payload drift trackers, retaining the old horizon for shorter frames and extending to the actual FEC/LDRO/CRC symbol count. Commit 783502b recovers unchanged capture002 NodeInfo in under four host seconds, with received CRC0a5e, and preserves both Retest006 texts.
- [x] Allow a bounded slow-scan opportunity at the ordinary cap floor only with a free detector worker and slot, less than one hop of unread IQ, and no emergency state; retain peak throttling, ring, spacing and single-scan guards. Commit 3b4f79d adds fresh capacity diagnostics. Real compute pressure was also measured, so the false-veto model does not explain every prior miss. Combined suite: 369 tests.
- [x] Repeat all 197 original captures against 3b4f79d: 42/42 messages, all47 original-live and54 prior-replay variants preserved; no additions, losses, conflicts or errors.
- [x] Complete the Horizon matrix on 3b4f79d: 5/8 texts received; the three controlled long PL97 texts remain unmatched. All eight have RF-completion evidence. All five live test messages and one long NodeInfo have independent received-CRC confirmation. Queue drained, no input-buffer drops, and all settings restored. Real unfinished-worker pressure prevented every floor-spare opportunity; no claim that this policy fixed the remaining misses.
- [x] Freeze application changes for a controlled original-versus-current check. Original b893a234 and current3b4f79d both recover42/42 messages and the same54 unique raw variants from identical197 captures under the same four-worker/10CPU-base/150wall recipe. Current wall time117.55s versus original139.46s is an observed15.7% reduction; offline recall is equal.
- [ ] Complete matched eight-message live runs on the isolated original b893a234 checkout and frozen current3b4f79d, then assess gains/regressions and any rollback before publication.
- [ ] Audit exact IDs, PHY, CRCs, queue/backlog/drop/timeout behavior, retained IQ and restoration. Investigate every remaining miss before stating the outcome.
- [ ] Publish the reviewed changes and final evidence report; retain Pi-specific configuration locally.

## Run interpretation

The completed runs used different source revisions and pacing; their success fractions must not be pooled into a reliability estimate. Original-corpus recovery is offline evidence, not live reception. The live transmitter exercises Meshtastic across LoRa PHY settings; enabling all decoder protocols does not provide on-air coverage of every protocol.

Long SF12/62.5 captures can retain 32.065792 seconds (about 128.3 MB at 500 ksps complex64). That prevents truncation of supported long frames, at the cost of memory and added decode latency. Zero input-buffer drops does not imply every detection window was searched; review pressure skips and worker timeouts separately.

Evidence is retained outside Git under `/home/david/Desktop/Heltec-Test-20261027-1951` and `/home/david/Desktop/Heltec-Regression-*`, with corresponding Pi directories under `/data/lora-tests`. Credentials, raw IQ and Pi-only service configuration are excluded from commits.
