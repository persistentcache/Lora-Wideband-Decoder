# Decode throughput and live regression plan

> Use executing-plans with independent implementation and review agents.

Goal: resolve the four captured-but-unreported Heltec messages and verify the fix on the Pi under a full live preset sweep.

Design approved by user: efficient transform sizes without removing IQ data; bounded opportunity for deferred decoding while preserving capture pressure guards; completion based on pending work rather than a fixed grace period. The investigation and retained reproductions are in `/home/david/Desktop/Heltec-Test-20261027-1951/INVESTIGATION.txt`.

Constraints: preserve original captures and user settings; no boot-service changes; retain all protocols, unknown reporting and IQ during test; respect the Heltec's 60-second reboot hold and 20-second send gap; restore previous radio/logging state; no false-positive claim based solely on unrelated node traffic.

- [ ] Baseline full pytest suite.
- [ ] Decoder: failing regression for problematic FFT sizes, preserve sample rate and decimation alignment; efficient zero padding, test unchanged payloads on retained originals and representative families.
- [ ] Scheduler: failing regression showing deferred work gets a bounded opportunity while fast work remains queued, only one slow job reserved and capture pressure protected; preserve reservation lifecycle and cleanup.
- [ ] Harness: isolated next run directory, exhaustive preset/custom coverage, numbered TX IDs and readback, receiver logs and retained IQ, queue-aware drain, explicit incomplete reporting and offline replay fallback before misses classification. Independent restoration fallback.
- [ ] Whole suite plus independent code review and retained-capture replay.
- [ ] Deploy reviewed source to Pi with backups; verify actual input rate and launch timed live sweep.
- [ ] Set in-session wake timer; inspect progress and complete audit when TX ends and pending work is accounted for.
- [ ] Preserve final evidence, restore state, report live vs offline recovery separately and unresolved limitations.

Review focus: FFT phase/frequency alignment and noninteger sample-rate tokens; SF and protocol diversity; slow job reservation leak/pressure bypass; incomplete transmissions and automatic repeats; errors during cleanup or loss of network connectivity.
