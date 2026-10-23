# USRP capture repair

The user authorized these repairs, regression tests, and a push to `main` after the tests pass.

The capture process must open UHD with 1,024 receive buffers by default. An explicit override must remain available.
The web launcher must reserve one physical CPU core for UHD capture when enough cores are available.
The launcher must respect the existing CPU assignment and keep sibling threads together.
The detector reader must stay within its assigned CPUs.
Live throughput must use recent sample counts and exclude radio startup time.
Warnings must distinguish slow input from a growing processing backlog and clear after recovery.
Detection sensitivity and packet decoding must remain unchanged.

## Implementation

- [x] Add failing tests for device arguments, CPU assignment, throughput, and web log handling.
- [x] Repair UHD device arguments in `src/soapy_rx.py`.
- [x] Add `src/cpu_affinity.py` and use it in the web launcher and detector reader.
- [x] Add `src/throughput.py` and use it for live statistics and warnings.
- [x] Update performance documentation and run the complete test suite.
- [x] Compare synthetic detection results against the original revision.
- [x] Repeat live tests at 3, 16, and 32 million samples per second.
- [x] Request an independent code review and resolve material findings.

After these tests pass, commit and push to `main` as the user requested.
Make sure that the remote commit matches the local commit.

## Test conditions

Tests cover non-UHD drivers, explicit UHD arguments, limited CPU assignments, missing CPU topology, small hosts, startup delay, slow input, overload, and recovery.
The connected radio is a B200mini with its antenna on RX2. No known LoRa transmitter is active.
Live tests can establish capture performance. Synthetic comparisons cover detection behavior, but do not replace live packet tests.

The reviewer found a warning recovery failure after a loss during the first read.
A regression test reproduced the failure, and the corrected meter passes that test.
The complete suite passes 30 tests.

Eight synthetic signal cases produce exactly the same detection results as revision `858f9d8`.
Five cases contain detections, including a weak signal case.
The comparison covers detection output, not packet decoding.

Each live test runs for 90 seconds with the connected B200mini and the full detector pipeline.
The runs at 3 and 16 million samples per second record zero radio overflows.
The run at 32 million samples per second records one radio overflow, compared with 2,211 before the repair.
The detector records no buffer skips or false startup warnings in these runs.
These measurements show reduced sample loss, not a guarantee of lossless capture.
