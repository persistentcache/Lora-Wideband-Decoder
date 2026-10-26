# Performance & tuning

Throughput scales with CPU and memory bandwidth. On a multi-core
machine with a USB-3 SDR, the receiver keeps up with real-time
Meshtastic and LoRaWAN traffic at 28 Msps across every SF and BW.

Heavy back-to-back SF10/SF11 bursts can briefly push the gate over
real time on hosts with weak DRAM bandwidth. When that happens the
receiver throttles its decode workers automatically so the gate keeps
draining samples. Most live traffic never gets close to that
threshold.

For UHD radios, the web launcher reserves one physical CPU core for capture when at least four cores are available.
It assigns detection and decoding to the remaining cores.
Sibling CPU threads stay together, and the launcher respects an existing `taskset` assignment.
If CPU topology or `taskset` is unavailable, the launcher keeps the existing assignment.
Set `LORA_CAPTURE_ISOLATION=0` to retain your own CPU placement.
Other radio drivers keep their existing placement.

The SoapyUHD input uses 1,024 receive buffers by default.
These buffers hold samples before Python reads them, so they protect against short scheduling pauses.
The B200 and B210 need about 8 MB for these buffers at the default frame size.
For direct `src/soapy_rx.py` use, `--uhd-recv-frames N` overrides the count.
A count of zero restores the UHD default.
The larger Python queue and `buf_seconds` do not prevent loss inside the USB transport.
Direct command-line pipelines need their own capture CPU assignment.

Live statistics report recent processing speed, input speed, and queued sample duration.
The measurement starts after samples arrive, so radio startup time does not lower the displayed rate.
An `INPUT WARNING` reports input below the requested rate.
A `KEEP-UP WARNING` reports processing below the received rate with a growing backlog.
Warnings clear after the rates recover and the backlog drains.

On Linux, the `powersave` governor adds noticeable latency variance.
For consistent measurements use `performance`:

```bash
sudo cpupower frequency-set -g performance
```

If your host can't hold 28 MHz, drop the rate in `lora.toml`:

```toml
[radio]
rate_hz      = 14000000   # 14 Msps still covers half the US915 band
bandwidth_hz = 14000000
```

## Knobs that solve common problems

Settings live in `lora.toml [detect]`; most are also CLI flags.

| Symptom | What to change |
|---|---|
| False positives | `threshold` up to 0.8 |
| Missed real packets | `threshold` down to 0.4 |
| Short weak bursts vanish | `energy_threshold` 12.0 → 5.0 |
| Spike at center frequency (LO leak) | `--dc-notch 0.5` |
| Duplicate detections / saturation | turn SDR gain down in `[radio]` |
| Detector buffer overflows during short bursts | Increase `buf_seconds` if RAM permits. For USB input loss, inspect the capture log. |
| `CATCHUP skip` / ring wraps during dense traffic on a small host | Automatic. `detect_workers = -1` takes 3 workers on 4-7-core hosts when RAM allows (~0.8 GB per worker + ~2 GB for the gate at 5 Msps), and under a sustained deficit the gate degrades gracefully — strongest peak only, then whole windows skipped (`hyb=…/Ns` in `[STAT]`) — instead of wrapping the ring. Frequent skips mean the host cannot keep up at this rate: lower `rate_hz` or add cooling. |

## Repo layout

```
Lora-Wideband-Decoder/
├── README.md, LICENSE, install.sh, requirements.txt
├── lora.toml          # all configuration
├── run/
│   ├── web.py            # web UI entry point
│   └── headless.py       # detector pipeline, no UI
├── src/
│   ├── detector.py        # gate + Schmidl-Cox + pipeline
│   ├── decoder.py         # multi-pass soft decoder + protocol parsers
│   ├── detect_pool.py     # multiprocess detection pool
│   ├── config.py, lora_config.py, sdr_profiles.py
│   ├── soapy_rx.py        # SoapySDR streamer
│   └── web/               # Flask UI
├── docs/
├── lora_web/          # runtime state (gitignored)
└── captures/          # output (gitignored)
```

## Regression tests

Install `pytest` in the Python environment that contains the project dependencies.
Run `python3 -m pytest -q` from the project root.
The tests use simulated device access and do not require a connected radio.
