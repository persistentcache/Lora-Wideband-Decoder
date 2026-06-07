# Performance & tuning

Throughput scales with CPU and memory bandwidth. On a multi-core
machine with a USB-3 SDR, the receiver keeps up with real-time
Meshtastic and LoRaWAN traffic at 28 Msps across every SF and BW.

Heavy back-to-back SF10/SF11 bursts can briefly push the gate over
real time on hosts with weak DRAM bandwidth. When that happens the
receiver throttles its decode workers automatically so the gate keeps
draining samples. Most live traffic never gets close to that
threshold.

CPU pinning is automatic. The first four cores in the process's
affinity set are reserved for the SDR reader and the detect pool;
decode workers run on the rest. If you want different placement, wrap
the launch in `taskset -c …`.

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
| Dropped samples on a slow host | `buf_seconds` 16 → 32 |

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
