# Performance & tuning

## What to expect

Performance scales with CPU bandwidth and core count. On a multi-core
host with a USB-3 SDR:

- **Normal Meshtastic / LoRaWAN traffic**: real time at 28 Msps
  wideband across every SF / BW combination.
- **Dense back-to-back SF10 / SF11 bursts**: the gate's per-window
  detect work can briefly exceed real time on hosts with limited DRAM
  bandwidth. The receiver auto-throttles workers when sample drops show
  up. Real-world networks almost never approach this burst rate.
- **CPU affinity**: auto-isolated. The first four cores of the
  available affinity set are reserved for the gate's reader and detect
  pool; decode workers run on the rest. Wrap the launch in
  `taskset -c …` to override.
- **CPU governor**: on Linux, `powersave` introduces latency variance.
  For consistent measurements, switch to `performance`:

  ```bash
  sudo cpupower frequency-set -g performance
  ```

If your host is too slow for 28 MHz, drop the rate in `lora.toml`:

```toml
[radio]
rate_hz      = 14000000   # 14 Msps still covers half the US915 band
bandwidth_hz = 14000000
```

## Common tuning knobs

In `lora.toml [detect]`, also exposed as CLI flags:

| Problem | Fix |
|---|---|
| False positives | raise `threshold` to 0.8 |
| Missed detections | lower `threshold` to 0.4 |
| Weak short bursts missed | lower `energy_threshold` from 12.0 to 5.0 |
| LO-leak false peak at center | `--dc-notch 0.5` |
| Adjacent-channel duplicates / saturation | lower SDR gain in `[radio]` |
| Dropped samples on slow hosts | raise `buf_seconds` from 16 to 32 |

## Project layout

```
lora_ml/
├── README.md
├── LICENSE
├── install.sh
├── lora.toml          # all configuration (radio / detect / decode / web)
├── requirements.txt
├── run/               # what you run
│   ├── web.py            # start the web UI
│   └── headless.py       # detector pipeline, no UI
├── src/               # backend
│   ├── detector.py        # gate + Schmidl-Cox + pipeline
│   ├── decoder.py         # multi-pass soft decoder + protocol parsers
│   ├── detect_pool.py     # multiprocess detection pool
│   ├── config.py / lora_config.py / sdr_profiles.py
│   ├── soapy_rx.py        # SoapySDR → CS16 IQ on stdout
│   └── web/
│       ├── app.py         # Flask web UI implementation
│       └── templates/
├── docs/              # this folder + reference papers
├── lora_web/          # runtime state (settings, keys, autosave — gitignored)
└── captures/          # local capture output (gitignored)
```
