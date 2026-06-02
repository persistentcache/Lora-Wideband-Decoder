# lora_ml — wideband passive LoRa receiver

A self-hosted, single-user **passive intercept receiver** for LoRa traffic.
Connects to any SDR supported by SoapySDR (bladeRF, HackRF, RTL-SDR, LimeSDR,
USRP, PlutoSDR, Airspy), demodulates the IQ in software, decodes the result,
and shows it in a local web UI.

```
SDR IQ → energy gate (Welch + max-hold) → Schmidl-Cox preamble lock
       → dechirp + soft demod → CRC / decrypt / verify → web UI
```

Detection is Schmidl-Cox (no ML at runtime). The decoder is a multi-pass
soft demodulator that covers **SF7–SF12** at any **BW (62.5 / 125 / 250 /
500 kHz)**. It handles typical Meshtastic / LoRaWAN / MeshCore traffic in
real time on a multi-core host; very dense back-to-back bursts at slow SFs
may exceed real-time on hosts with limited memory bandwidth (see
**Performance notes** below).

## What it identifies

A protocol is **named** only when there's discriminating evidence — never on
a structural guess. Confidence tiers:

| Tier | Meaning |
|---|---|
| **verified** | cryptographic proof — Meshtastic AES decrypt, MeshCore ADVERT Ed25519 signature, or MeshCore public-channel decrypt + MAC |
| **confirmed** | behavioral proof — LoRaWAN DevAddr + monotonic FCnt over ≥2 frames, or a MeshCore frame routed through ≥2 verified nodes |
| **unknown (+hint)** | structurally resembles a protocol but unconfirmed — shown honestly, never claimed |

Meshtastic broadcasts on the public default channel decrypt with the
built-in PSK. Meshtastic direct messages are PKI-encrypted and surface as
header-only entries (the link is visible, payload content is not).

## Install

**Debian / Ubuntu — one command:**
```bash
./install.sh
```
Installs SoapySDR + every device module in the repo (universal SDR support),
the native vendor tools (`bladeRF-cli`, `hackrf_transfer`, `rtl_sdr`), the
Python dependencies, and SDR device access (`plugdev` group). Reboot or
re-login after the group change so it takes effect.

**Other systems / manual:**
```bash
pip install -r requirements.txt   # numpy, scipy, cryptography, flask
# Plus, from your distro's package manager:
#   - SoapySDR with its python3 bindings
#   - your SDR's SoapySDR module (e.g. soapysdr-module-bladerf)
#   - libusb-1.0
```

Python 3.11+ recommended (uses stdlib `tomllib` for `lora.toml`; older Python
needs `pip install tomli`). No paths are hardcoded — run from any directory.

## Run

Edit **`lora.toml`** (`[radio] [detect] [decode] [web]`) — the example file
is heavily commented and covers most use cases. Then start the web UI; it
launches the SDR → detector pipeline for you:

```bash
python3 lora_web/lora_web.py
# Open the URL it prints (default http://127.0.0.1:5000).
```

In the UI you can: start / stop the receiver, watch the live packet feed,
inspect decoded message content, view the node graph and waterfall, manage
channel keys, and replay recorded captures offline.

**Headless / no web UI** — invoke the detector directly:

```bash
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000;
    set bandwidth rx 28000000; set agc rx on;
    rx config file=/dev/stdout format=bin n=0; rx start; rx wait" \
| python3 scripts/lora_detect.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
    --decode --export-iq captures/
```

Or, **offline replay** of a recorded `.sc16` / `.cf32` file:

```bash
python3 scripts/lora_detect.py -f recording.sc16 \
    -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```

## Channel keys

The web UI's **Config → Channel keys** tab manages decryption keys (also
editable as `lora_keys.json`). The two **Public / Default** keys
(Meshtastic `AQ==`, MeshCore `8b33…`) are built in and value-locked. Add
custom per-protocol keys as needed. LoRaWAN doesn't need a key —
identification is structural + behavioral.

## Security & deployment

This is a **self-hosted, single-user, local tool**. It requires a physical
SDR, so you run your own instance and view it in your own browser.

- **No authentication.** The web UI has no login. Anyone who can reach the
  port can start / stop the receiver, change settings, read intercepted
  traffic, and read keys.
- **Localhost-only by default** (`127.0.0.1`). To allow LAN access set
  `[web] host = "0.0.0.0"` in `lora.toml`, but only on a trusted network,
  and ideally behind a reverse proxy (nginx, Caddy) that adds auth + TLS.
  The app prints a warning when bound to a non-loopback address. Do not
  expose it directly to the public internet.
- **SDR access via the `plugdev` group** — don't run as root.
- **Legal / privacy.** This passively intercepts RF. You are responsible
  for complying with local laws on radio reception and privacy. Don't
  publish decoded traffic, node identities, or channel keys you don't have
  the right to share.

The bundled `app.run()` (Werkzeug) server is fine for local single-user use;
a production WSGI server is only relevant if you deliberately host it for
others.

## Performance notes

Performance scales with CPU bandwidth and core count. Reasonable
expectations on a multi-core host with a USB-3 SDR:

- **Normal traffic** (Meshtastic / LoRaWAN at typical message rates):
  works in real time at 28 Msps wideband across all SF / BW combinations.
- **Dense bursts** (many SF10 / SF11 messages back-to-back with no gap):
  the gate's per-window detection work can momentarily exceed real time
  on hosts with limited DRAM bandwidth. The receiver auto-throttles
  workers when sample drops are detected, but extreme bursts on slow
  hosts may still drop a fraction of samples — bursts on real-world
  networks rarely approach this regime.
- **CPU affinity** is auto-isolated: the first 4 cores of the available
  affinity set are reserved for the gate's reader / detect pool; decode
  workers are pinned to the remaining cores. If you explicitly wrap the
  pipeline in `taskset -c …` it's respected and the workers share that
  affinity.
- **CPU governor.** On Linux, the `powersave` governor can introduce
  noticeable latency variance. Set the `performance` governor for
  consistent measurements: `sudo cpupower frequency-set -g performance`.

If your host is too slow for the full 28 MHz, drop the sample rate in
`lora.toml`:

```toml
[radio]
rate_hz      = 14000000   # 14 Msps still covers half the US915 band
bandwidth_hz = 14000000
```

## Tuning

Common knobs in `lora.toml [detect]` (also available as CLI flags):

| Problem | Fix |
|---|---|
| False positives | raise `threshold` (e.g. `0.8`) |
| Missed detections | lower `threshold` (e.g. `0.4`) |
| Weak short bursts missed | lower `energy_threshold` (default 12.0; try 5.0) |
| LO-leak false peak at center | add `--dc-notch 0.5` |
| Adjacent-channel duplicates / saturation | lower SDR gain in `[radio]` |
| Dropped samples on slow hosts | raise `buf_seconds` (default 16, try 32) |

## Meshtastic presets (reference)

| Preset | SF | BW | CR |   | Preset | SF | BW | CR |
|---|---|---|---|---|---|---|---|---|
| SHORT_TURBO | 7 | 500k | 4/5 |   | LONG_FAST | 11 | 250k | 4/5 |
| SHORT_FAST | 7 | 250k | 4/5 |   | LONG_MODERATE | 11 | 125k | 4/8 |
| SHORT_SLOW | 8 | 250k | 4/5 |   | LONG_SLOW | 12 | 125k | 4/8 |
| MEDIUM_FAST | 9 | 250k | 4/5 |   | VERY_LONG_SLOW | 12 | 62.5k | 4/8 |
| MEDIUM_SLOW | 10 | 250k | 4/5 |   | | | | |

## Layout

```
lora_ml/
├── install.sh          # one-command installer (Debian / Ubuntu)
├── lora.toml           # all configuration
├── .env.example        # environment-variable overrides
├── requirements.txt    # Python dependencies
├── scripts/            # core runtime
│   ├── lora_detect.py     # gate + Schmidl-Cox + pipeline
│   ├── decode_header_v3.py # multi-pass soft decoder + parsers
│   ├── detect_pool.py     # multiprocess detection pool
│   ├── config.py / lora_config.py / sdr_profiles.py
│   └── soapy_rx.py        # universal SoapySDR → stdout streamer
├── lora_web/           # Flask web UI
├── docs/               # reference papers
└── captures/           # local capture output (gitignored)
```

## License

MIT — see `LICENSE`.
