# lora_ml — wideband passive LoRa receiver

A self-hosted, single-user intercept receiver for LoRa traffic. Streams IQ
from your SDR, demodulates and decodes in software, surfaces everything in a
local web UI.

```
SDR IQ → energy gate (Welch + max-hold) → Schmidl-Cox preamble lock
       → dechirp + soft demod → CRC / decrypt / verify → web UI
```

The detector uses Schmidl-Cox autocorrelation. The decoder is a multi-pass
soft demodulator covering SF7–SF12 at every standard bandwidth — 62.5, 125,
250, and 500 kHz. It keeps up with real Meshtastic, LoRaWAN, and MeshCore
traffic in real time on a multi-core host. Heavy back-to-back SF10 / SF11
bursts can outrun the receiver on hosts with limited memory bandwidth; see
**Performance notes** below for the knobs that handle that.

## Quick start

```bash
./install.sh                # Debian / Ubuntu
python3 run/web.py          # opens http://127.0.0.1:5000
```

That's it. Point your browser at the URL it prints, hit **Start** in the UI,
and intercepted packets stream in live. The pipeline (SDR capture + detector
+ decoder) is launched from the web UI based on your `lora.toml`.

If your SDR is already plugged in and you've rebooted (or re-logged in) after
the installer added you to `plugdev`, the UI's **Config → SDR/Radio →
Detect** button finds it automatically.

## SDR support

Two capture paths, picked automatically per device:

- **bladeRF** uses its native `bladeRF-cli` streamer. This is the only path
  that reliably sustains 28 Msps; SoapyBladeRF can't open a stream at that
  rate. Recommended SDR for the full bandwidth.
- **Every other device** — HackRF, RTL-SDR, LimeSDR, USRP, PlutoSDR, Airspy,
  and anything else with a SoapySDR module — streams through `src/soapy_rx.py`,
  which emits raw CS16 IQ to stdout for the detector.

Named profiles live in `src/sdr_profiles.py`.

## What it identifies

The receiver names a protocol only when it has discriminating evidence.
Confidence tiers:

| Tier | Meaning |
|---|---|
| **verified** | cryptographic proof — Meshtastic AES decrypt, MeshCore ADVERT Ed25519 signature, or MeshCore public-channel decrypt + MAC |
| **confirmed** | behavioral proof — LoRaWAN DevAddr with monotonic FCnt over ≥2 frames, or a MeshCore frame routed through ≥2 verified nodes |
| **unknown + hint** | structurally resembles a protocol but unconfirmed — shown honestly, never claimed |

Meshtastic broadcasts on the public default channel decrypt with the
built-in PSK. Direct messages are PKI-encrypted, so they surface as
header-only entries — the link is visible, the payload is not.

## Install

Debian / Ubuntu, one command:

```bash
./install.sh
```

That installs SoapySDR and every device module, the native vendor tools
(`bladeRF-cli`, `hackrf_transfer`, `rtl_sdr`), Python dependencies, and adds
your user to the `plugdev` group for SDR access. Reboot or re-login after so
the group change takes effect.

On other systems, install the Python deps yourself and pull SoapySDR plus
your SDR's Soapy module from your package manager:

```bash
pip install -r requirements.txt
# distro packages: soapysdr-tools, python3-soapysdr,
# soapysdr-module-{bladerf,hackrf,rtlsdr,...}, libusb-1.0
```

Python 3.11+ is recommended. Older Python needs `pip install tomli` to load
`lora.toml`.

## Channel keys

Manage decryption keys in the web UI's **Config → Channel keys** tab. The
public defaults for Meshtastic (`AQ==`) and MeshCore (`8b33…`) are built in
and value-locked. Add custom per-protocol keys as needed. LoRaWAN doesn't
need a key — identification is structural and behavioral.

## Headless mode

If you don't want the web UI, pipe the SDR directly into the detector:

```bash
# bladeRF + 28 Msps:
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000;
    set bandwidth rx 28000000; set agc rx on;
    rx config file=/dev/stdout format=bin n=0; rx start; rx wait" \
| python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
    --decode --export-iq captures/

# Or via SoapySDR (HackRF / RTL-SDR / LimeSDR / USRP / etc.):
python3 src/soapy_rx.py --driver hackrf -f 915000000 -s 20000000 -b 20000000 \
| python3 run/headless.py -r 20000000 -b 20000000 -c 915.0 -t sc16 --decode

# Replay a recorded file:
python3 run/headless.py -f recording.sc16 \
    -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```

## Security & deployment

This is a self-hosted local tool. It needs a physical SDR, so you run your
own instance and view it in your own browser.

The web UI has no authentication. Anyone who can reach the port can start
and stop the receiver, change settings, read decoded traffic, and read keys.
For that reason it binds to `127.0.0.1` by default and prints a warning if
you change that. If you need LAN access, set `[web] host = "0.0.0.0"` in
`lora.toml` — but only on a trusted network, and ideally behind a reverse
proxy that adds auth and TLS. Never expose it directly to the public
internet.

Run as a normal user in the `plugdev` group. Not as root.

You're responsible for the legal and ethical side of passive RF reception
wherever you live. Don't publish traffic, node identities, or keys you don't
have the right to share.

The bundled Werkzeug server is fine for local single-user use. Reach for a
real WSGI server only if you're deliberately hosting for others.

## Performance notes

Performance scales with CPU bandwidth and core count. On a multi-core host
with a USB-3 SDR, expect:

- **Normal Meshtastic / LoRaWAN traffic**: real time at 28 Msps wideband
  across every SF / BW combination.
- **Dense back-to-back SF10 / SF11 bursts**: the gate's per-window detect
  work can briefly exceed real time on hosts with limited DRAM bandwidth.
  The receiver auto-throttles workers when sample drops show up. Real-world
  networks almost never approach this burst rate.
- **CPU affinity**: auto-isolated. The first four cores of the available
  affinity set are reserved for the gate's reader and detect pool; decode
  workers run on the rest. Wrap the launch in `taskset -c …` to override.
- **CPU governor**: on Linux, `powersave` introduces latency variance. For
  consistent measurements, switch to `performance`:

  ```bash
  sudo cpupower frequency-set -g performance
  ```

If your host is too slow for 28 MHz, drop the rate in `lora.toml`:

```toml
[radio]
rate_hz      = 14000000   # 14 Msps still covers half the US915 band
bandwidth_hz = 14000000
```

## Tuning

Common knobs in `lora.toml [detect]`, also exposed as CLI flags:

| Problem | Fix |
|---|---|
| False positives | raise `threshold` to 0.8 |
| Missed detections | lower `threshold` to 0.4 |
| Weak short bursts missed | lower `energy_threshold` from 12.0 to 5.0 |
| LO-leak false peak at center | `--dc-notch 0.5` |
| Adjacent-channel duplicates / saturation | lower SDR gain in `[radio]` |
| Dropped samples on slow hosts | raise `buf_seconds` from 16 to 32 |

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
├── README.md
├── LICENSE
├── install.sh
├── lora.toml          # all configuration (radio / detect / decode / web)
├── .env.example       # environment-variable overrides
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
├── docs/              # reference papers
├── lora_web/          # runtime state (settings, keys, autosave — gitignored)
└── captures/          # local capture output (gitignored)
```

## License

MIT. See `LICENSE`.
