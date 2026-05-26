# lora_ml — wideband LoRa / Meshtastic intercept receiver

A passive, wideband (up to 28 MHz) bladeRF receiver that detects, decodes, and
**identifies** LoRa traffic — Meshtastic, LoRaWAN, and MeshCore — with a live
web UI. Detection is **Schmidl-Cox** (no ML model needed at runtime); decoding is
a multi-pass soft demod that reaches 100% on Meshtastic across SF7–SF12.

```
bladeRF IQ → energy gate (Welch + max-hold) → Schmidl-Cox preamble lock
           → dechirp + soft demod → CRC / decrypt / verify → web UI
```

## Protocol identification (no keys required)

A protocol is only **named** when there's discriminating evidence — never on a
structural guess. Confidence tiers:

| Tier | Meaning |
|---|---|
| **verified** | cryptographic proof — Meshtastic AES decrypt, MeshCore ADVERT Ed25519 signature, or MeshCore public-channel decrypt+MAC |
| **confirmed** | behavioral proof — LoRaWAN DevAddr + monotonic FCnt over ≥2 frames, or a MeshCore frame routed through ≥2 verified nodes |
| **unknown (+hint)** | structurally resembles a protocol but unconfirmed — shown honestly, never claimed |

MeshCore verified nodes (from ADVERTs) are exposed at `/api/mc_nodes`.

## Install

**Debian/Ubuntu — one command:**
```bash
./install.sh
```
Installs SoapySDR + per-device modules (universal SDR support — HackRF, RTL-SDR,
LimeSDR, USRP, PlutoSDR, Airspy, bladeRF), the native bladeRF/HackRF/RTL tools,
the Python requirements, and SDR device access (plugdev). Then run the web UI and
pick your SDR in **Config → SDR/Radio → Detect**.

**Manual / other OSes:**
```bash
pip install -r requirements.txt          # numpy, scipy, cryptography, flask
# + SoapySDR with its python3 bindings and your SDR's Soapy module
#   (or the native tool: bladeRF-cli / hackrf_transfer / rtl_sdr)
```
Python 3.11+ recommended (reads `lora.toml` via the stdlib; older Python: `pip install tomli`).
No paths are hardcoded — the program locates its own files relative to itself, so it
runs from any directory on any machine.

## Run

Edit **`lora.toml`** (`[radio] [detect] [decode] [web]`), then start the web UI —
it launches the bladeRF→detector pipeline for you:

```bash
python3 lora_web/lora_web.py            # serves on [web] host:port, reads lora.toml
```
Open the web UI to Start/Stop the receiver and watch the live feed, nodes,
network graph, waterfall, channel keys, and decoded packets.

**Manual pipeline (no web UI):**
```bash
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000;
    set bandwidth rx 28000000; set agc rx on;
    rx config file=/dev/stdout format=bin n=0; rx start; rx wait" \
| python3 scripts/lora_detect.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
    --decode --export-iq captures/

# Offline replay of a recording:
python3 scripts/lora_detect.py -f recording.sc16 -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```

## Channel keys

Manage decryption keys in the web **Config → Channel keys** tab (or `lora_keys.json`).
The two **Public/Default** keys (Meshtastic `AQ==`, MeshCore `8b33…`) are built in
and value-locked; add custom per-protocol keys as needed. LoRaWAN needs no key
(identified structurally + behaviorally).

## Security & deployment

This is a **self-hosted, single-user, local tool** — it needs a physical SDR
attached, so you run your own instance and open it in your own browser. Treat it
accordingly:

- **No authentication.** The web UI has no login. Anyone who can reach the port can
  start/stop the receiver, change settings, and read intercepted traffic + keys.
- **Binds to `127.0.0.1` (localhost) by default** — not reachable from other
  machines. To allow LAN access set `lora.toml [web] host = "0.0.0.0"`, but **only
  on a trusted network**, and ideally behind a reverse proxy (nginx/Caddy) that adds
  authentication + TLS. The app prints a warning when bound to a non-localhost
  address. Do **not** expose it directly to the public internet.
- **It runs the local SDR via shell commands.** Don't run it as root; a normal user
  in the `plugdev` (SDR) group is enough.
- **Legal / privacy.** This passively intercepts RF. You are responsible for
  complying with local laws on radio reception and privacy. Don't publish decoded
  traffic, node identities, or channel keys you don't own the rights to.

The bundled `app.run()` (Werkzeug) server is fine for this local single-user use; a
production WSGI server is only relevant if you deliberately host it for others.

## Tuning (`lora.toml [detect]` / CLI flags)

| Problem | Fix |
|---|---|
| False positives | `--threshold 0.8` |
| Missed detections | `--threshold 0.4` |
| Weak short bursts missed | `--energy-threshold` (default 5.0; max-hold is on) |
| LO-leak false peak at center | `--dc-notch 0.5` |
| Duplicates / ADC saturation | `--spur-reject 10`, or lower RX gain |
| Dropped samples on slow hosts | `--buf-seconds 16` |

## Meshtastic presets

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
├── install.sh             # one-command installer (Debian/Ubuntu)
├── lora.toml              # all configuration (radio / detect / decode / web)
├── requirements.txt       # Python runtime dependencies
├── scripts/               # CORE RUNTIME
│   ├── lora_detect.py        # energy gate + Schmidl-Cox detector + pipeline
│   ├── decode_header_v3.py   # multi-pass soft decoder + protocol parsers/verification
│   ├── detect_pool.py        # multiprocess detection pool
│   ├── config.py             # SF/BW/preset parameter classes
│   ├── lora_config.py        # lora.toml loader
│   ├── sdr_profiles.py       # SDR registry (capture/probe/limits per device)
│   └── soapy_rx.py           # SoapySDR → stdout IQ streamer (universal SDR capture)
├── lora_web/              # Flask web UI (lora_web.py + templates/) + runtime state
├── tools/                # preset_test.py (record→process→tally harness), sendtest.py
├── ml/                   # LEGACY CNN training (superseded by Schmidl-Cox)
├── dev/                  # dev/analysis scripts (analyze_capture.py, lora_watch.py)
├── docs/                 # reference papers
├── archive/              # old backups
├── captures/  recordings/  # scratch capture output
```
