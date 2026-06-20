# LORA Wideband Decoder

A self-hosted, single-user intercept receiver for LoRa traffic on typical and non-typical frequencies. Streams
IQ from your SDR, demodulates and decodes in software near-real-time, surfaces
everything in a local web UI.

## I wanna use it!

Debian / Ubuntu:

```bash
./install.sh
python3 run/web.py          # opens http://127.0.0.1:5000
```

Open the URL it prints, configure your SDR in the Config tab, hit **Start** in the UI, and intercepted
packets stream in live. The pipeline (SDR capture + detector + decoder)
is launched from the web UI based on `lora.toml`.

If your SDR is already plugged in and you've rebooted (or re-logged in)
after the installer added you to `plugdev`, the UI's **Config →
SDR/Radio → Detect** finds it automatically.

## What it does

- Captures wideband IQ from any SoapySDR-compatible SDR or bladeRF.
- Decodes 9 LoRa protocols across SF7–SF12 at 6 standard bandwidths
  (see [Supported protocols](#supported-protocols-bandwidths--spreading-factors) below).
- Surfaces decoded packets, node identities, and a live spectrum
  waterfall in a local Flask web UI.
- Keeps up with real-world Meshtastic and LoRaWAN traffic in real time
  on a multi-core host at 28 Msps wideband.
- Will attempt to give you as much information as possible for LoRa protocols that have non-typical byte ordering, to allow for custom protocol implementation.

## Supported protocols, bandwidths & spreading factors

### Protocols

| Protocol | Highest tier reachable | How identified | Notes |
|---|---|---|---|
| **Meshtastic** | verified (with channel key) | AES-CCM decrypt + valid protobuf portnum | LongFast PSK (`AQ==`) bundled by default; supports PKI DMs |
| **MeshCore** | verified (with key) / confirmed (path-hash) / candidate | ADVERT Ed25519 sig OR HMAC-AES channel decrypt OR ECDH per-pair DM decrypt | Cross-worker ADVERT pubkey registry persists across pipeline restarts |
| **LoRaWAN** | candidate | MHDR + on-grid frequency check; MIC verify not implemented | Identifies type, DevAddr, FCnt for the UI |
| **LoRa APRS** | verified (structural) | 24-bit magic `0x3CFF01` + strict ASCII TNC2 grammar | Plaintext — no wire crypto |
| **Reticulum** | verified | Ed25519 ANNOUNCE signature verification | Full PKI announce decode |
| **disaster.radio** | confirmed | totalLength + TTL/hopcount sanity | Plaintext mesh, sudomesh recovery network |
| **LoRaMesher** | confirmed | type whitelist + payload_size match | 14 valid msg types, src/dst behavioral track |
| **EByte LoRa** | confirmed | fixed-format broadcast detection | Common Chinese E22/E220/E32 modules |
| **RadioHead** | confirmed | structural + repeat-sighting promotion | Hobby/research projects |
| **(other)** | unknown | falls through as `unknown · N bytes · sync 0xXX` | Surfaces if `LORA_UNKNOWN=1` env set |

Tier semantics:
- **verified** — cryptographically proven (AEAD/MAC decrypt OR in-frame signature verify) OR structural match so tight the FP rate is < ~1e-6 (LoRa APRS).
- **confirmed** — behavioral / structural identification with very low FP rate but no wire-level crypto check.
- **candidate** — structurally valid but unproven; treat with caution.

Each protocol can be enabled/disabled via the `LORA_PROTOCOLS` env var (default: all). Unknown-frame surfacing is opt-in via `LORA_UNKNOWN=1`.

### Spreading factors

| SF | Symbol rate (at 125 kHz) | Range trade-off |
|---|---|---|
| **SF7** | 6.83 kbps | fastest, shortest range |
| **SF8** | 3.91 kbps | |
| **SF9** | 2.20 kbps | |
| **SF10** | 1.22 kbps | |
| **SF11** | 0.67 kbps | |
| **SF12** | 0.37 kbps | slowest, longest range |

All six SFs are probed every detection window. The pipeline is SF-agnostic — adding or removing SFs is a one-line config change in `src/detector.py`.

### Bandwidths

| BW (kHz) | In default probe list | Common use |
|---|---|---|
| 7.81 | ❌ | extreme-range / FSK-style hobby apps |
| 10.42 | ❌ | extreme-range / FSK-style hobby apps |
| 15.63 | ❌ | extreme-range / FSK-style hobby apps |
| 20.83 | ❌ | extreme-range / FSK-style hobby apps |
| **31.25** | ✅ | LoRaWAN narrow regional plans |
| **41.67** | ✅ | custom long-range / non-mesh deployments |
| **62.5** | ✅ | Meshtastic VERY_LONG_SLOW, MeshCore typical |
| **125** | ✅ | LoRaWAN default, Meshtastic LONG_MODERATE / LONG_SLOW |
| **250** | ✅ | Meshtastic LongFast / MediumFast / ShortFast |
| **500** | ✅ | Meshtastic SHORT_TURBO |

The 4 smallest bandwidths (< 31 kHz) are valid Semtech-defined LoRa but no standard mesh protocol uses them. Enable them by editing `BW_LIST` in `src/config.py` — each additional BW multiplies the per-window Schmidl-Cox lag scan, so the detector compute scales linearly with the BW probe list.

### Wire-format presets recognized by name

When a frame matches a Meshtastic preset, the UI surfaces the preset name:

| Preset | SF | BW | CR |
|---|---|---|---|
| SHORT_TURBO | 7 | 500 kHz | 4/5 |
| SHORT_FAST | 7 | 250 kHz | 4/5 |
| SHORT_SLOW | 8 | 250 kHz | 4/5 |
| MEDIUM_FAST | 9 | 250 kHz | 4/5 |
| MEDIUM_SLOW | 10 | 250 kHz | 4/5 |
| LONG_FAST | 11 | 250 kHz | 4/5 |
| LONG_MODERATE | 11 | 125 kHz | 4/8 |
| LONG_SLOW | 12 | 125 kHz | 4/8 |
| VERY_LONG_SLOW | 12 | 62.5 kHz | 4/8 |

Other protocols (MeshCore / LoRaWAN / etc.) have their own SF/BW configurations the parser identifies by content rather than by named preset.

## Supported SDRs

| Device | Path | Notes |
|---|---|---|
| **bladeRF** | native `bladeRF-cli` | Validated default at 28 Msps. (SoapyBladeRF fails at this rate; native CLI works) |
| **USRP B210 / B205mini** | SoapyUHD | Rated ~30 Msps over USB 3.0. (Tested at 28MHz)|
| HackRF | SoapySDR | Capped at 20 Msps (hardware). |
| RTL-SDR, Airspy, LimeSDR, PlutoSDR, … | SoapySDR | Varies — see device specs. |

Named profiles + live capability probing live in `src/sdr_profiles.py`.

## Building from source

If `./install.sh` doesn't fit (non-Debian distro, custom Python):

```bash
pip install -r requirements.txt
# distro packages: soapysdr-tools, python3-soapysdr,
# soapysdr-module-{bladerf,hackrf,rtlsdr,...},
# libusb-1.0, libfftw3-dev, python3-dev
```

Requires Python 3.11+ for `lora.toml` (stdlib `tomllib`). Older Python
runs fine but falls back to coded defaults from `src/lora_config.py`.

### Running in a Python venv (optional)

If you prefer an isolated install (e.g. deploying to `/opt/lwd/`),
create the venv with `--system-site-packages` so it can reach the
apt-installed SoapySDR:

```bash
python3 -m venv --system-site-packages /opt/lwd/virtenv
source /opt/lwd/virtenv/bin/activate
pip install -r requirements.txt
```

**Why this flag is required**: `python3-soapysdr` ships as a SWIG-
generated C extension bound to the system python interpreter — it is
NOT pip-installable. Without `--system-site-packages`, `import SoapySDR`
from inside the venv fails even though `apt` reports it installed.
Other Python deps (numpy, scipy, flask, etc.) still pin cleanly inside
the venv; the flag only adds a fallback to system `dist-packages` for
modules the venv itself doesn't provide.

A venv created without the flag can't be retrofitted — delete and
recreate it. If `install.sh` is run inside an active venv, it detects
whether the flag is set and warns if it isn't.

Most users don't need a venv — `./install.sh` installs to the user site
via `pip --user` and the project runs without any activation step. Use
a venv only if you specifically want isolation or are deploying to a
controlled path.

## Reporting issues

If something isn't working — pipeline failure, crash, or a protocol that
isn't being recognised — please include the matching artifact below so the
problem can be reproduced without guessing.

1. **Verbose server (recommended for almost everything)**:

   ```bash
   python3 run/web.py --debug
   ```

   Starts the server normally and writes `lora_debug_<timestamp>.txt`
   to the project root, while also streaming the SDR capture
   subprocess output live to the terminal. Click **Start** in the UI
   and reproduce the issue; the bundle grows as the pipeline runs.
   Attach the file to the GitHub issue.

2. **Standalone collector** — use when the web UI won't start at all:

   ```bash
   python3 run/collect_debug.py
   ```

   Writes the same `lora_debug_<timestamp>.txt` to the project root —
   environment info, SoapySDR/SDR state, USB inventory, your config,
   plus a brief capture probe against your configured SDR. Attach the
   file to the GitHub issue.

3. **Unknown-protocol report** — use when frames are being received but
   not decoded (mystery devices, "no decodes" with traffic visible in
   the waterfall, suspected new/unsupported protocol):

   - In the web UI: open the **Advanced Options** panel and toggle
     **Unknown** on.
   - Click **Start** and let it run while the unknown traffic is
     active — a few minutes is usually enough, longer is better.
   - Scroll back to **Advanced Options → Unknown-protocol report**
     and click **Download**. That saves
     `lora_unknown_report.jsonl` — raw bytes plus PHY parameters
     (SF, BW, sync word, frequency) for every captured frame the
     decoder couldn't identify.
   - Attach the file to the GitHub issue.

All three modes scrub `$HOME` paths, hostname, IPs, and MAC addresses
before output; SDR serials and SoapySDR module versions are kept
because they are needed to diagnose hardware-specific bugs. The
unknown-protocol report contains only on-air bytes and PHY parameters
— no PII, no decoded content.

## Docs

- [Identification & protocols](docs/identification.md): how protocols
  are labeled, channel keys, Meshtastic preset list.
- [Headless mode](docs/headless.md): running it without the web UI.
- [Performance & tuning](docs/performance.md): throughput knobs and
  what to change when things drop samples.
- [Environment-variable overrides](docs/env-overrides.md): optional
  runtime overrides for `lora.toml`.
- [Security & deployment](docs/security.md): network exposure,
  privileges, hosting.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
Free for personal, research, educational, and other noncommercial use.
Not for commercial use or resale. See `LICENSE`.
