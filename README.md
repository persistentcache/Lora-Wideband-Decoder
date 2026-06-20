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
- Decodes Meshtastic, LoRaWAN, and MeshCore across SF7–SF12 at every
  standard bandwidth (62.5, 125, 250, 500 kHz).
- Surfaces decoded packets, node identities, and a live spectrum
  waterfall in a local Flask web UI.
- Keeps up with real-world Meshtastic and LoRaWAN traffic in real time
  on a multi-core host at 28 Msps wideband.
- Will attempt to give you as much information as possible for LORA protocols that has non-typical byte ordering to allow for custom protocol implementation.

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

### Running in a Python venv

If you prefer an isolated install (e.g. at `/opt/lwd/`), create the
venv with `--system-site-packages` so it can reach the apt-installed
SoapySDR:

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
recreate it. If `install.sh` detects an active venv missing this flag,
it prints a warning before continuing.

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
