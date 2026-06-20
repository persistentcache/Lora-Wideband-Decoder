# LORA Wideband Decoder

A self-hosted, single-user intercept receiver for LoRa traffic on typical and non-typical frequencies. Streams
IQ from your SDR, demodulates and decodes in software near-real-time, surfaces
everything in a local web UI.

## I wanna use it!

Debian / Ubuntu — venv install (recommended):

```bash
sudo apt install -y python3-venv python3-soapysdr
python3 -m venv --system-site-packages ~/lwd-venv
source ~/lwd-venv/bin/activate
./install.sh
python3 run/web.py          # opens http://127.0.0.1:5000
```

The `--system-site-packages` flag is required so the venv can reach
the apt-installed SoapySDR (which ships as a system-bound C extension,
not a pip wheel). All other Python deps stay pinned in the venv.

Open the URL it prints, configure your SDR in the Config tab, hit **Start** in the UI, and intercepted
packets stream in live. The pipeline (SDR capture + detector + decoder)
is launched from the web UI based on `lora.toml`.

If your SDR is already plugged in and you've rebooted (or re-logged in)
after the installer added you to `plugdev`, the UI's **Config →
SDR/Radio → Detect** finds it automatically.

**Subsequent runs**: re-activate the venv before launching:

```bash
source ~/lwd-venv/bin/activate
python3 run/web.py
```

### Why a venv?

Modern Debian/Ubuntu enforce [PEP 668](https://peps.python.org/pep-0668/)
(externally-managed environments), which blocks system-wide pip installs
to protect the distro's apt-managed Python. A venv sidesteps that cleanly,
keeps deps isolated per-project, and makes uninstall a single `rm -rf`.

If you'd rather not use a venv, see [Without a venv](#without-a-venv)
below — `./install.sh` still works and uses `pip --user --break-system-packages`,
but that path is increasingly fragile as distros tighten PEP 668 enforcement.

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

If `./install.sh` doesn't fit (non-Debian distro, custom Python), the
manual steps are:

```bash
# 1. distro packages (Debian/Ubuntu shown — adapt for your distro):
sudo apt install -y python3-venv python3-soapysdr soapysdr-tools \
    soapysdr-module-all libusb-1.0-0 libfftw3-dev python3-dev

# 2. venv with --system-site-packages so SoapySDR is reachable:
python3 -m venv --system-site-packages ~/lwd-venv
source ~/lwd-venv/bin/activate

# 3. Python deps:
pip install -r requirements.txt
```

Requires Python 3.11+ for `lora.toml` (stdlib `tomllib`). Older Python
runs fine but falls back to coded defaults from `src/lora_config.py`.

**Why `--system-site-packages` is required**: `python3-soapysdr` ships
as a SWIG-generated C extension bound to the system python interpreter
— it is NOT pip-installable. Without the flag, `import SoapySDR` from
inside the venv fails even though `apt` reports it installed. The flag
only adds a fallback to system `dist-packages` for modules the venv
itself doesn't provide — your pip-installed deps (numpy, scipy, flask,
etc.) still pin cleanly inside the venv.

A venv created without the flag can't be retrofitted — delete and
recreate it. If `install.sh` detects an active venv missing this flag,
it prints a warning before continuing.

### Without a venv

`./install.sh` also works without a venv — it falls back to
`pip install --user --break-system-packages` (the latter flag only
when supported and needed). This used to be the recommended path on
older distros, but PEP 668 enforcement on Debian 12+ / Ubuntu 23.04+
makes it increasingly fragile (system pip is blocked from touching
the apt-managed Python). Use a venv unless you have a specific reason
not to.

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
