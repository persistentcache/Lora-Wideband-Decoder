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

## Reporting issues

If something doesn't work, please include a diagnostic bundle so the failure
can be reproduced without guessing. Two ways to capture one:

1. **Standalone collector** (works even if the web UI never starts):

   ```bash
   python3 run/collect_debug.py            # writes /tmp/lora_debug_<ts>.txt
   python3 run/collect_debug.py --probe    # also runs a ~5 s capture test
   ```

   Attach the generated file to the GitHub issue.

2. **Verbose server**:

   ```bash
   python3 run/web.py --debug
   ```

   Prints a diagnostic dump on startup and streams the SDR capture
   subprocess output to the terminal (instead of only to
   `/tmp/lora_web_pipeline.log`). Copy everything the terminal prints.

Both modes scrub `$HOME` paths, hostname, IPs, and MAC addresses before
output. SDR serials and SoapySDR module versions are kept because they're
needed to diagnose hardware-specific bugs.

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
