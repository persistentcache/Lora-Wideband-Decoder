# lora_ml — wideband passive LoRa receiver

A self-hosted, single-user intercept receiver for LoRa traffic. Streams
IQ from your SDR, demodulates and decodes in software, surfaces
everything in a local web UI.

## I wanna use it!

Debian / Ubuntu:

```bash
./install.sh
python3 run/web.py          # opens http://127.0.0.1:5000
```

Open the URL it prints, hit **Start** in the UI, and intercepted
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

## Supported SDRs

| Device | Path | Notes |
|---|---|---|
| **bladeRF** | native `bladeRF-cli` | Validated default at 28 Msps. SoapyBladeRF fails at this rate; native CLI works. |
| **USRP B210 / B205mini** | SoapyUHD | Rated ~30 Msps over USB 3.0. Not author-validated but should sustain 28 Msps. |
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

## Docs

- [Identification & protocols](docs/identification.md) — confidence
  tiers, channel keys, Meshtastic preset reference.
- [Headless mode](docs/headless.md) — running without the web UI; SDR
  pipe examples.
- [Performance & tuning](docs/performance.md) — sample drops, CPU
  affinity, throughput knobs, `lora.toml` reference.
- [Environment-variable overrides](docs/env-overrides.md) — optional
  runtime knobs that override `lora.toml`.
- [Security & deployment](docs/security.md) — what to expose, what to
  isolate, what not to do.

## License

MIT. See `LICENSE`.
