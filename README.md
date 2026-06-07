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
| **bladeRF** | native `bladeRF-cli` | Recommended — only path that sustains 28 Msps. |
| HackRF, RTL-SDR, LimeSDR, USRP, PlutoSDR, Airspy, … | `src/soapy_rx.py` via SoapySDR | Any device with a SoapySDR module works. |

Named profiles live in `src/sdr_profiles.py`.

## Building from source

If `./install.sh` doesn't fit (non-Debian distro, custom Python), pull
the deps yourself:

```bash
pip install -r requirements.txt
# distro packages: soapysdr-tools, python3-soapysdr,
# soapysdr-module-{bladerf,hackrf,rtlsdr,...},
# libusb-1.0, libfftw3-dev, python3-dev
```

Python 3.11+ recommended. Older Python needs `pip install tomli` for
`lora.toml`.

## Docs

- [Identification & protocols](docs/identification.md) — confidence
  tiers, channel keys, Meshtastic preset reference.
- [Headless mode](docs/headless.md) — running without the web UI; SDR
  pipe examples.
- [Performance & tuning](docs/performance.md) — sample drops, CPU
  affinity, throughput knobs, `lora.toml` reference.
- [Security & deployment](docs/security.md) — what to expose, what to
  isolate, what not to do.

## License

MIT. See `LICENSE`.
