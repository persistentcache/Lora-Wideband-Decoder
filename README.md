# LoRa Wideband Decoder

**One SDR, the whole band, every LoRa conversation on it - live.**

Point a software-defined radio at a chunk of spectrum and this decodes
what's flying around in it: Meshtastic, MeshCore, and LoRaWAN, at every
spreading factor and bandwidth, on standard *and* non-standard
frequencies, all at the same time. No channel hopping, no preset
guessing, no scan mode - it watches up to 28 MHz at once and pulls out
every frame it can hear, in near-real-time, on a normal multi-core PC.

- **Wideband, not channelized** - one capture covers every channel in
  the in-band width at once; frequency lists are not a thing here.
- **Three protocols, zero configuration** - Meshtastic, MeshCore, and
  LoRaWAN are recognized and decoded automatically, including node
  identities and channel labeling.
- **All spreading factors, all bandwidths** - SF7 through SF12 at
  41.7 / 62.5 / 125 / 250 / 500 kHz, concurrently.
- **Live web UI** - decoded packets, node identities, and a spectrum
  waterfall stream into a local Flask app as they happen.
- **Unknown-protocol forensics** - frames that decode but do not match
  a known protocol are surfaced with raw bytes + PHY parameters so you
  can reverse your own.

## Quick install

Debian / Ubuntu:

```bash
./install.sh
python3 run/web.py          # opens http://127.0.0.1:5000
```

Open the URL it prints, pick your SDR in the Config tab (**Config →
SDR/Radio → Detect** finds it automatically if you have re-logged in
since the installer added you to `plugdev`), hit **Start**, and packets
stream in live. The whole pipeline - SDR capture, detector, decoder -
is launched from the web UI based on `lora.toml`.

## Supported SDRs

| Device | Path | Notes |
|---|---|---|
| **bladeRF** | native `bladeRF-cli` | Validated default at 28 Msps. (SoapyBladeRF fails at this rate; native CLI works) |
| **USRP B210 / B205mini** | SoapyUHD | Rated ~30 Msps over USB 3.0. (Tested at 28 MHz) |
| HackRF | SoapySDR | Capped at 20 Msps (hardware). |
| RTL-SDR, Airspy, LimeSDR, PlutoSDR, … | SoapySDR | Varies - see device specs. |

Named profiles + live capability probing live in `src/sdr_profiles.py`.

## System requirements

The one number that drives everything is the **sample rate**, which for a
complex SDR *is* the instantaneous bandwidth: 20 Msps = 20 MHz of spectrum
watched at once. The pipeline scans the *entire* sampled band in real time,
so **CPU and RAM both scale with the rate** - a wider view costs more of
both. Choose the rate to fit the host; the program sizes itself to whatever
it is given.

**What a host can sustain live (measured, over-the-air):**

| Host | Sustained rate / IBW | Notes |
|---|---|---|
| Desktop / laptop, many fast cores (dev host: 24-core x86, 64 GB) | **20-28 Msps / the full band** | validated default |
| Raspberry Pi 4 (4× Cortex-A72, 8 GB) | **~5 Msps / ~5 MHz** | 20 Msps overruns it (~75 % of the air dropped) |

Between those two the ceiling tracks **cores × per-core speed**. If a host
cannot keep up it degrades gracefully - the gate skips ahead to stay
real-time and decode is deferred - so you lose the dropped air but nothing
crashes or leaks memory.

**Minimums to run at all:**

- **OS** - 64-bit Linux (uses `/proc`, CPU affinity, `fork`, POSIX shared
  memory).
- **Software** - Python 3 with numpy + scipy (pyfftw is used automatically
  on ≤4-core hosts), plus an SDR capture tool (`hackrf_transfer`,
  `bladeRF-cli`, SoapySDR, …).
- **CPU** - any 64-bit multi-core, ARM or x86. More/faster cores → higher
  sustainable rate (table above).
- **RAM** - **~4 GB** at low rates; **8 GB for 20 Msps** (the ring buffer,
  detect-pool shared memory and gate working set all grow with the rate);
  **16 GB+** to run more decode workers in parallel for lower decode
  latency. The decode-worker count auto-caps to fit whatever RAM is present,
  so a small host is throttled, never OOM-killed.
- **SDR** - a complex-sampling radio covering your target bandwidth (see
  **Supported SDRs**), on **USB 3.0** for rates above ~8 Msps (20 Msps is
  ≈ 40 MB/s off the device).

**Rule of thumb:** a single LoRa channel needs only ~1 Msps, so *any* host
can watch one known channel. The cost is the **coverage** - watching the
whole band at once is what needs the CPU. Match the rate to the host: a Pi
does a narrow slice, a desktop does the whole band.

## Numbers worth knowing

**How weak can a signal be?** Measured with calibrated SNR ladders at
the shipped production settings (in-band SNR, i.e. relative to the
noise inside the packet's own bandwidth):

| Preset | Detection floor |
|---|---|
| SF7 / 125 kHz | 6 dB |
| SF7 / 500 kHz | 6 dB |
| SF9 / 125 kHz | 4 dB |
| SF10 / 250 kHz | 0 dB |
| SF11 / 125 kHz | 0 dB |
| SF12 / 125 kHz | 2 dB |

The slow spreading factors are caught **at the noise floor**, and once
a packet is detected the decoder holds on well below it - decode
ladders on real off-air captures stay at 100% down to **−8 dB in-band
SNR** before letting go. Treat the table as slightly optimistic bounds
(ladders are calibrated, but a real radio adds CFO drift and PA ramp;
the ladder battery was cross-validated against noise-injected off-air
recordings and matched within one step).

**Throughput reality check:** see **System requirements** above for what
each host class sustains in real time. Every release is regression-gated
against a corpus of real off-air recordings (overload, clipping, DC spurs,
two-node collisions, weak injections) plus live over-the-air legs against
real Meshtastic and MeshCore nodes.

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

Most users don't need a venv - `./install.sh` installs to the user site
via `pip --user` and the project runs without any activation step. If
you do want isolation (e.g. deploying to `/opt/lwd/`), create the venv
with `--system-site-packages` so it can reach the apt-installed
SoapySDR:

```bash
python3 -m venv --system-site-packages /opt/lwd/virtenv
source /opt/lwd/virtenv/bin/activate
pip install -r requirements.txt
```

**Why the flag matters**: `python3-soapysdr` is a SWIG-generated C
extension bound to the system interpreter - it is NOT pip-installable.
Without `--system-site-packages`, `import SoapySDR` fails inside the
venv even though `apt` says it's installed. A venv created without the
flag can't be retrofitted - delete and recreate it. (`install.sh`
checks for this and warns.)

## Reporting issues

If something isn't working - pipeline failure, crash, or a protocol
that isn't being recognised - please attach the matching artifact so
the problem can be reproduced without guessing.

1. **Verbose server** (right for almost everything):

   ```bash
   python3 run/web.py --debug
   ```

   Runs the server normally and writes `lora_debug_<timestamp>.txt` to
   the project root while streaming SDR subprocess output to the
   terminal. Click **Start**, reproduce the issue, attach the file.

2. **Standalone collector** - when the web UI won't start at all:

   ```bash
   python3 run/collect_debug.py
   ```

   Writes the same debug bundle (environment, SoapySDR/SDR state, USB
   inventory, config, a brief capture probe). Attach the file.

3. **Unknown-protocol report** - when frames are received but not
   decoded (mystery devices, traffic visible in the waterfall but no
   decodes): toggle **Unknown** on in **Advanced Options**, run a few
   minutes while the traffic is active, then **Advanced Options →
   Unknown-protocol report → Download**. That saves
   `lora_unknown_report.jsonl` - raw bytes plus PHY parameters (SF, BW,
   sync word, frequency) for every frame the decoder couldn't identify.

All three modes scrub `$HOME` paths, hostname, IPs, and MAC addresses
before output; SDR serials and SoapySDR module versions are kept
because they diagnose hardware-specific bugs. The unknown-protocol
report contains only on-air bytes and PHY parameters - no PII, no
decoded content.

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
