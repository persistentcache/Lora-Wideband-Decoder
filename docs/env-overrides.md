# Environment-variable overrides

`lora.toml` is the primary configuration file. The env vars below are
optional runtime overrides — `export` them in your shell or prefix the
launch command:

```bash
LORA_DECODE_WORKERS=8 python3 run/web.py
```

Nothing auto-loads from a `.env` file; these vars must be present in
the process environment when `run/web.py` or `run/headless.py` starts.
Most users never need to touch any of these — defaults are tuned for
plug-and-play.

## Config file location

| Variable | Default | Purpose |
|---|---|---|
| `LORA_CONFIG` | (auto-locate) | Path to `lora.toml`. Overrides the cwd → project-root search. |

## Decode pipeline

| Variable | Default | Purpose |
|---|---|---|
| `LORA_DECODE_WORKERS` | auto-scale from cpu_count | Decode worker count; positive integer pins it. |
| `LORA_DECODE_BUDGET_S` | `10.0` | Per-capture fast-pass budget. Raise if SF11/12 bail with `[BUDGET]`. |
| `LORA_SLOW_BUDGET_S` | `45` | Slow-pass budget for re-decoded stragglers. |
| `LORA_COMMIT_LAG` | `4` | Detect-pool commit lag. Lower = lower latency; higher = more stable on slow disks. |
| `LORA_EXPORT_DEC` | `8` | Capture oversampling factor. `4` = 4 MHz / faster decode; `16` = 1 MHz / max fingerprint resolution. |
| `LORA_DEDUP_KEEP` | `2` | Duplicate captures per (carrier, RF-second). `1` = decode each packet once. |
| `LORA_FFT_WORKERS` | `-1` (all cores) | Internal FFT thread count. Pin lower on shared hosts. |

## Detection sensitivity

| Variable | Default | Purpose |
|---|---|---|
| `LORA_MAXHOLD` | `1` | Max-hold detection (low-SNR booster). `0` disables; reduces weak-distant-packet recovery. |
| `LORA_FINGERPRINT` | `1` | RF hardware fingerprinting (UMOP → device clustering). `0` saves ~10-15% decode CPU on low-core hosts. Header decode unchanged. |

## Protocol parsing

| Variable | Default | Purpose |
|---|---|---|
| `LORA_PROTOCOLS` | (all enabled) | Comma-separated list. Available: `meshtastic`, `meshcore`, `lorawan`, `loramesher`, `lora_aprs`, `reticulum`, `disaster_radio`, `ebyte_lora`, `radiohead`. Normally toggled via the web UI Advanced Options. |
| `LORA_UNKNOWN` | `0` | Surface unknown-protocol frames (unparsed but valid-looking LoRa). |
| `LORA_REGION` | `US915` | LoRaWAN region. Values: `US915`, `EU868`, `AS923`, `AU915`, `IN865`, `CN470`, `KR920`. |
| `LORA_GRID_TOL_KHZ` | `60` | LoRaWAN channel-grid tolerance. Raise on hosts with poor TCXO discipline. |
| `LORA_MC_CHANNEL_KEYS` | (empty) | MeshCore channel pre-shared keys (comma-separated hex/base64). |

## File paths (advanced — usually managed by lora_web automatically)

| Variable | Default | Purpose |
|---|---|---|
| `LORA_PKT_LOG` | `/tmp/lora_packets.jsonl` | Packet log the web UI tails. |
| `LORA_KEYS` | (none) | Multi-key file (Meshtastic PSK list, one per line). |
| `LORA_PSD_FILE` | (none) | Waterfall PSD shared-memory frame file. |
| `LORA_PSD_FPS` | `10` | Waterfall framerate. |
| `LORA_UNKNOWN_REPORT` | (none) | Unknown-protocol detail log. |

## Debug / diagnostics (rarely needed)

| Variable | Default | Purpose |
|---|---|---|
| `LORA_DETECT_SERIAL` | (unset) | Force serial detect-pool mode (easier profiler attach). |
| `LORA_SFD_DEBUG` | (unset) | SFD CFO refinement debug. Verbose. |
| `LORA_STUB_DETECT` | (unset) | Stub-mode detection (test harness only). |
| `LORA_SCAN_FULL` | (unset) | Full-spectrum scan mode flag. |
