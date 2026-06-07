# Environment-variable overrides

`lora.toml` is what the receiver reads at startup. The env vars below
override individual values from that file at runtime. There's no
auto-loading from a `.env` file; export them in your shell or prefix
the launch:

```bash
LORA_DECODE_WORKERS=8 python3 run/web.py
```

Most people never set any of these. The defaults are fine.

## Config

| Variable | Default | Purpose |
|---|---|---|
| `LORA_CONFIG` | auto-locate | Path to `lora.toml`. Overrides the cwd → project-root search. |

## Decode

| Variable | Default | Purpose |
|---|---|---|
| `LORA_DECODE_WORKERS` | auto | Decode worker count. Positive integer pins it. |
| `LORA_DECODE_BUDGET_S` | `10.0` | Per-capture fast-pass budget. Raise if SF11/12 keep bailing with `[BUDGET]`. |
| `LORA_SLOW_BUDGET_S` | `45` | Budget for re-decoded stragglers. |
| `LORA_COMMIT_LAG` | `4` | Detect-pool commit lag. Lower for latency, higher for stability on slow disks. |
| `LORA_EXPORT_DEC` | `8` | Capture oversampling factor. `4` is faster, `16` keeps more fingerprint detail. |
| `LORA_DEDUP_KEEP` | `2` | Duplicate captures per (carrier, RF-second). `1` decodes each packet once. |
| `LORA_FFT_WORKERS` | `-1` | Internal FFT thread count. Pin lower on shared hosts. |

## Detection sensitivity

| Variable | Default | Purpose |
|---|---|---|
| `LORA_MAXHOLD` | `1` | Max-hold detection for low SNR. `0` disables; you'll miss the weakest distant packets. |
| `LORA_FINGERPRINT` | `1` | RF hardware fingerprinting (UMOP, device clustering). `0` saves ~10-15% decode CPU. Doesn't affect header decode. |

## Protocols

| Variable | Default | Purpose |
|---|---|---|
| `LORA_PROTOCOLS` | all | Comma-separated. Options: `meshtastic`, `meshcore`, `lorawan`, `loramesher`, `lora_aprs`, `reticulum`, `disaster_radio`, `ebyte_lora`, `radiohead`. The web UI's Advanced Options does the same thing. |
| `LORA_UNKNOWN` | `0` | Surface unparsed but valid-looking LoRa. |
| `LORA_REGION` | `US915` | LoRaWAN region. One of `US915`, `EU868`, `AS923`, `AU915`, `IN865`, `CN470`, `KR920`. |
| `LORA_GRID_TOL_KHZ` | `60` | LoRaWAN channel-grid tolerance. Raise on hosts with bad TCXO. |
| `LORA_MC_CHANNEL_KEYS` | empty | MeshCore PSKs, comma-separated hex/base64. |

## File paths

These are normally managed by `lora_web` automatically.

| Variable | Default | Purpose |
|---|---|---|
| `LORA_PKT_LOG` | `/tmp/lora_packets.jsonl` | Packet log the web UI tails. |
| `LORA_KEYS` | unset | Meshtastic PSK list, one per line. |
| `LORA_PSD_FILE` | unset | Waterfall PSD shared-memory frame file. |
| `LORA_PSD_FPS` | `10` | Waterfall framerate. |
| `LORA_UNKNOWN_REPORT` | unset | Unknown-protocol detail log. |

## Debug

You almost certainly don't need these.

| Variable | Purpose |
|---|---|
| `LORA_DETECT_SERIAL` | Forces serial detect-pool mode. Easier to attach a profiler. |
| `LORA_SFD_DEBUG` | Verbose SFD CFO output. |
| `LORA_STUB_DETECT` | Test harness only. |
| `LORA_SCAN_FULL` | Full-spectrum scan mode flag. |
