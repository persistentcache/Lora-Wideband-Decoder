# Identification & protocols

The receiver only labels a frame with a protocol name when it has
evidence. There are three confidence tiers you'll see in the **state**
column:

- **`verified`** — cryptographic proof: a Meshtastic AES-CCM decrypt,
  a MeshCore ADVERT with a valid Ed25519 signature, a MeshCore
  channel/DM decrypt with passing HMAC, or a Reticulum ANNOUNCE with
  a valid Ed25519 signature. LoRa-APRS reaches verified through its
  24-bit magic + strict TNC2 grammar (FP rate < ~1e-6).
- **`confirmed`** — behavioral proof. Examples: a MeshCore frame
  routed through one or more ADVERT-verified pubkeys (path-hash
  promotion), a RadioHead frame from a previously-seen sender, a
  LoRaMesher frame matching the msg-type whitelist + payload-size.
- **`candidate`** — structurally valid but unproven. Treat with
  caution. The `hint` badge (e.g. "encrypted DM (no key)") explains
  why the frame couldn't reach a higher tier.

Frames that don't match ANY parser surface as `unknown · N bytes ·
sync 0xXX` when `LORA_UNKNOWN=1` is set. The unknown-frame report
(downloadable from the Advanced Options panel) is the authoritative
log for diagnosing new/unsupported protocols.

## Supported protocols

| Protocol | Highest tier | How identified | Notes |
|---|---|---|---|
| **Meshtastic** | verified (w/ channel key) | AES-CCM decrypt + valid protobuf portnum | Default LongFast PSK (`AQ==`) bundled; supports PKI DMs when both endpoints' keys are loaded |
| **MeshCore** | verified / confirmed / candidate | ADVERT Ed25519 sig OR HMAC-AES channel decrypt OR ECDH per-pair DM decrypt OR path-hash promotion against ADVERT-verified pubkeys | Cross-worker ADVERT pubkey registry persists across pipeline restarts |
| **LoRaWAN** | candidate | MHDR + on-grid frequency check | Identifies type / DevAddr / FCnt for the UI; MIC verify not implemented |
| **LoRa APRS** | verified (structural) | 24-bit magic `0x3CFF01` + strict ASCII TNC2 grammar | Plaintext protocol — no wire crypto exists |
| **Reticulum** | verified | Ed25519 ANNOUNCE signature verification | Full PKI announce decode |
| **disaster.radio** | confirmed | totalLength byte + TTL/hop-count sanity | Plaintext mesh, sudomesh recovery network |
| **LoRaMesher** | confirmed | type whitelist + payload_size match | 14 valid msg types, src/dst behavioral track |
| **EByte LoRa** | confirmed | fixed-format broadcast detection | Common Chinese E22 / E220 / E32 modules |
| **RadioHead** | confirmed | structural + repeat-sighting promotion | Hobby / research projects |
| _(other)_ | unknown | falls through as `unknown · N bytes · sync 0xXX` | Surfaces only when `LORA_UNKNOWN=1` is set |

Each protocol can be enabled or disabled via the `LORA_PROTOCOLS`
environment variable (comma-separated). Default is all enabled.

## Channel keys

The web UI has a **Config → Channel keys** tab. The public defaults
for Meshtastic (`AQ==`) and MeshCore (`8b33…`) are built in. Add your
own keys there if you have them. LoRaWAN doesn't take a key at all —
identification is structural and behavioral.

Meshtastic broadcasts on the public default channel decrypt with the
built-in PSK. Direct messages use PKI, so for those you only see the
header — the link exists, but you can't read the payload unless you
have both endpoints' X25519 identity keys loaded.

## Spreading factors

The detector probes all six standard SFs every detection window. Each
SF doubles the symbol duration vs the next one down — slower data
rate, longer range, more sensitivity.

| SF | Symbol rate @ 125 kHz | Range trade-off |
|---|---|---|
| **SF7** | 6.83 kbps | fastest, shortest range |
| **SF8** | 3.91 kbps | |
| **SF9** | 2.20 kbps | |
| **SF10** | 1.22 kbps | |
| **SF11** | 0.67 kbps | |
| **SF12** | 0.37 kbps | slowest, longest range |

## Bandwidths

There are two layers:

- **Default scan** — the curated `_SC_LAGS_1M` table in
  `src/detector.py` defines which (SF, BW) combinations the
  Schmidl-Cox preamble detector tests every window. Tightly scoped to
  combinations standard mesh protocols actually use, to keep
  per-window compute minimal and avoid harmonic mis-resolution on
  shared lag buckets.
- **Wide scan (opt-in)** — toggle **Wide LoRa scan** in the web UI
  (or `LORA_SCAN_FULL=1` env var) to extend the lag table with every
  (SF × BW) combination from `BW_LIST` in `src/config.py`. Covers
  more deployments at the cost of more per-window compute and some
  increased risk of mis-resolution on shared lag buckets.

| BW (kHz) | Default scan | Wide scan | Common use |
|---|---|---|---|
| 7.81 | ❌ | ❌ | extreme-range / FSK-style hobby apps |
| 10.42 | ❌ | ❌ | extreme-range / FSK-style hobby apps |
| 15.63 | ❌ | ❌ | extreme-range / FSK-style hobby apps |
| 20.83 | ❌ | ❌ | extreme-range / FSK-style hobby apps |
| **31.25** | ❌ | ✅ | LoRaWAN narrow regional plans |
| **41.67** | ✅ | ✅ | tinyGS / satellite TT&C, long-range hobby |
| **62.5** | ✅ | ✅ | Meshtastic VERY_LONG_SLOW, MeshCore typical |
| **125** | ✅ | ✅ | LoRaWAN default, Meshtastic LONG_MODERATE / LONG_SLOW |
| **250** | ✅ | ✅ | Meshtastic LongFast / MediumFast / ShortFast |
| **500** | ✅ | ✅ | Meshtastic SHORT_TURBO |

To probe for any of the four smallest BWs (< 31 kHz), add them to
`BW_LIST` in `src/config.py` AND enable wide scan.

## Meshtastic presets

When a captured frame matches one of these SF/BW/CR triples, the UI
labels it with the Meshtastic preset name:

| Preset | SF | BW | CR | | Preset | SF | BW | CR |
|---|---|---|---|---|---|---|---|---|
| SHORT_TURBO | 7 | 500 kHz | 4/5 | | LONG_FAST | 11 | 250 kHz | 4/5 |
| SHORT_FAST | 7 | 250 kHz | 4/5 | | LONG_MODERATE | 11 | 125 kHz | 4/8 |
| SHORT_SLOW | 8 | 250 kHz | 4/5 | | LONG_SLOW | 12 | 125 kHz | 4/8 |
| MEDIUM_FAST | 9 | 250 kHz | 4/5 | | VERY_LONG_SLOW | 12 | 62.5 kHz | 4/8 |
| MEDIUM_SLOW | 10 | 250 kHz | 4/5 | | | | | |

Other protocols (MeshCore / LoRaWAN / LoRaMesher / etc.) have their
own SF/BW/CR configurations; the parser identifies those by frame
content rather than by a named preset.
