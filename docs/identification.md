# Identification & protocols

## Confidence tiers

The receiver names a protocol only when it has discriminating evidence:

| Tier | Meaning |
|---|---|
| **verified** | cryptographic proof — Meshtastic AES decrypt, MeshCore ADVERT Ed25519 signature, or MeshCore public-channel decrypt + MAC |
| **confirmed** | behavioral proof — LoRaWAN DevAddr with monotonic FCnt over ≥2 frames, or a MeshCore frame routed through ≥2 verified nodes |
| **unknown + hint** | structurally resembles a protocol but unconfirmed — shown honestly, never claimed |

Meshtastic broadcasts on the public default channel decrypt with the
built-in PSK. Direct messages are PKI-encrypted, so they surface as
header-only entries — the link is visible, the payload is not.

## Channel keys

Manage decryption keys in the web UI's **Config → Channel keys** tab.
The public defaults for Meshtastic (`AQ==`) and MeshCore (`8b33…`) are
built in and value-locked. Add custom per-protocol keys as needed.
LoRaWAN doesn't need a key — identification is structural and
behavioral.

## Meshtastic preset reference

| Preset | SF | BW | CR |   | Preset | SF | BW | CR |
|---|---|---|---|---|---|---|---|---|
| SHORT_TURBO | 7 | 500k | 4/5 |   | LONG_FAST | 11 | 250k | 4/5 |
| SHORT_FAST | 7 | 250k | 4/5 |   | LONG_MODERATE | 11 | 125k | 4/8 |
| SHORT_SLOW | 8 | 250k | 4/5 |   | LONG_SLOW | 12 | 125k | 4/8 |
| MEDIUM_FAST | 9 | 250k | 4/5 |   | VERY_LONG_SLOW | 12 | 62.5k | 4/8 |
| MEDIUM_SLOW | 10 | 250k | 4/5 |   | | | | |
