# Identification & protocols

The receiver only labels a frame with a protocol name when it has
evidence. There are three labels you'll see:

- `verified` means cryptographic proof: a Meshtastic AES decrypt, a
  MeshCore ADVERT with a valid Ed25519 signature, or a MeshCore
  public-channel decrypt with passing MAC.
- `confirmed` means behavioral proof. A LoRaWAN DevAddr with FCnt
  going up across two or more frames, or a MeshCore frame that's been
  routed through two or more nodes you've already verified.
- `unknown + hint` means it looks like a protocol structurally but
  nothing's been proven, so the hint is shown for what it is.

Meshtastic broadcasts on the public default channel decrypt with the
built-in PSK. Direct messages use PKI, so all you see for those is the
header — the link exists, but you can't read the payload.

## Channel keys

The web UI has a **Config → Channel keys** tab. The public defaults
for Meshtastic (`AQ==`) and MeshCore (`8b33…`) are built in. Add your
own keys there if you have them. LoRaWAN doesn't take a key at all —
identification is structural and behavioral.

## Meshtastic presets

| Preset | SF | BW | CR |   | Preset | SF | BW | CR |
|---|---|---|---|---|---|---|---|---|
| SHORT_TURBO | 7 | 500k | 4/5 |   | LONG_FAST | 11 | 250k | 4/5 |
| SHORT_FAST | 7 | 250k | 4/5 |   | LONG_MODERATE | 11 | 125k | 4/8 |
| SHORT_SLOW | 8 | 250k | 4/5 |   | LONG_SLOW | 12 | 125k | 4/8 |
| MEDIUM_FAST | 9 | 250k | 4/5 |   | VERY_LONG_SLOW | 12 | 62.5k | 4/8 |
| MEDIUM_SLOW | 10 | 250k | 4/5 |   | | | | |
