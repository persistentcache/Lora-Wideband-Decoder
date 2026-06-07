"""LoRa parameter constants used by the detector pipeline."""

# Bandwidths the detection pipeline tests as candidates.
BW_LIST = [31250, 62500, 125000, 250000, 500000]

# Meshtastic preset → (SF, BW, CR) lookup, used by the protocol parser
# when surfacing a frame's detected preset in the web UI.
MESHTASTIC_PRESETS = {
    'SHORT_TURBO':   {'sf': 7,  'bw': 500000, 'cr': '4/5'},
    'SHORT_FAST':    {'sf': 7,  'bw': 250000, 'cr': '4/5'},
    'SHORT_SLOW':    {'sf': 8,  'bw': 250000, 'cr': '4/5'},
    'MEDIUM_FAST':   {'sf': 9,  'bw': 250000, 'cr': '4/5'},
    'MEDIUM_SLOW':   {'sf': 10, 'bw': 250000, 'cr': '4/5'},
    'LONG_FAST':     {'sf': 11, 'bw': 250000, 'cr': '4/5'},
    'LONG_MODERATE': {'sf': 11, 'bw': 125000, 'cr': '4/8'},
    'LONG_SLOW':     {'sf': 12, 'bw': 125000, 'cr': '4/8'},
    'VERY_LONG_SLOW':{'sf': 12, 'bw': 62500,  'cr': '4/8'},
}
