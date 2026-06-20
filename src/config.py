"""LoRa parameter constants used by the detector pipeline."""

# Bandwidths the detection pipeline tests as candidates.
#
# Per the SX1276/SX1262 datasheets, LoRa supports 10 standard bandwidths:
#   7.81, 10.42, 15.63, 20.83, 31.25, 41.67, 62.5, 125, 250, 500 kHz
# The ones included here are the bandwidths Meshtastic / MeshCore / LoRaWAN
# / LoRaMesher actually use in deployments — covering virtually all real
# LoRa-mesh traffic.  The smaller bandwidths (< 31 kHz) are valid LoRa
# but used only for extremely-low-rate / extreme-range FSK-style apps;
# they're not added by default because each additional BW in the list
# multiplies the per-window detector compute (the SC preamble lag scan
# runs once per (SF, BW) pair).  Users running such a deployment can add
# them here and accept the modest detection cost.
#
# 41.67 kHz is included for parity with experimental / custom LoRa setups
# (some long-range hobby deployments use this BW); standard mesh protocols
# default to ≥ 62.5 kHz.
BW_LIST = [31250, 41667, 62500, 125000, 250000, 500000]

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
