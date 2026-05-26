"""
config.py — Single source of truth for LoRa parameter classes.

The CNN classifies detection (is it LoRa?) and SF (which spreading factor?).
BW is NOT classified by the CNN because all BWs produce identical spectrograms
when sampled at fs=2×BW. Instead, BW is determined by the detection pipeline:
it tests all candidate BWs and picks the one with highest detection confidence.
"""

# All spreading factors
SF_LIST = [7, 8, 9, 10, 11, 12]

# All bandwidths — used by the detection pipeline, NOT by the CNN
BW_LIST = [31250, 62500, 125000, 250000, 500000]

# --- CNN class mappings (detection + SF only) ---
# Index 0 = noise/no signal

SF_TO_CLASS = {0: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6}
CLASS_TO_SF = {0: 0, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12}
N_SF_CLASSES = 7   # noise + SF7–12
N_DET_CLASSES = 2  # no_signal, lora_present

# --- Meshtastic presets ---
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

def nb_sample_rate(bw):
    return bw * 2

def symbol_duration(sf, bw):
    return (2 ** sf) / bw

def symbol_samples(sf, bw, fs):
    return int(symbol_duration(sf, bw) * fs)
