#!/usr/bin/env python3
"""LoRa soft decoder v3 — full soft pipeline matching lora_stdin.cpp v5.16.

Pipeline: dechirp → FFT → max-log LLR → soft gray map → soft deinterleave → ML Hamming
+ fractional CFO correction via complex multiply (no drift/net_off extrapolation)
+ chase decoder (flip 2 least-confident nibbles on CRC fail)
+ Meshtastic packet decryption (AES-128-CTR) and protobuf decode

Run on .cf32 captures:
  python3 decoder.py captures/*.cf32
  python3 decoder.py -k NOKEY captures/*.cf32       # HAM mode (no encryption)
  python3 decoder.py -k '1PG7OiApB1nwvP+rz05pAQ==' captures/*.cf32  # custom key
"""
import numpy as np
import glob, sys, os, time
from itertools import combinations
try:
    from scipy.fft import (fft as _fft, ifft as _ifft, rfft as _rfft,
                           next_fast_len as _next_fast_len)
except ImportError:
    from numpy.fft import fft as _fft, ifft as _ifft, rfft as _rfft
    def _next_fast_len(n):
        return n


# ============================================================================
# Equivalence harness — when LORA_HARNESS_OUT=<path> is set, dumps a per-attempt
# fingerprint of decoder intermediate values to JSONL.  Zero overhead when unset.
# Used by tools/equiv_harness.py to validate that algorithmic optimisations are
# bit-identical / provably-equivalent BEFORE running expensive live tests.
# ============================================================================
_HARNESS_OUT = os.environ.get('LORA_HARNESS_OUT', '')
_HARNESS_FH = None
_HARNESS_ATTEMPT = [0]

def _harness_emit(event, **kw):
    if not _HARNESS_OUT:
        return
    global _HARNESS_FH
    if _HARNESS_FH is None:
        _HARNESS_FH = open(_HARNESS_OUT, 'a', buffering=1)
    import json as _json
    rec = {'event': event, 'attempt': _HARNESS_ATTEMPT[0], **kw}
    _HARNESS_FH.write(_json.dumps(rec, default=lambda o: str(o)) + '\n')


# ============================================================================
# Whitening sequence (from lora_stdin.cpp, matches gr-lora_sdr)
# ============================================================================
WHITENING_SEQ = bytes([
    0xff,0xfe,0xfc,0xf8,0xf0,0xe1,0xc2,0x85,0x0b,0x17,0x2f,0x5e,0xbc,0x78,0xf1,0xe3,
    0xc6,0x8d,0x1a,0x34,0x68,0xd0,0xa0,0x40,0x80,0x01,0x02,0x04,0x08,0x11,0x23,0x47,
    0x8e,0x1c,0x38,0x71,0xe2,0xc4,0x89,0x12,0x25,0x4b,0x97,0x2e,0x5c,0xb8,0x70,0xe0,
    0xc0,0x81,0x03,0x06,0x0c,0x19,0x32,0x64,0xc9,0x92,0x24,0x49,0x93,0x26,0x4d,0x9b,
    0x37,0x6e,0xdc,0xb9,0x72,0xe4,0xc8,0x90,0x20,0x41,0x82,0x05,0x0a,0x15,0x2b,0x56,
    0xad,0x5b,0xb6,0x6d,0xda,0xb5,0x6b,0xd6,0xac,0x59,0xb2,0x65,0xcb,0x96,0x2c,0x58,
    0xb0,0x61,0xc3,0x87,0x0f,0x1f,0x3e,0x7d,0xfb,0xf6,0xed,0xdb,0xb7,0x6f,0xde,0xbd,
    0x7a,0xf5,0xeb,0xd7,0xae,0x5d,0xba,0x74,0xe8,0xd1,0xa2,0x44,0x88,0x10,0x21,0x43,
    0x86,0x0d,0x1b,0x36,0x6c,0xd8,0xb1,0x63,0xc7,0x8f,0x1e,0x3c,0x79,0xf3,0xe7,0xce,
    0x9c,0x39,0x73,0xe6,0xcc,0x98,0x31,0x62,0xc5,0x8b,0x16,0x2d,0x5a,0xb4,0x69,0xd2,
    0xa4,0x48,0x91,0x22,0x45,0x8a,0x14,0x29,0x52,0xa5,0x4a,0x95,0x2a,0x54,0xa9,0x53,
    0xa7,0x4e,0x9d,0x3b,0x77,0xee,0xdd,0xbb,0x76,0xec,0xd9,0xb3,0x67,0xcf,0x9e,0x3d,
    0x7b,0xf7,0xef,0xdf,0xbf,0x7e,0xfd,0xfa,0xf4,0xe9,0xd3,0xa6,0x4c,0x99,0x33,0x66,
    0xcd,0x9a,0x35,0x6a,0xd4,0xa8,0x51,0xa3,0x46,0x8c,0x18,0x30,0x60,0xc1,0x83,0x07,
    0x0e,0x1d,0x3a,0x75,0xea,0xd5,0xaa,0x55,0xab,0x57,0xaf,0x5f,0xbe,0x7c,0xf9,0xf2,
    0xe5,0xca,0x94,0x28,0x50,0xa1,0x42,0x84,0x09,0x13,0x27,0x4f,0x9f,0x3f,0x7f,0x00,
])


# ============================================================================
# Gray mapping
# ============================================================================
def b2g(x):
    """Binary to Gray code (used on RX side to undo TX's g2b)."""
    return x ^ (x >> 1)


def circ_dist(a, b, N):
    """Circular distance between FFT bins."""
    d = abs(int(a) - int(b)) % int(N)
    return min(d, int(N) - d)


def circular_mean_bin(vals, N):
    """Circular mean of FFT bins, rounded back to an integer bin."""
    if not vals:
        return 0
    ang = 2.0 * np.pi * (np.asarray(vals, dtype=np.float64) % N) / float(N)
    z = np.exp(1j * ang).mean()
    if abs(z) < 1e-12:
        return int(np.median(np.asarray(vals, dtype=np.float64))) % int(N)
    return int(round((np.angle(z) % (2.0 * np.pi)) * float(N) / (2.0 * np.pi))) % int(N)


def ordered_unique_offsets(vals):
    """Unique integer offsets ordered by smallest absolute value first."""
    return sorted({int(v) for v in vals}, key=lambda x: (abs(x), x))


def ordered_unique_floats(vals, digits=4):
    """Unique float offsets ordered by smallest absolute value first."""
    seen = []
    for v in vals:
        rv = round(float(v), digits)
        if rv not in seen:
            seen.append(rv)
    return sorted(seen, key=lambda x: (abs(x), x))


# ============================================================================
# Soft FFT demodulation — max-log LLR per bit
# Matches lora_stdin.cpp soft_fft_demod exactly.
#
# Returns ppm LLR values. llr[b] = LLR for bit b (0=LSB, ppm-1=MSB).
# Positive LLR → bit more likely 1.
# ============================================================================
def soft_fft_demod(seg, downchirp, N, levels, ppm, bin_group, cfo_shift=0,
                    _precomp_mag_sq=None):
    """Soft demod: returns list of ppm floats (LLR per bit).
    cfo_shift: circular-shift FFT bins by this amount (= preamble_bin)
    to remove CFO without phase correction.

    `_precomp_mag_sq`: optional precomputed |FFT(seg*downchirp)|² for this
    symbol.  When supplied, the internal FFT is skipped — used by
    `decode_header_variant` to avoid doing the SAME FFT twice (once for
    PMR, once for soft LLR) on every header symbol of every timing
    variant.  Bit-identical math, just no duplicate FFT.
    """
    if _precomp_mag_sq is not None:
        mag_sq = _precomp_mag_sq
    else:
        # 1. Dechirp + FFT → magnitude squared per bin
        x = _fft(seg * downchirp)
        mag_sq = np.abs(x) ** 2  # use magnitude squared (= norm in C++)

    # Circular-shift to remove CFO (equivalent to subtracting preamble_bin)
    if cfo_shift != 0:
        mag_sq = np.roll(mag_sq, -cfo_shift)

    # 2. Magnitude per level = max within each bin_group (levels*bin_group == N
    # for every caller).  Vectorised reshape-max — byte-identical to the old
    # per-bin Python loop (mag_sq >= 0, groups never empty).
    mag = mag_sq[:levels * bin_group].reshape(levels, bin_group).max(axis=1).astype(np.float64)

    # 3. Gray-decoded value for each level (b2g(x) = x ^ (x>>1)).
    mask = levels - 1
    _lv = np.arange(levels, dtype=np.int64)
    base = ((_lv - 1 + levels) & mask) if bin_group == 1 else (_lv & mask)
    gvals = base ^ (base >> 1)

    # 4. Max-log LLR per bit: max mag over levels whose gray-bit b is 1, minus
    # the max over levels whose bit b is 0 (-1e30 sentinel when a side is empty,
    # matching the old nested loop exactly).
    bits = ((gvals[None, :] >> np.arange(ppm)[:, None]) & 1).astype(bool)  # (ppm,levels)
    magr = mag[None, :]
    llr = (np.where(bits, magr, -1e30).max(axis=1)
           - np.where(~bits, magr, -1e30).max(axis=1))

    return llr


def soft_fft_demod_batch(segs, dc, N, levels, ppm, bin_group, cfo_shift=0):
    """Vectorised soft_fft_demod over MANY symbols at once.

    `segs` is an (n, N) complex array (one symbol per row).  Returns an (n, ppm)
    LLR array.  Mathematically identical to calling soft_fft_demod per row, but a
    single batched FFT replaces n individual ones — removing scipy's ~6 µs/call
    uarray dispatch overhead (the dominant decode cost: ~210k tiny FFTs) and
    letting pocketfft thread across rows."""
    # Single-threaded batched FFT: the win is ONE scipy dispatch for n rows, not
    # threading — workers=-1 spins a thread pool per call, which costs far more
    # than it saves on tiny (N=128..4096) symbol FFTs.
    X = _fft(segs * dc, axis=1)
    mag_sq = np.abs(X) ** 2                                  # (n, N)
    if cfo_shift != 0:
        mag_sq = np.roll(mag_sq, -cfo_shift, axis=1)
    n = mag_sq.shape[0]
    mag = mag_sq[:, :levels * bin_group].reshape(n, levels, bin_group).max(axis=2).astype(np.float64)
    mask = levels - 1
    _lv = np.arange(levels, dtype=np.int64)
    base = ((_lv - 1 + levels) & mask) if bin_group == 1 else (_lv & mask)
    gvals = base ^ (base >> 1)
    bits = ((gvals[None, :] >> np.arange(ppm)[:, None]) & 1).astype(bool)   # (ppm, levels)
    magr = mag[:, None, :]                                   # (n, 1, levels)
    b = bits[None, :, :]                                     # (1, ppm, levels)
    llr = (np.where(b, magr, -1e30).max(axis=2)
           - np.where(~b, magr, -1e30).max(axis=2))          # (n, ppm)
    return llr


# ============================================================================
# Soft diagonal deinterleaver
# cw[k].bit[i] = sym[i].bit[ppm - 1 - ((i - k - 1) mod ppm)]
# Input: list of n_sym LLR arrays (each ppm floats)
# Output: list of ppm LLR arrays (each n_sym floats) — codewords
# ============================================================================
def soft_deinterleave(sym_llrs, n_sym, ppm):
    """Diagonal soft deinterleave: cw[k][i] = sym[i].bit[bp(k,i)].

    Vectorised over (k, i) with a single fancy-index gather.  Bit-identical
    to the original double loop (same modular index math, same MSB-first bp).
    """
    # sym_llrs may be a list of arrays (each ppm floats) or a (n_sym, ppm) matrix.
    M = np.asarray(sym_llrs, dtype=np.float64)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    # Precompute bp[k,i] = ppm - 1 - (((i-k-1) % ppm + ppm) % ppm)
    k = np.arange(ppm)[:, None]               # (ppm, 1)
    i = np.arange(n_sym)[None, :]             # (1, n_sym)
    idx = ((i - k - 1) % ppm + ppm) % ppm     # (ppm, n_sym)
    bp = ppm - 1 - idx                        # MSB-first column for each (k, i)
    # Gather: out[k, i] = M[i, bp[k, i]]
    out = M[i, bp]                            # (ppm, n_sym) via broadcasting
    return out


# ============================================================================
# Soft Hamming ML decode with confidence tracking
# Matches lora_stdin.cpp soft_hamming_ml_decode_ext.
# ============================================================================
def _hamming_codeword(d, rdd):
    """Generate valid codeword for data nibble d at coding rate rdd."""
    c = [0] * 8
    c[0] = (d >> 0) & 1
    c[1] = (d >> 1) & 1
    c[2] = (d >> 2) & 1
    c[3] = (d >> 3) & 1
    if rdd == 5:
        c[4] = c[0] ^ c[1] ^ c[2] ^ c[3]
    else:
        if rdd >= 5: c[4] = c[0] ^ c[1] ^ c[2]
        if rdd >= 6: c[5] = c[1] ^ c[2] ^ c[3]
        if rdd >= 7: c[6] = c[0] ^ c[1] ^ c[3]
        if rdd >= 8: c[7] = c[0] ^ c[2] ^ c[3]
    return c


# Precomputed ±1 sign vectors: _HAMMING_SIGNS[rdd][d] = tuple(2*codeword(d,rdd)[i]-1).
# The codewords are fixed (16 nibbles × 4 rates), so we build them ONCE instead
# of rebuilding via _hamming_codeword 7.3 M times in the inner chase loop.  A
# plain Python metric loop (not numpy) is kept — for a 16×8 op called ~450 k
# times the numpy/BLAS per-call dispatch overhead is LARGER than the loop, so a
# matmul is actually slower.  Byte-identical (same values, same summation order).
_HAMMING_SIGNS = {
    rdd: [tuple(2 * _hamming_codeword(d, rdd)[i] - 1 for i in range(8))
          for d in range(16)]
    for rdd in (5, 6, 7, 8)
}


def soft_hamming_decode(cw_llr, rdd):
    """ML Hamming decode from LLRs. Returns (best_nibble, second_nibble, margin)."""
    n = min(rdd, len(cw_llr))
    signs = _HAMMING_SIGNS.get(rdd)
    metrics = np.zeros(16)
    if signs is None:   # uncommon rdd — rebuild on the fly
        for d in range(16):
            c = _hamming_codeword(d, rdd)
            metric = 0.0
            for i in range(n):
                metric += cw_llr[i] * (2 * c[i] - 1)
            metrics[d] = metric
    else:
        for d in range(16):
            sd = signs[d]
            metric = 0.0
            for i in range(n):
                metric += cw_llr[i] * sd[i]
            metrics[d] = metric

    # Find best and second-best
    sorted_idx = np.argsort(metrics)[::-1]
    best = int(sorted_idx[0])
    second = int(sorted_idx[1])
    margin = metrics[best] - metrics[second]
    return best, second, margin


# Numpy-vectorised sign tables: (16, 8) per rdd, dtype float64 for stable
# dot products with arbitrary LLR magnitudes.  Built ONCE at import.
_HAMMING_SIGNS_NP = {
    rdd: np.array(_HAMMING_SIGNS[rdd], dtype=np.float64) for rdd in (5, 6, 7, 8)
}


def soft_hamming_decode_batch(cw_llr_list, rdd):
    """Vectorised ML Hamming decode over MANY codewords at once.

    Input: list/array of ppm codewords (each rdd LLRs).
    Output: (best_arr, second_arr, margin_arr) — three length-ppm arrays.

    Bit-identical to calling `soft_hamming_decode` per row (same metric sum,
    same argsort tie-breaks for the top two — numpy partial-sort would be
    faster but argsort matches the scalar path's behaviour, and the cost is
    16-wide which is trivial).

    Single numpy matmul replaces the per-codeword 16×rdd Python double loop
    (the dominant tottime contributor in the soft-payload pipeline).
    """
    signs = _HAMMING_SIGNS_NP.get(rdd)
    cw_mat = np.asarray(cw_llr_list, dtype=np.float64)
    # cw_mat: (ppm, rdd or longer).  Slice to rdd columns the table covers.
    n = min(rdd, cw_mat.shape[1] if cw_mat.ndim == 2 else len(cw_mat))
    if cw_mat.ndim == 1:
        cw_mat = cw_mat[None, :]
    cw_n = cw_mat[:, :n]
    if signs is None:
        # uncommon rdd — fall back to per-row scalar decode
        out_best, out_second, out_margin = [], [], []
        for row in cw_mat:
            b, s, m = soft_hamming_decode(row, rdd)
            out_best.append(b); out_second.append(s); out_margin.append(m)
        return (np.array(out_best, dtype=np.int64),
                np.array(out_second, dtype=np.int64),
                np.array(out_margin, dtype=np.float64))
    metrics = cw_n @ signs[:, :n].T                # (ppm, 16)
    # argsort descending — match the scalar path's tie-break exactly.
    sorted_idx = np.argsort(-metrics, axis=1, kind='stable')
    best = sorted_idx[:, 0].astype(np.int64)
    second = sorted_idx[:, 1].astype(np.int64)
    rows = np.arange(metrics.shape[0])
    margin = metrics[rows, best] - metrics[rows, second]
    return best, second, margin


# ============================================================================
# Hard Hamming ML decode (for header, from integer codeword)
# ============================================================================
def hamming_ml_hard(cw_byte, rdd):
    """ML Hamming decode from hard bits. Returns best nibble."""
    rx_bits = [(cw_byte >> i) & 1 for i in range(8)]
    best_d, best_dist = 0, 999
    for d in range(16):
        c = _hamming_codeword(d, rdd)
        dist = sum(c[i] ^ rx_bits[i] for i in range(rdd))
        if dist < best_dist:
            best_dist = dist
            best_d = d
    return best_d


# ============================================================================
# Header parsing
# ============================================================================
def parse_header(nibs):
    if len(nibs) < 5:
        return None
    payload_len = (nibs[0] << 4) | nibs[1]
    cr = (nibs[2] >> 1) & 0x7
    crc_present = (nibs[2] & 1) != 0
    cr = max(1, min(4, cr))

    bits = []
    for i in range(3):
        for j in range(4):
            bits.append((nibs[i] >> (3 - j)) & 1)
    H = [
        [1,1,1,1,0,0,0,0,0,0,0,0], [1,0,0,0,1,1,1,0,0,0,0,1],
        [0,1,0,0,1,0,0,1,1,0,1,0], [0,0,1,0,0,1,0,1,0,1,1,1],
        [0,0,0,1,0,0,1,0,1,1,1,1],
    ]
    calc_bits = []
    for r in range(5):
        val = 0
        for j in range(12):
            val ^= (H[r][j] & bits[j])
        calc_bits.append(val)
    calc = (calc_bits[0] << 4) | (calc_bits[1] << 3) | (calc_bits[2] << 2) | (calc_bits[3] << 1) | calc_bits[4]
    got = ((nibs[3] & 1) << 4) | (nibs[4] & 0xF)
    return payload_len, cr, crc_present, (got == calc)


# ============================================================================
# CRC-16
# ============================================================================
def _crc16_table(poly):
    """256-entry table for an MSB-first (non-reflected) CRC-16 with `poly`."""
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
        table.append(crc)
    return table


_CRC16_CCITT_TABLE = _crc16_table(0x1021)
_CRC16_IBM_TABLE = _crc16_table(0x8005)


def crc16_ccitt(data):
    # Table-driven equivalent of the MSB-first bit loop (8× fewer iterations,
    # byte-identical — verified exhaustively against the original).
    crc = 0
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_CCITT_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


def crc16_ibm(data):
    crc = 0
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_IBM_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


def check_crc(raw_bytes, payload_len, strict=False):
    """Check LoRa payload CRC.

    strict=True:  Only try xor-CCITT-LE (the known LoRa/Meshtastic CRC).
                  Use this in sweep/chase to avoid false positives.
    strict=False: Also try alternative polynomials/byte orders (for research).
    """
    if len(raw_bytes) < payload_len + 2:
        return False, ""
    payload = raw_bytes[:payload_len]
    got_le = raw_bytes[payload_len] | (raw_bytes[payload_len + 1] << 8)

    # ---- Known correct LoRa CRC: xor-CCITT-LE ----
    # CRC-16/CCITT of payload[:-2], XORed with last 2 payload bytes, stored LE
    if payload_len >= 2:
        final_xor = (payload[-2] << 8) | payload[-1]
        xor_crc = crc16_ccitt(payload[:-2]) ^ final_xor
        if xor_crc == got_le:
            return True, "xor-CCITT-LE"

    if strict:
        return False, "FAIL"

    # ---- Fallback: try other variants (higher false-positive risk) ----
    got_be = (raw_bytes[payload_len] << 8) | raw_bytes[payload_len + 1]

    for name, func in [("CCITT", crc16_ccitt), ("IBM", crc16_ibm)]:
        std = func(payload)
        if std == got_le: return True, "std-%s-LE" % name
        if std == got_be: return True, "std-%s-BE" % name

    if payload_len >= 2:
        final_xor = (payload[-2] << 8) | payload[-1]
        for name, func in [("CCITT", crc16_ccitt), ("IBM", crc16_ibm)]:
            xor_crc = func(payload[:-2]) ^ final_xor
            if xor_crc == got_be: return True, "xor-%s-BE" % name
            # IBM-LE variant
            if func != crc16_ccitt:
                if func(payload[:-2]) ^ final_xor == got_le:
                    return True, "xor-%s-LE" % name

    return False, "FAIL"


# ============================================================================
# AES-128 implementation (for Meshtastic decryption)
# ============================================================================
AES_SBOX = bytes([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
])

AES_RCON = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

def aes128_expand_key(key):
    rk = bytearray(176)
    rk[:16] = key[:16]
    for i in range(4, 44):
        t = list(rk[(i-1)*4:(i-1)*4+4])
        if i % 4 == 0:
            tmp = t[0]
            t[0] = AES_SBOX[t[1]] ^ AES_RCON[i//4 - 1]
            t[1] = AES_SBOX[t[2]]
            t[2] = AES_SBOX[t[3]]
            t[3] = AES_SBOX[tmp]
        for j in range(4):
            rk[i*4+j] = rk[(i-4)*4+j] ^ t[j]
    return bytes(rk)

def aes128_encrypt_block(rk, inp):
    s = bytearray(inp[i] ^ rk[i] for i in range(16))
    for rnd in range(1, 11):
        # SubBytes
        t = bytearray(AES_SBOX[s[i]] for i in range(16))
        # ShiftRows
        u = bytearray(16)
        u[0],u[1],u[2],u[3]    = t[0],t[5],t[10],t[15]
        u[4],u[5],u[6],u[7]    = t[4],t[9],t[14],t[3]
        u[8],u[9],u[10],u[11]  = t[8],t[13],t[2],t[7]
        u[12],u[13],u[14],u[15]= t[12],t[1],t[6],t[11]
        if rnd < 10:
            # MixColumns
            def xtime(x):
                return ((x << 1) ^ (0x1b if (x >> 7) & 1 else 0)) & 0xFF
            for c in range(4):
                a = [u[c*4+j] for j in range(4)]
                s[c*4+0] = xtime(a[0]) ^ xtime(a[1]) ^ a[1] ^ a[2] ^ a[3]
                s[c*4+1] = a[0] ^ xtime(a[1]) ^ xtime(a[2]) ^ a[2] ^ a[3]
                s[c*4+2] = a[0] ^ a[1] ^ xtime(a[2]) ^ xtime(a[3]) ^ a[3]
                s[c*4+3] = xtime(a[0]) ^ a[0] ^ a[1] ^ a[2] ^ xtime(a[3])
        else:
            s = bytearray(u)
        # AddRoundKey
        for i in range(16):
            s[i] ^= rk[rnd*16+i]
    return bytes(s)

def aes128_ctr_decrypt(key, nonce, ciphertext):
    rk = aes128_expand_key(key)
    ctr = bytearray(nonce)
    plaintext = bytearray(len(ciphertext))
    for off in range(0, len(ciphertext), 16):
        ks = aes128_encrypt_block(rk, bytes(ctr))
        blk = min(16, len(ciphertext) - off)
        for i in range(blk):
            plaintext[off+i] = ciphertext[off+i] ^ ks[i]
        # Increment counter (big-endian, last 4 bytes)
        for i in range(15, -1, -1):
            ctr[i] = (ctr[i] + 1) & 0xFF
            if ctr[i] != 0:
                break
    return bytes(plaintext)


# --- Generalised AES (128/192/256) for multi-key custom channel decryption ---
def aes_expand_key(key):
    """Key schedule for AES-128/192/256.  Returns (round_keys, num_rounds)."""
    Nk = len(key) // 4          # 4, 6, 8
    Nr = Nk + 6                 # 10, 12, 14
    nwords = 4 * (Nr + 1)
    rk = bytearray(nwords * 4)
    rk[:len(key)] = key
    for i in range(Nk, nwords):
        t = list(rk[(i - 1) * 4:(i - 1) * 4 + 4])
        if i % Nk == 0:
            tmp = t[0]
            t[0] = AES_SBOX[t[1]] ^ AES_RCON[i // Nk - 1]
            t[1] = AES_SBOX[t[2]]; t[2] = AES_SBOX[t[3]]; t[3] = AES_SBOX[tmp]
        elif Nk > 6 and i % Nk == 4:
            t = [AES_SBOX[x] for x in t]
        for j in range(4):
            rk[i * 4 + j] = rk[(i - Nk) * 4 + j] ^ t[j]
    return bytes(rk), Nr


def aes_encrypt_block(rk, Nr, inp):
    s = bytearray(inp[i] ^ rk[i] for i in range(16))
    for rnd in range(1, Nr + 1):
        t = bytearray(AES_SBOX[s[i]] for i in range(16))
        u = bytearray(16)
        u[0], u[1], u[2], u[3]      = t[0], t[5], t[10], t[15]
        u[4], u[5], u[6], u[7]      = t[4], t[9], t[14], t[3]
        u[8], u[9], u[10], u[11]    = t[8], t[13], t[2], t[7]
        u[12], u[13], u[14], u[15]  = t[12], t[1], t[6], t[11]
        if rnd < Nr:
            def xtime(x):
                return ((x << 1) ^ (0x1b if (x >> 7) & 1 else 0)) & 0xFF
            for c in range(4):
                a = [u[c*4+j] for j in range(4)]
                s[c*4+0] = xtime(a[0]) ^ xtime(a[1]) ^ a[1] ^ a[2] ^ a[3]
                s[c*4+1] = a[0] ^ xtime(a[1]) ^ xtime(a[2]) ^ a[2] ^ a[3]
                s[c*4+2] = a[0] ^ a[1] ^ xtime(a[2]) ^ xtime(a[3]) ^ a[3]
                s[c*4+3] = xtime(a[0]) ^ a[0] ^ a[1] ^ a[2] ^ xtime(a[3])
        else:
            s = bytearray(u)
        for i in range(16):
            s[i] ^= rk[rnd*16+i]
    return bytes(s)


def aes_ctr_decrypt(key, nonce, ciphertext):
    """AES-CTR for any key length (16/24/32 bytes)."""
    rk, Nr = aes_expand_key(key)
    ctr = bytearray(nonce)
    out = bytearray(len(ciphertext))
    for off in range(0, len(ciphertext), 16):
        ks = aes_encrypt_block(rk, Nr, bytes(ctr))
        blk = min(16, len(ciphertext) - off)
        for i in range(blk):
            out[off+i] = ciphertext[off+i] ^ ks[i]
        for i in range(15, -1, -1):
            ctr[i] = (ctr[i] + 1) & 0xFF
            if ctr[i] != 0:
                break
    return bytes(out)


# Default Meshtastic AES key: base64 "1PG7OiApB1nwvP+rz05pAQ=="
MESH_AES_KEY = bytes([
    0xd4,0xf1,0xbb,0x3a,0x20,0x29,0x07,0x59,
    0xf0,0xbc,0xff,0xab,0xcf,0x4e,0x69,0x01,
])


# --- Multi-key channel decryption ---------------------------------------
# The web's Config tab manages a key list (LORA_KEYS json file):
#   {"try_default": true, "default_priority": 50,
#    "default_nodes": [],          # [] = default key applies to ALL nodes;
#                                   #      a non-empty list = default only for those
#    "keys": [{"label":..., "key":"<base64|hex>", "scope":"all"|"nodes",
#              "nodes":["!id", ...], "priority": 100}]}   # may target MULTIPLE nodes
# For a packet from sender S, every APPLICABLE key (an "all"-scope key, or one that
# lists S; the default key if enabled and default_nodes is empty/lists S) is tried in
# ascending `priority` order (lower number = tried first), so the user can rank e.g. an
# "all"-nodes custom key ahead of the default.  First valid decode wins.  Defaults when
# priority is absent: node-scoped=10, default=50, "all"-scoped=100 → preserves the
# original node-specific → default → all-custom ordering out of the box.
# If try_default is off (or default is node-restricted and S isn't listed) AND
# no custom key applies, the packet is NOT decrypted (header-only / encrypted).
# Back-compat: a legacy {"scope":"node","node":"!id"} is read as nodes=["!id"].
# Re-read on file mtime change → live updates.
_KEYS_CACHE = {'mtime': None, 'cfg': None}


def _decode_key_str(s):
    """Parse a channel key string (base64 or hex) → 16/24/32 raw bytes, or None."""
    import base64
    s = (s or '').strip()
    if not s:
        return None
    try:
        h = s.lower()
        if all(c in '0123456789abcdef' for c in h) and len(h) in (32, 48, 64):
            return bytes.fromhex(h)
    except Exception:
        pass
    try:
        b = base64.b64decode(s + '=' * (-len(s) % 4))
        if len(b) in (16, 24, 32):
            return b
    except Exception:
        pass
    return None


def _load_keys():
    path = os.environ.get('LORA_KEYS')
    if not path or not os.path.exists(path):
        return None
    try:
        m = os.path.getmtime(path)
        if m != _KEYS_CACHE['mtime']:
            import json as _json
            with open(path) as f:
                cfg = _json.load(f)
            for k in cfg.get('keys', []):
                k['_bytes'] = _decode_key_str(k.get('key', ''))
            _KEYS_CACHE['mtime'] = m
            _KEYS_CACHE['cfg'] = cfg
        return _KEYS_CACHE['cfg']
    except Exception:
        return None


def _key_nodes(k):
    """A key's target node list (supports legacy single 'node')."""
    ns = k.get('nodes')
    if ns is None and k.get('node'):
        ns = [k['node']]
    return ns or []


def _ordered_keys(cfg, sender):
    """Applicable (key_bytes, label) for `sender`, ordered by user priority (low = tried
    first).  A key applies if it is "all"-scope or lists the sender; the default key
    applies if enabled and its default_nodes is empty or lists the sender.  A missing
    priority defaults so that node-scoped keys precede the default precede "all"-scope
    keys (preserves the pre-priority ordering for keys saved without an explicit one)."""
    cands = []                            # (priority, tiebreak, key_bytes, label)
    for i, k in enumerate(cfg.get('keys', [])):
        if not k.get('_bytes'):
            continue
        if k.get('protocol', 'meshtastic') == 'meshcore':   # MeshCore keys handled separately
            continue
        if k.get('enabled') is False:                        # explicitly disabled in the UI
            continue
        if k.get('scope') == 'all' or sender in _key_nodes(k):
            prio = k.get('priority')
            if prio is None:
                prio = 100 if k.get('scope') == 'all' else 10
            cands.append((prio, i, k['_bytes'], k.get('label', 'custom')))
    if cfg.get('try_default', True):
        dn = cfg.get('default_nodes') or []
        if not dn or sender in dn:
            cands.append((cfg.get('default_priority', 50), -1, MESH_AES_KEY, 'default'))
    cands.sort(key=lambda c: (c[0], c[1]))
    return [(c[2], c[3]) for c in cands]

# Set of (pkt_id, hops_taken) decoded during the current top-level process_file
# call.  Used by the carrier rescue to stop as soon as it recovers a NEW hop
# (instead of running the full shift sweep).  Reset per top-level call.
_DECODED_SET = set()
# Per-capture hardware fingerprint, set by process_file ONCE on the unshifted
# IQ before any recenter pass.  Read by _decode_attempt and attached to every
# emitted [PKT] record (so all decode-pass outcomes share the same per-device
# RF fingerprint).
_CURRENT_HW_FP = [None]
_CURRENT_PRECISE_CFO = [None]



# ============================================================================
# Minimal protobuf varint + field parser
# ============================================================================
def parse_protobuf(data):
    """Parse protobuf fields. Returns list of (field_num, wire_type, value)."""
    fields = []
    pos = 0
    length = len(data)
    while pos < length:
        # Read tag varint
        tag = 0; shift = 0
        while pos < length:
            b = data[pos]; pos += 1
            tag |= (b & 0x7f) << shift
            shift += 7
            if not (b & 0x80): break
        field_num = tag >> 3
        wire_type = tag & 7

        if wire_type == 0:  # varint
            val = 0; shift = 0
            while pos < length:
                b = data[pos]; pos += 1
                val |= (b & 0x7f) << shift
                shift += 7
                if not (b & 0x80): break
            fields.append((field_num, wire_type, val))
        elif wire_type == 2:  # length-delimited
            flen = 0; shift = 0
            while pos < length:
                b = data[pos]; pos += 1
                flen |= (b & 0x7f) << shift
                shift += 7
                if not (b & 0x80): break
            if pos + flen <= length:
                fields.append((field_num, wire_type, bytes(data[pos:pos+flen])))
                pos += flen
            else:
                break  # truncated
        elif wire_type == 5:  # 32-bit fixed
            if pos + 4 <= length:
                val = data[pos] | (data[pos+1]<<8) | (data[pos+2]<<16) | (data[pos+3]<<24)
                fields.append((field_num, wire_type, val))
                pos += 4
            else:
                break
        elif wire_type == 1:  # 64-bit fixed
            pos += 8
        else:
            break
    return fields


# ============================================================================
# Portnum names (from meshtastic protobuf)
# ============================================================================
PORTNUM_NAMES = {
    0: "UNKNOWN_APP", 1: "TEXT_MESSAGE_APP", 3: "POSITION_APP",
    4: "NODEINFO_APP", 5: "ROUTING_APP", 6: "ADMIN_APP",
    7: "TEXT_MESSAGE_COMPRESSED_APP", 8: "WAYPOINT_APP", 9: "AUDIO_APP",
    10: "DETECTION_SENSOR_APP", 32: "REPLY_APP", 33: "IP_TUNNEL_APP",
    34: "PAXCOUNTER_APP", 64: "SERIAL_APP", 65: "STORE_FORWARD_APP",
    66: "RANGE_TEST_APP", 67: "TELEMETRY_APP", 68: "ZPS_APP",
    69: "SIMULATOR_APP", 70: "TRACEROUTE_APP", 71: "NEIGHBORINFO_APP",
    72: "ATAK_PLUGIN", 73: "MAP_REPORT_APP",
}


def portnum_name(pn):
    return PORTNUM_NAMES.get(pn, "UNKNOWN(%d)" % pn)


# Real Meshtastic PortNums — used to validate a decrypt (random bytes from a wrong
# key rarely land on one of these).  Shared by the live decode + web retry paths.
KNOWN_PORTNUMS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 33, 34, 64, 65, 66,
                  67, 68, 69, 70, 71, 72, 73, 74, 256, 257}


def node_id_str(p):
    """Swap byte order of 4-byte field for display as Meshtastic node ID."""
    return "!%02x%02x%02x%02x" % (p[3], p[2], p[1], p[0])


# ============================================================================
# Decode specific Meshtastic protobuf payloads
# ============================================================================
import struct

HW_MODEL_NAMES = {
    0: "UNSET", 1: "TLORA_V2", 2: "TLORA_V1", 3: "TLORA_V2_1_1P6", 4: "TBEAM",
    5: "HELTEC_V2_0", 6: "TBEAM_V0P7", 7: "T_ECHO", 8: "TLORA_V1_1P3",
    9: "RAK4631", 10: "HELTEC_V2_1", 11: "HELTEC_V1", 12: "LILYGO_TBEAM_S3_CORE",
    13: "RAK11200", 14: "NANO_G1", 15: "TLORA_V2_1_1P8", 16: "TLORA_T3_S3",
    17: "NANO_G1_EXPLORER", 18: "NANO_G2_ULTRA", 19: "LORA_TYPE",
    20: "WIPHONE", 21: "WIO_WM1110", 22: "RAK2560", 23: "HELTEC_HRU_3601",
    24: "HELTEC_WIRELESS_BRIDGE", 25: "STATION_G1", 26: "RAK11310",
    27: "SENSELORA_RP2040", 28: "SENSELORA_S3", 29: "CANARYONE",
    30: "RP2040_LORA", 31: "STATION_G2", 32: "LORA_RELAY_V1",
    33: "T_ECHO_PLUS", 34: "PPR", 35: "GENIEBLOCKS", 36: "NRF52_UNKNOWN",
    37: "PORTDUINO", 38: "ANDROID_SIM", 39: "DIY_V1", 40: "NRF52840_PCA10059",
    41: "DR_DEV", 42: "M5STACK", 43: "HELTEC_V3", 44: "HELTEC_WSL_V3",
    45: "BETAFPV_2400_TX", 46: "BETAFPV_900_NANO_TX", 47: "RPI_PICO",
    48: "HELTEC_WIRELESS_TRACKER", 49: "HELTEC_WIRELESS_PAPER",
    50: "T_DECK", 51: "T_WATCH_S3", 52: "PICOMPUTER_S3", 53: "HELTEC_HT62",
    54: "EBYTE_ESP32_S3", 55: "ESP32_S3_PICO", 56: "CHATTER_2",
    57: "HELTEC_WIRELESS_PAPER_V1_0", 58: "HELTEC_WIRELESS_TRACKER_V1_0",
    59: "UNPHONE", 60: "TD_LORAC", 61: "CDEBYTE_EORA_S3", 62: "TWC_MESH_V4",
    63: "NRF52_PROMICRO_DIY", 64: "RADIOMASTER_900_BANDIT_NANO",
    65: "HELTEC_CAPSULE_SENSOR_V3", 66: "HELTEC_VISION_MASTER_T190",
    67: "HELTEC_VISION_MASTER_E213", 68: "HELTEC_VISION_MASTER_E290",
    69: "HELTEC_MESH_NODE_T114", 70: "SENSECAP_INDICATOR",
    71: "TRACKER_T1000_E", 72: "RAK3172", 73: "WIO_E5",
    74: "RADIOMASTER_900_BANDIT", 75: "ME25LS01_4Y10TD",
    76: "RP2040_FEATHER_RFM95", 77: "M5STACK_COREBASIC", 78: "M5STACK_CORE2",
    79: "RPI_PICO2", 80: "M5STACK_CORES3", 81: "SEEED_XIAO_S3", 82: "MS24SF1",
    83: "TLORA_C6", 84: "WISMESH_TAP", 85: "ROUTASTIC", 86: "MESH_TAB",
    87: "MESHLINK", 88: "XIAO_NRF52_KIT", 89: "THINKNODE_M1",
    90: "THINKNODE_M2", 91: "T_ETH_ELITE", 92: "HELTEC_SENSOR_HUB",
    93: "MUZI_BASE", 94: "HELTEC_MESH_POCKET", 95: "SEEED_SOLAR_NODE",
    96: "NOMADSTAR_METEOR_PRO", 97: "CROWPANEL", 98: "LINK_32",
    99: "SEEED_WIO_TRACKER_L1", 100: "SEEED_WIO_TRACKER_L1_EINK",
    101: "MUZI_R1_NEO", 102: "T_DECK_PRO", 103: "T_LORA_PAGER",
    104: "M5STACK_RESERVED", 105: "WISMESH_TAG", 106: "RAK3312",
    107: "THINKNODE_M5", 108: "HELTEC_MESH_SOLAR", 109: "T_ECHO_LITE",
    110: "HELTEC_V4", 111: "M5STACK_C6L", 112: "M5STACK_CARDPUTER_ADV",
    113: "HELTEC_WIRELESS_TRACKER_V2", 114: "T_WATCH_ULTRA",
    115: "THINKNODE_M3", 116: "WISMESH_TAP_V2", 117: "RAK3401",
    118: "RAK6421", 119: "THINKNODE_M4", 120: "THINKNODE_M6",
    121: "MESHSTICK_1262", 122: "TBEAM_1_WATT", 123: "T5_S3_EPAPER_PRO",
    124: "TBEAM_BPF", 125: "MINI_EPAPER_S3", 126: "TDISPLAY_S3_PRO",
    127: "HELTEC_MESH_NODE_T096", 128: "TRACKER_T1000_E_PRO",
    255: "PRIVATE_HW",
}

NODE_ROLES = {
    0: "CLIENT", 1: "CLIENT_MUTE", 2: "ROUTER", 3: "ROUTER_CLIENT",
    4: "REPEATER", 5: "TRACKER", 6: "SENSOR", 7: "ATAK",
    8: "CLIENT_HIDDEN", 9: "LOST_AND_FOUND",
}

ROUTING_ERRORS = {
    0: "NONE", 1: "NO_ROUTE", 2: "GOT_NAK", 3: "TIMEOUT",
    4: "NO_INTERFACE", 5: "MAX_RETRANSMIT", 6: "NO_CHANNEL",
    7: "TOO_LARGE", 8: "NO_RESPONSE", 9: "PKT_TOO_GOOD",
    32: "BAD_REQUEST", 33: "NOT_AUTHORIZED",
}


def decode_position(data):
    fields = parse_protobuf(data)
    lat, lon, alt = None, None, None
    speed, track, sats, ts = None, None, None, None
    for fn, wt, val in fields:
        if fn == 1 and wt == 5:   # latitude_i (sfixed32)
            lat = struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] * 1e-7
        elif fn == 2 and wt == 5: # longitude_i (sfixed32)
            lon = struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] * 1e-7
        elif fn == 3 and wt == 0: # altitude (m)
            alt = int(val)
        elif fn == 9 and wt == 0: # sats_in_view
            sats = int(val)
        elif fn == 14 and wt == 0: # ground_speed (m/s)
            speed = int(val)
        elif fn == 15 and wt == 0: # ground_track (degrees * 1e-5)
            track = int(val) * 1e-5
        elif fn == 17 and wt == 5: # gps_timestamp (unix epoch, fixed32)
            ts = val
    if lat is not None and lon is not None:
        s = "    Position: %.6f, %.6f" % (lat, lon)
        if alt is not None: s += " alt=%dm" % alt
        if speed is not None: s += " speed=%dm/s" % speed
        if track is not None: s += " heading=%.1f°" % track
        if sats is not None: s += " sats=%d" % sats
        if ts: s += " gps_time=%d" % ts
        print(s)
        return {'lat': round(lat, 6), 'lon': round(lon, 6),
                **({'alt': alt} if alt is not None else {})}
    return None


def decode_nodeinfo(data):
    fields = parse_protobuf(data)
    hw = None
    out = {}
    for fn, wt, val in fields:
        if fn == 1 and wt == 2:
            print("    NodeID: %s" % val.decode('utf-8', errors='replace'))
        elif fn == 2 and wt == 2:
            out['node_long'] = val.decode('utf-8', errors='replace')
            print("    LongName: %s" % out['node_long'])
        elif fn == 3 and wt == 2:
            out['node_short'] = val.decode('utf-8', errors='replace')
            print("    ShortName: %s" % out['node_short'])
        elif fn == 6 and wt == 0:
            hw = int(val)
            hw_name = HW_MODEL_NAMES.get(hw, "UNKNOWN(%d)" % hw)
            out['hw'] = hw_name
            print("    HW Model: %s (%d)" % (hw_name, hw))
        elif fn == 7 and wt == 0:
            if val: print("    Licensed: yes (HAM)")
        elif fn == 9 and wt == 0:
            role_name = NODE_ROLES.get(int(val), "UNKNOWN(%d)" % val)
            out['role'] = role_name
            print("    Role: %s" % role_name)
    return out


def decode_routing(data):
    """Decode Routing message — ACK/NACK or route discovery."""
    fields = parse_protobuf(data)
    out = {}
    for fn, wt, val in fields:
        if fn in (1, 2) and wt == 2:  # route_request / route_reply (RouteDiscovery)
            hops = []
            sub = parse_protobuf(val)
            for sfn, swt, sval in sub:
                if sfn == 1 and swt == 5:  # route node IDs (fixed32)
                    hops.append("!%08x" % sval)
            if hops:
                print("    Route: %s" % " → ".join(hops))
                out['route'] = hops
        elif fn == 3 and wt == 0:  # error_reason
            err_name = ROUTING_ERRORS.get(int(val), "UNKNOWN(%d)" % val)
            if val == 0:
                print("    ACK (delivered)")
                out['routing'] = 'ACK'
            else:
                print("    NAK: %s" % err_name)
                out['routing'] = 'NAK: %s' % err_name
    if 'routing' not in out and 'route' not in out:
        out['routing'] = 'ACK'   # empty Routing payload = bare delivery ack
    return out


def decode_waypoint(data):
    """Decode Waypoint — a shared map pin (name + location)."""
    fields = parse_protobuf(data)
    wp = {}
    for fn, wt, val in fields:
        if fn == 2 and wt == 5:    # latitude_i
            wp['lat'] = round(struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] * 1e-7, 6)
        elif fn == 3 and wt == 5:  # longitude_i
            wp['lon'] = round(struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] * 1e-7, 6)
        elif fn == 6 and wt == 2:  # name
            wp['name'] = val.decode('utf-8', errors='replace')
        elif fn == 7 and wt == 2:  # description
            wp['desc'] = val.decode('utf-8', errors='replace')
        elif fn == 8 and wt == 5:  # icon (emoji codepoint)
            try:
                wp['icon'] = chr(val) if 0x20 < val < 0x10FFFF else None
            except (ValueError, OverflowError):
                pass
    if wp:
        print("    Waypoint: %s @ %s,%s" % (wp.get('name', '?'), wp.get('lat'), wp.get('lon')))
        return {'waypoint': wp}
    return None


def decode_neighborinfo(data):
    """Decode NeighborInfo — the sender's heard RF neighbors and their SNR."""
    fields = parse_protobuf(data)
    out = {'neighbors': []}
    for fn, wt, val in fields:
        if fn == 3 and wt == 0:    # node_broadcast_interval_secs
            out['nbr_interval'] = int(val)
        elif fn == 4 and wt == 2:  # repeated Neighbor
            n = {}
            for sfn, swt, sval in parse_protobuf(val):
                if sfn == 1 and swt == 0:   # node_id (uint32)
                    n['node'] = "!%08x" % sval
                elif sfn == 2 and swt == 5:  # snr (float)
                    n['snr'] = round(struct.unpack('<f', struct.pack('<I', sval & 0xFFFFFFFF))[0], 1)
            if n.get('node'):
                out['neighbors'].append(n)
    if out['neighbors']:
        print("    Neighbors: %s" % ", ".join(
            "%s(%.1fdB)" % (x['node'], x['snr']) if 'snr' in x else x['node']
            for x in out['neighbors']))
        return out
    return None


def decode_mapreport(data):
    """Decode MapReport — identity + position a node broadcasts to the public map."""
    fields = parse_protobuf(data)
    out = {}
    for fn, wt, val in fields:
        if fn == 1 and wt == 2:    # long_name
            out['node_long'] = val.decode('utf-8', errors='replace')
        elif fn == 2 and wt == 2:  # short_name
            out['node_short'] = val.decode('utf-8', errors='replace')
        elif fn == 3 and wt == 0:  # role
            out['role'] = NODE_ROLES.get(int(val), "UNKNOWN(%d)" % val)
        elif fn == 4 and wt == 0:  # hw_model
            out['hw'] = HW_MODEL_NAMES.get(int(val), "UNKNOWN(%d)" % val)
        elif fn == 5 and wt == 2:  # firmware_version
            out['fw'] = val.decode('utf-8', errors='replace')
        elif fn == 9 and wt == 5:  # latitude_i
            out['lat'] = round(struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] * 1e-7, 6)
        elif fn == 10 and wt == 5: # longitude_i
            out['lon'] = round(struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] * 1e-7, 6)
        elif fn == 11 and wt == 0: # altitude
            out['alt'] = int(val)
        elif fn == 13 and wt == 0: # num_online_local_nodes
            out['num_online'] = int(val)
    if out:
        print("    MapReport: %s" % out.get('node_long', '?'))
        return out
    return None


def decode_paxcounter(data):
    """Decode Paxcount — wifi/ble device counts."""
    fields = parse_protobuf(data)
    out = {}
    for fn, wt, val in fields:
        if fn == 1 and wt == 0:
            out['pax_wifi'] = int(val)
        elif fn == 2 and wt == 0:
            out['pax_ble'] = int(val)
        elif fn == 3 and wt == 0:
            out['pax_uptime'] = int(val)
    return out or None


# Plain-text portnums: the inner payload is just UTF-8 text.
_TEXT_PORTNUMS = {1, 10, 32, 66}   # TEXT, DETECTION_SENSOR, REPLY, RANGE_TEST


def _portnum_record(portnum, inner_payload):
    """Enrichment dict for a decrypted Data payload (text/pos/telemetry/…).
    Shared by the live decode path and the web retry-decode path so new portnums
    only need adding here once."""
    rec = {}
    if not inner_payload:
        return rec
    if portnum in _TEXT_PORTNUMS:
        rec['text'] = inner_payload.decode('utf-8', errors='replace')
    elif portnum == 3:    # POSITION
        rec.update(decode_position(inner_payload) or {})
    elif portnum == 4:    # NODEINFO
        rec.update(decode_nodeinfo(inner_payload) or {})
    elif portnum == 5:    # ROUTING
        rec.update(decode_routing(inner_payload) or {})
    elif portnum == 8:    # WAYPOINT
        rec.update(decode_waypoint(inner_payload) or {})
    elif portnum == 34:   # PAXCOUNTER
        rec.update(decode_paxcounter(inner_payload) or {})
    elif portnum == 67:   # TELEMETRY
        rec.update(decode_telemetry(inner_payload) or {})
    elif portnum == 70:   # TRACEROUTE
        rec.update(decode_traceroute(inner_payload) or {})
    elif portnum == 71:   # NEIGHBORINFO
        rec.update(decode_neighborinfo(inner_payload) or {})
    elif portnum == 73:   # MAP_REPORT
        rec.update(decode_mapreport(inner_payload) or {})
    else:
        rec['payload_hex'] = inner_payload.hex()
    return rec


def retry_decrypt(enc_hex, pktid, from_str, key_str):
    """Re-attempt decryption of a stored ENCRYPTED packet with one key (web retry).
    Rebuilds the same AES-CTR nonce as parse_meshtastic_packet from the packet id
    and sender, decrypts, validates, and returns an enrichment dict (with portnum/
    port_name) or None if it doesn't decode to a valid packet.  key_str=None/''
    uses the default Meshtastic key."""
    try:
        kb = _decode_key_str(key_str) if key_str else MESH_AES_KEY
        if kb is None:
            return None
        enc = bytes.fromhex(enc_hex)
        pid = int(pktid, 16) if isinstance(pktid, str) else int(pktid)
        fb = bytes.fromhex(str(from_str).lstrip('!'))[::-1]   # sender bytes p[4:8]
        nonce = bytearray(16)
        nonce[0:4] = pid.to_bytes(4, 'little')
        nonce[8:12] = fb
        dec = aes_ctr_decrypt(kb, bytes(nonce), enc)
        portnum, inner = -1, None
        for fn, wt, val in parse_protobuf(dec):
            if fn == 1 and wt == 0:
                portnum = int(val)
            elif fn == 2 and wt == 2:
                inner = val
        if portnum < 0 or portnum not in KNOWN_PORTNUMS:
            return None
        if portnum in _TEXT_PORTNUMS and inner:
            try:
                inner.decode('utf-8')
            except UnicodeDecodeError:
                return None
        out = {'decrypted': True, 'portnum': portnum, 'port_name': portnum_name(portnum)}
        out.update(_portnum_record(portnum, inner))
        return out
    except Exception:
        return None


def decode_traceroute(data):
    """Decode RouteDiscovery — list of node IDs the packet traversed."""
    fields = parse_protobuf(data)
    hops = []
    snr_list = []
    for fn, wt, val in fields:
        if fn == 1 and wt == 5:   # route (repeated fixed32 node IDs)
            hops.append("!%08x" % val)
        elif fn == 2 and wt == 5: # snr_towards (repeated sfixed32, scaled *4)
            snr_list.append(struct.unpack('<i', struct.pack('<I', val & 0xFFFFFFFF))[0] / 4.0)
    if hops:
        if snr_list and len(snr_list) == len(hops):
            pairs = ["%s(%.1fdB)" % (h, s) for h, s in zip(hops, snr_list)]
            print("    Hops: %s" % " → ".join(pairs))
        else:
            print("    Hops: %s" % " → ".join(hops))
        return {'route': hops, 'route_snr': snr_list if len(snr_list) == len(hops) else None}
    return None


def decode_telemetry(data):
    fields = parse_protobuf(data)
    out = {}
    for fn, wt, val in fields:
        if fn == 1 and wt == 0:
            print("    Time: %d" % val)
        elif fn == 2 and wt == 2:  # device_metrics sub-message
            sub = parse_protobuf(val)
            for sfn, swt, sval in sub:
                if sfn == 1 and swt == 0:
                    out['battery'] = int(sval)
                    print("    Battery: %d%%" % sval)
                elif sfn == 2 and swt == 5:
                    v = struct.unpack('<f', struct.pack('<I', sval & 0xFFFFFFFF))[0]
                    out['voltage'] = round(v, 2)
                    print("    Voltage: %.2fV" % v)
                elif sfn == 3 and swt == 5:
                    v = struct.unpack('<f', struct.pack('<I', sval & 0xFFFFFFFF))[0]
                    out['chutil'] = round(v, 1)
                    print("    ChUtil: %.1f%%" % v)
                elif sfn == 4 and swt == 5:
                    v = struct.unpack('<f', struct.pack('<I', sval & 0xFFFFFFFF))[0]
                    out['airutil'] = round(v, 1)
                    print("    AirUtilTX: %.1f%%" % v)
    return out


# ============================================================================
# LoRaWAN frame parser
# ============================================================================
_LORAWAN_MTYPE = {
    0b000: "Join Request",   0b001: "Join Accept",
    0b010: "Unconfirmed Data Up",  0b011: "Unconfirmed Data Down",
    0b100: "Confirmed Data Up",    0b101: "Confirmed Data Down",
    0b110: "Rejoin Request",       0b111: "Proprietary",
}

# Region channel plan.  A genuine LoRaWAN frame is regulatorily pinned to an
# exact (frequency, SF, BW) grid point; garbage that merely shares the 0x34 sync
# word is almost never on-grid (e.g. our co-channel artifacts sit on the
# Meshtastic frequency, well off the LoRaWAN grid).  This is corroborating
# evidence, NOT a hard drop: we have no calibrated real-LoRaWAN captures yet, and
# a miscalibrated frequency measurement must never silently discard real traffic.
_LORA_REGION = os.environ.get('LORA_REGION', 'US915').upper()
_LORAWAN_GRID_TOL_KHZ = float(os.environ.get('LORA_GRID_TOL_KHZ', '60'))

# ---- Protocol/unknown gating (Config → Advanced Options; applies on next Start) ----
# LORA_PROTOCOLS = comma list of enabled protocols (default all).  LORA_UNKNOWN = 1
# surfaces unrecognised intact frames as proto='unknown' (default 0 = off, no clutter
# AND lets the decoder early-bail on irrelevant sync words to save compute).
_PROTO_ENABLED = set(p for p in os.environ.get(
    'LORA_PROTOCOLS',
    'meshtastic,meshcore,lorawan,loramesher,lora_aprs,reticulum,disaster_radio,ebyte_lora,radiohead'
).replace(' ', '').lower().split(',') if p)
_PROTO_UNKNOWN = os.environ.get('LORA_UNKNOWN', '0') == '1'
# LORA_FINGERPRINT = '1' (default) enables hardware-fingerprint extraction on
# every decoded packet (feeds web UI device clustering + Mystery Devices).
# Set to '0' on low-core hosts to skip the per-packet UMOP feature extraction —
# the decoder runs ~10-15 % faster.  Trade-off: no device clustering / RF
# attribution / Mystery Devices panel.  Header decode coverage unchanged.
_FINGERPRINT_ON = os.environ.get('LORA_FINGERPRINT', '1').strip().lower() not in ('0', 'false', 'off', 'no')
# Meshtastic sync tolerance for the early-bail: 0x2B/0x0F plus the 1-bit neighbours of
# 0x2B (a sync symbol can demod 1 bit off on a marginal capture).
_MESH_SYNCS = {0x2B, 0x0F} | {0x2B ^ (1 << _i) for _i in range(8)}


def _protocol_skip(sync):
    """Compute-saving early bail: True → skip the expensive decode entirely.  Only
    skips when there is provably nothing of interest: unknown reporting is OFF,
    MeshCore (whose sync word is user-configurable, so ANY sync could be MeshCore)
    is DISABLED, and the sync word matches no ENABLED fixed-sync protocol."""
    if _PROTO_UNKNOWN:                       # must decode everything to surface unknowns
        return False
    if 'meshcore' in _PROTO_ENABLED:         # arbitrary sync → can't filter by sync
        return False
    if sync is None:                         # ambiguous; meshtastic tolerates None
        return 'meshtastic' not in _PROTO_ENABLED
    allowed = set()
    if 'meshtastic' in _PROTO_ENABLED:
        allowed |= _MESH_SYNCS
    if 'lorawan' in _PROTO_ENABLED:
        allowed.add(0x34)
    return sync not in allowed

def _on_lorawan_grid(freq_mhz, sf, bw, region=None, tol_khz=None):
    """True if (freq, SF, BW) lands on a region channel/DR grid point within
    tolerance; False if clearly off-grid; None if freq/sf/bw unknown (cannot judge)."""
    if freq_mhz is None or sf is None or bw is None:
        return None
    region = region or _LORA_REGION
    tol = (tol_khz if tol_khz is not None else _LORAWAN_GRID_TOL_KHZ) / 1000.0
    bw_khz = bw / 1000.0
    chans = []
    if region == 'US915':
        if abs(bw_khz - 125) < 30 and 7 <= sf <= 10:           # uplink 125k, DR0-3
            chans = [902.3 + 0.2 * n for n in range(64)]
        elif abs(bw_khz - 500) < 100:
            if sf == 8:                                         # uplink 500k, DR4
                chans += [903.0 + 1.6 * n for n in range(8)]
            if 7 <= sf <= 12:                                   # downlink 500k, DR8-13
                chans += [923.3 + 0.6 * n for n in range(8)]
    else:
        return None                                            # unknown region: don't judge
    if not chans:
        return False
    return any(abs(freq_mhz - c) <= tol for c in chans)


def parse_lorawan_packet(payload, rf=None):
    """Parse a LoRaWAN frame: print human-readable fields AND return a normalized
    record dict (proto='lorawan') for the [PKT] stream, or None."""
    if len(payload) < 1:
        return None
    mhdr   = payload[0]
    mtype  = (mhdr >> 5) & 0x07
    major  = mhdr & 0x03
    # --- Structural validity gate (CRC-independent; valid for CRC-off downlinks) ---
    # The MHDR's RFU bits (4:2) and Major (1:0) are spec-fixed to zero — only the
    # top 3 bits (MType) are free, so a valid MHDR satisfies (mhdr & 0x1F) == 0
    # (8 of 256 byte values).  Then length must be consistent with the MType.
    # These are fixed LoRaWAN-spec rules that hold for uplinks and downlinks, so
    # they reject sync-0x34 garbage without needing keys or CRC.
    n = len(payload)
    if (mhdr & 0x1F) != 0:
        return None
    if   mtype == 0b000:                            # Join Request — fixed 23 bytes
        if n != 23: return None
    elif mtype == 0b001:                            # Join Accept — base or +CFList
        if n not in (17, 33): return None
    elif mtype in (0b010, 0b011, 0b100, 0b101):     # Data up/down
        if n < 12 or 8 + (payload[5] & 0x0F) + 4 > n: return None  # FOpts + MIC must fit
    elif mtype == 0b110:                            # Rejoin Request
        if n < 18: return None
    else:                                           # 0b111 Proprietary — unvalidatable w/o keys
        return None
    msg_type = _LORAWAN_MTYPE.get(mtype, "Unknown (%d)" % mtype)
    # STRICT positive-ID: do NOT claim 'lorawan' on structure alone (a tight but
    # still non-discriminating fingerprint).  Surface as an 'unknown' frame that
    # *looks like* LoRaWAN, carrying the parsed fields; the web layer promotes it
    # to a NAMED, confirmed LoRaWAN device only once DevAddr + monotonic FCnt agree
    # across frames (behavioral proof, no keys needed).
    # Confidence is 'candidate' — structural-only.  We can't crypto-verify
    # without the NwkSKey (MIC check needs it).  The web layer promotes this
    # to 'confirmed' when DevAddr + monotonic FCnt agree across multiple frames
    # (behavioral proof; no keys needed).
    rec = {'proto': 'unknown', 'hint': 'lorawan', 'confidence': 'candidate',
           'msg_type': msg_type, 'decrypted': False}
    if rf:
        _og = _on_lorawan_grid(rf.get('freq_mhz'), rf.get('sf'), rf.get('bw'))
        if _og is not None:
            rec['on_grid'] = _og
    print("\n  --- LoRaWAN Frame ---")
    print("  MsgType: %s" % msg_type)
    if major:
        print("  Major: %d" % major)
    if mtype in (0b000, 0b110):             # Join Request / Rejoin
        if len(payload) >= 19:
            join_eui = payload[1:9][::-1].hex().upper()
            dev_eui  = payload[9:17][::-1].hex().upper()
            dev_nonce = int.from_bytes(payload[17:19], 'little')
            print("  JoinEUI:  %s" % ':'.join(join_eui[i:i+2] for i in range(0, 16, 2)))
            print("  DevEUI:   %s" % ':'.join(dev_eui[i:i+2] for i in range(0, 16, 2)))
            print("  DevNonce: %d" % dev_nonce)
            rec['deveui'] = dev_eui
            rec['summary'] = '%s · DevEUI %s' % (msg_type, dev_eui)
        if len(payload) >= 23:
            print("  MIC: %s" % payload[-4:].hex().upper())
    elif mtype == 0b001:                    # Join Accept — encrypted
        print("  [Join Accept - encrypted without AppKey]")
        rec['summary'] = 'Join Accept (encrypted)'
    elif mtype in (0b010, 0b011, 0b100, 0b101):  # Data frames
        if len(payload) < 8:
            print("  [too short for data frame: %d bytes]" % len(payload))
            rec['summary'] = msg_type + ' (truncated)'
            return rec
        devaddr   = payload[1:5][::-1].hex().upper()
        fctrl     = payload[5]
        fopts_len = fctrl & 0x0F
        adr       = bool(fctrl & 0x80)
        ack       = bool(fctrl & 0x20)
        fpending  = bool(fctrl & 0x10)
        fcnt      = int.from_bytes(payload[6:8], 'little')
        fopts_end = 8 + fopts_len
        ctrl_flags = ",".join(f for f in ["ADR" if adr else "", "ACK" if ack else "",
                                          "FPending" if fpending else ""] if f)
        print("  DevAddr: %s" % devaddr)
        print("  FCnt: %d%s" % (fcnt, ("  [%s]" % ctrl_flags) if ctrl_flags else ""))
        rec['devaddr'] = devaddr
        rec['fcnt'] = fcnt
        fport = None
        remaining = payload[fopts_end:-4] if len(payload) > fopts_end + 4 else b""
        if len(remaining) >= 1:
            fport = remaining[0]
            frm_len = len(remaining) - 1
            print("  FPort: %d" % fport)
            rec['fport'] = fport
            if frm_len > 0:
                print("  FRMPayload: %d bytes (encrypted)" % frm_len)
        if len(payload) >= 4:
            print("  MIC: %s" % payload[-4:].hex().upper())
        rec['summary'] = 'DevAddr %s · FCnt %d%s%s' % (
            devaddr, fcnt, (' · FPort %d' % fport) if fport is not None else '',
            (' · ' + ctrl_flags) if ctrl_flags else '')
    return rec


# ============================================================================
# MeshCore packet parser
# ============================================================================
_MESHCORE_ROUTE_TYPES   = {0: 'FLOOD', 1: 'DIRECT', 2: 'BACK', 3: 'DIRECT_OOB'}
_MESHCORE_PAYLOAD_TYPES = {
    0x00: 'REQ',    0x01: 'RESPONSE', 0x02: 'TXT_MSG', 0x03: 'ACK',
    0x04: 'ADVERT', 0x05: 'GRP_TXT', 0x06: 'GRP_DATA', 0x07: 'ANON_REQ',
    0x09: 'TRACE',  0x0B: 'CONTROL',
}

_ED25519_VERIFY = None   # lazy cache: verify callable, or False if no crypto lib

def _get_ed25519_verifier():
    """Lazy-load an Ed25519 verify callable: (pub32, sig64, msg) -> bool.
    Returns None if no Ed25519 library is installed."""
    global _ED25519_VERIFY
    if _ED25519_VERIFY is None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature
            def _v(pub32, sig64, msg):
                try:
                    Ed25519PublicKey.from_public_bytes(pub32).verify(sig64, msg)
                    return True
                except InvalidSignature:
                    return False
                except Exception:
                    return False        # malformed key/sig
            _ED25519_VERIFY = _v
        except Exception:
            _ED25519_VERIFY = False     # library unavailable
    return _ED25519_VERIFY or None


def _verify_advert_sig(payload_rest):
    """Cryptographically verify a MeshCore ADVERT.

    Payload layout: pubkey(32) | timestamp(4) | signature(64) | appdata(var).
    The Ed25519 signature covers pubkey || timestamp || appdata (everything BUT
    the 64 signature bytes) = payload_rest[0:36] + payload_rest[100:].  A valid
    signature proves the frame is a genuine advert from the holder of `pubkey` —
    no shared/pre-provisioned key needed, because the key travels in the packet,
    so a frame that merely *looks* like an advert (sync+type match on noise)
    cannot forge it.

    Returns True (valid), False (too short or invalid → not a real advert),
    or None (no Ed25519 library; cannot verify)."""
    if len(payload_rest) < 100:
        return False                    # a real advert always carries the 64-byte sig
    verify = _get_ed25519_verifier()
    if verify is None:
        return None
    pub = bytes(payload_rest[0:32])
    sig = bytes(payload_rest[36:100])
    msg = bytes(payload_rest[0:36]) + bytes(payload_rest[100:])
    return verify(pub, sig, msg)


# --- MeshCore group-channel decryption ---------------------------------------
# A channel is keyed by a shared 16-byte AES-128 PSK; the well-known DEFAULT
# PUBLIC channel key below is MeshCore's analogue of Meshtastic's 'AQ=='.
# Decrypting a group message both recovers its text AND verifies it is real —
# noise won't pass the HMAC and decrypt to valid text.  GRP_TXT/GRP_DATA body
# layout: channel_hash(1) | mac(2) | ciphertext(16*n).  channel_hash =
# SHA256(key)[0]; mac = HMAC-SHA256(key||16 zero bytes, ciphertext)[:2];
# plaintext = AES-128-ECB(key, ciphertext) (no padding) = ts(4 LE) | flags(1) | text.
_MC_PUBLIC_KEY = bytes.fromhex('8b3387e9c5cdea6ac9e5edbaa115cd72')   # default public channel
import hashlib as _hashlib
_MC_PUBLIC_HASH = _hashlib.sha256(_MC_PUBLIC_KEY).digest()[0]        # 0x11
_MC_ENV_KEYS = []                          # extra channel keys from env (legacy)
for _k in os.environ.get('LORA_MC_CHANNEL_KEYS', '').replace(' ', '').split(','):
    if len(_k) == 32:
        try: _MC_ENV_KEYS.append(bytes.fromhex(_k))
        except ValueError: pass

def _mc_channel_keys():
    """channel_hash(int) -> 16-byte AES key for every ENABLED MeshCore channel
    key: the public default (unless disabled in the shared key list), legacy env
    keys, and protocol='meshcore' custom keys from LORA_KEYS.  Sourced from the
    same key store as Meshtastic (live mtime reload via _load_keys), so the web
    UI's enable/disable applies without a restart."""
    import hashlib
    cfg = _load_keys() or {}
    m = {}
    if cfg.get('meshcore_default', {}).get('enabled', True):
        m[_MC_PUBLIC_HASH] = _MC_PUBLIC_KEY
    for kb in _MC_ENV_KEYS:
        m[hashlib.sha256(kb).digest()[0]] = kb
    for k in cfg.get('keys', []):
        if k.get('protocol') != 'meshcore' or k.get('enabled') is False:
            continue
        kb = k.get('_bytes')
        if kb and len(kb) == 16:           # MeshCore channel crypto is AES-128
            m[hashlib.sha256(kb).digest()[0]] = kb
    return m


def _decrypt_mc_channel(payload_rest):
    """Decrypt a MeshCore group-channel message if its channel hash matches a
    known key and the 2-byte HMAC verifies.  Returns {'text','ts'} or None."""
    if len(payload_rest) < 1 + 2 + 16:
        return None
    key = _mc_channel_keys().get(payload_rest[0])
    if key is None:
        return None
    mac, ct = bytes(payload_rest[1:3]), bytes(payload_rest[3:])
    if not ct or len(ct) % 16:
        return None
    try:
        import hmac, hashlib
        if hmac.new(key + b'\x00' * 16, ct, hashlib.sha256).digest()[:2] != mac:
            return None
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        pt = Cipher(algorithms.AES(key), modes.ECB()).decryptor().update(ct)
    except Exception:
        return None
    if len(pt) < 5:
        return None
    return {'ts': int.from_bytes(pt[0:4], 'little'),
            'text': pt[5:].split(b'\x00')[0].decode('utf-8', 'replace')}


def parse_meshcore_packet(payload):
    """Parse a MeshCore packet: print human-readable fields AND return a normalized
    record dict (proto='meshcore') for the [PKT] stream, or None."""
    if len(payload) < 1:
        return None
    header       = payload[0]
    route_type   = header & 0x03
    payload_type = (header >> 2) & 0x0F
    version      = (header >> 6) & 0x03
    # --- Structural validity gate (CRC-independent) ---
    # Reject frames that cannot be MeshCore: an undefined payload type, or a
    # non-zero protocol version (current MeshCore uses version 0).  Garbage that
    # merely shares the 0x12 sync word almost never satisfies both.  (Path-fit is
    # checked below once the path length is known.)
    if payload_type not in _MESHCORE_PAYLOAD_TYPES or version != 0:
        return None
    route_name   = _MESHCORE_ROUTE_TYPES.get(route_type, "unknown(%d)" % route_type)
    ptype_name   = _MESHCORE_PAYLOAD_TYPES[payload_type]
    rec = {'proto': 'meshcore', 'mc_route': route_name, 'mc_type': ptype_name,
           'decrypted': False, 'confidence': 'candidate'}
    print("\n  --- MeshCore Packet ---")
    print("  RouteType: %s" % route_name)
    print("  PayloadType: %s" % ptype_name)
    if version:
        print("  Version: %d" % version)
    offset = 1
    if route_type in (0x00, 0x03):          # FLOOD / DIRECT_OOB have transport codes
        if len(payload) < offset + 4:
            return None        # truncated — cannot positively identify as MeshCore
        tc1 = int.from_bytes(payload[offset:offset + 2], 'big')
        tc2 = int.from_bytes(payload[offset + 2:offset + 4], 'big')
        print("  Transport: 0x%04X 0x%04X" % (tc1, tc2))
        offset += 4
    if len(payload) <= offset:
        return None        # truncated — cannot positively identify as MeshCore
    path_byte  = payload[offset]
    # Per MeshCore packet_format.md: path_byte = 0bHHCCCCCC where HH encodes
    # hash_size-1 (valid 0/1/2 → hash_size 1/2/3) and CCCCCC = hop_count.
    # HH=0b11 is RESERVED and MUST NOT appear in real MeshCore frames — reject.
    # This single check eliminates 25% of random-byte false positives.
    if ((path_byte >> 6) & 0x03) == 3:
        return None
    hash_count = path_byte & 0x3F
    hash_size  = ((path_byte >> 6) & 0x03) + 1
    offset    += 1
    _mc_path = []                              # routing path-hash (pubkey[0]) of each hop
    if hash_count > 0:
        path_bytes_needed = hash_count * hash_size
        if len(payload) < offset + path_bytes_needed:
            return None     # declared path overruns the frame → not a real MeshCore packet
        rec['mc_hops'] = hash_count
        _mc_path = [payload[offset + i*hash_size] for i in range(hash_count)]
        print("  HopCount: %d" % hash_count)
        hops = [payload[offset + i*hash_size:offset + (i+1)*hash_size].hex().upper()
                for i in range(hash_count)]
        print("  Path: %s" % ' -> '.join(hops))
        offset += path_bytes_needed
    rec['mc_path'] = _mc_path
    payload_rest = payload[offset:]
    if payload_type in (0x05, 0x06) and len(payload_rest) >= 1:   # GRP_TXT / GRP_DATA
        print("  ChannelHash: 0x%02X" % payload_rest[0])
        _dec = _decrypt_mc_channel(payload_rest)
        if _dec is not None:
            rec['decrypted'] = True
            rec['confidence'] = 'verified'
            rec['mc_channel'] = 'public' if payload_rest[0] == _MC_PUBLIC_HASH else ('ch:0x%02x' % payload_rest[0])
            if payload_type == 0x05 and _dec['text']:
                rec['text'] = _dec['text']
            print("  [DECRYPTED %s] %s" % (rec['mc_channel'], _dec.get('text', '')))
    elif payload_type == 0x04:                                    # ADVERT (Ed25519-signed)
        _sig_ok = _verify_advert_sig(payload_rest)
        if _sig_ok is False:
            return None        # signature absent/invalid → provably not a real advert
        rec['verified'] = bool(_sig_ok)
        if _sig_ok:
            rec['confidence'] = 'verified'
        print("  PubKey: %s..." % payload_rest[0:8].hex().upper())
        print("  Timestamp: %d" % int.from_bytes(payload_rest[32:36], 'little'))
        print("  Signature: %s" % ('VERIFIED (Ed25519)' if _sig_ok else 'unverified (no crypto lib)'))
        # Node identity for the registry: full pubkey + routing hash (pubkey[0]) +
        # appdata (role / location / name).  appdata = flags | [lat,lon] | name.
        rec['mc_pubkey'] = payload_rest[0:32].hex()
        rec['mc_hash'] = payload_rest[0]
        _ad = payload_rest[100:]
        if _ad:
            _flags = _ad[0]; _ao = 1
            rec['mc_role'] = {1: 'ChatNode', 2: 'Repeater', 3: 'RoomServer',
                              4: 'Sensor'}.get(_flags & 0x0F, 'role%d' % (_flags & 0x0F))
            if (_flags & 0x10) and len(_ad) >= _ao + 8:
                rec['lat'] = int.from_bytes(_ad[_ao:_ao + 4], 'little', signed=True) / 1e6
                rec['lon'] = int.from_bytes(_ad[_ao + 4:_ao + 8], 'little', signed=True) / 1e6
                _ao += 8
            if (_flags & 0x80) and len(_ad) > _ao:
                rec['mc_name'] = _ad[_ao:].split(b'\x00')[0].decode('utf-8', 'replace')
            if rec.get('mc_name'):
                print("  Name: %s  Role: %s" % (rec['mc_name'], rec.get('mc_role', '?')))
        rec['from'] = 'MC:' + payload_rest[0:4].hex().upper()
    # GENERAL-IDENTIFIER GATE: only CLAIM MeshCore when positively identified — a
    # valid ADVERT Ed25519 signature, or a channel message that decrypts +
    # MAC-verifies.  MeshCore's header check is loose (accepts ~1 in 5 random
    # frames), so a structurally-valid but unverified frame is indistinguishable
    # from a LoRa protocol we haven't implemented; return None so it surfaces as
    # 'unknown' rather than being mislabeled MeshCore.
    if rec.get('confidence') != 'verified':
        # Not crypto-verified.  Surface as an UNKNOWN frame that *looks like*
        # MeshCore, carrying its path hashes so the web can corroborate it against
        # the ADVERT-verified node registry (a frame routed through known nodes is
        # behaviorally confirmed).  NOT a MeshCore claim until the web promotes it.
        return {'proto': 'unknown', 'hint': 'meshcore',
                'confidence': 'candidate',
                'mc_route': route_name,
                'mc_type': ptype_name, 'mc_path': _mc_path, 'decrypted': False,
                'summary': 'looks like MeshCore · %s/%s%s' % (
                    route_name, ptype_name,
                    (' · %d hops' % len(_mc_path)) if _mc_path else '')}
    _id = ''
    if rec.get('mc_name'):
        _id = ' · ' + rec['mc_name'] + ((' (%s)' % rec['mc_role']) if rec.get('mc_role') else '')
    rec['summary'] = '%s / %s%s%s · verified' % (
        route_name, ptype_name,
        (' · %d hops' % rec['mc_hops']) if rec.get('mc_hops') else '', _id)
    return rec


# ============================================================================
# LoRaMesher packet parser
# ============================================================================
# Wire format (per LoRaMesher/LoRaMesher PROTOCOL_SPEC.md v1.6):
#   BaseHeader (6 bytes):  dst(2 LE) | src(2 LE) | msg_type(1) | payload_size(1)
# Where dst == 0xFFFF is broadcast.  msg_type uses the high nibble for the
# main category (0x10=DATA, 0x20=CONTROL, 0x30=ROUTING, 0x40=SYSTEM) and the
# low nibble for the subtype.  Strict-whitelist values from
# src/types/messages/message_type.hpp (verified directly from the cloned
# repo): 14 specific bytes = 5.5% random-byte pass rate.  Combined with the
# payload_size match against actual remaining length (~1/64 random pass
# under our 4-byte tolerance), the joint structural FP rate is well under
# 0.1%.  We CLAIM 'loramesher' only when dst or src match a previously-seen
# LoRaMesher node (behavioral confirmation); otherwise surface as
# 'unknown hint=loramesher' carrying the parsed fields so the web layer
# can promote it once enough corroborating frames arrive.

_LORAMESHER_MSG_TYPES = {
    0x11: 'DATA',           0x12: 'DATA_BROADCAST',
    0x21: 'ACK',            0x23: 'PING',           0x24: 'PONG',
    0x31: 'HELLO',          0x32: 'ROUTE_TABLE',
    0x41: 'SYNC',           0x42: 'JOIN_REQUEST',   0x43: 'JOIN_RESPONSE',
    0x44: 'SLOT_REQUEST',   0x45: 'SLOT_ALLOCATION',
    0x46: 'SYNC_BEACON',    0x47: 'NM_CLAIM',
}

# Behavioral promotion state for LoRaMesher.  `_LORAMESHER_PENDING` counts
# structurally-valid sightings per src; only after ≥2 independent sightings do
# we promote into `_KNOWN_LORAMESHER_NODES` (the confirmed-node set).  This
# matches the multi-frame corroboration policy used elsewhere (LoRaWAN's web
# layer promotes on DevAddr+FCnt agreement across frames; Meshtastic's known-
# node set is rooted in crypto-verified clean decrypts).  A single CRC-16
# coincidence that satisfies LoRaMesher's structural validator never promotes.
_LORAMESHER_PENDING = {}    # src(uint16) -> sighting count
_KNOWN_LORAMESHER_NODES = set()

def parse_loramesher_packet(payload, rf=None):
    """Parse a LoRaMesher BaseHeader.  Returns a record dict (proto='loramesher'
    when behaviorally confirmed, else proto='unknown' hint='loramesher') or None
    when the bytes don't structurally match LoRaMesher."""
    if len(payload) < 6:
        return None
    dst = payload[0] | (payload[1] << 8)
    src = payload[2] | (payload[3] << 8)
    msg_type = payload[4]
    payload_size = payload[5]
    # --- Strict structural validity (CRC-independent) ---
    # 1. msg_type must be one of the 14 defined values
    if msg_type not in _LORAMESHER_MSG_TYPES:
        return None
    # 2. src must be nonzero (real node); dst==0xFFFF means broadcast
    if src == 0 or src == 0xFFFF:
        return None
    # 3. payload_size byte must match actual remaining payload within ±4 bytes
    #    (LoRaMesher's payload_size is "message-specific fields + payload" so
    #    it should equal len(payload) - 6 exactly, but allow tiny slack for
    #    captures with PHY padding artifacts).
    remaining = len(payload) - 6
    if not (abs(payload_size - remaining) <= 4):
        return None
    # 4. dst != src (a node addressing itself is nonsensical at this layer)
    if dst == src:
        return None

    type_name = _LORAMESHER_MSG_TYPES[msg_type]
    summary = '%s · %04X→%04X · pl=%d' % (
        type_name, src, 0xFFFF if dst == 0xFFFF else dst, payload_size)

    # Behavioral promotion: if EITHER endpoint is in the CONFIRMED known-node
    # set (≥2 prior independent sightings), promote to 'confirmed'.  Otherwise
    # surface as 'unknown hint=loramesher' (candidate).
    is_known = (dst in _KNOWN_LORAMESHER_NODES) or (src in _KNOWN_LORAMESHER_NODES)
    # Register this sighting.  Bump count; promote to confirmed-set only on the
    # 2nd structurally-valid sighting of the same src.  A single CRC-16 + LoRa-
    # Mesher-structural false-positive (combined rate ~3 × 10⁻⁸) does NOT enter
    # the confirmed set — the second matching sighting that would, at ~1.4 × 10⁻²⁰
    # joint probability, is effectively impossible.
    _LORAMESHER_PENDING[src] = _LORAMESHER_PENDING.get(src, 0) + 1
    if _LORAMESHER_PENDING[src] >= 2:
        _KNOWN_LORAMESHER_NODES.add(src)
        # Bound the pending dict — once a src is promoted it lives in the
        # known set; no need to keep counting it.
        if src in _KNOWN_LORAMESHER_NODES:
            _LORAMESHER_PENDING.pop(src, None)

    print("\n  --- LoRaMesher Frame ---")
    print("  Type: %s (0x%02X)" % (type_name, msg_type))
    print("  From: 0x%04X  To: %s" % (src, 'BROADCAST' if dst == 0xFFFF else '0x%04X' % dst))
    print("  PayloadSize: %d (actual remaining: %d)" % (payload_size, remaining))
    if is_known:
        print("  Behavioral: known LoRaMesher node — promoting to confirmed")
    return {
        'proto': 'loramesher' if is_known else 'unknown',
        'hint': None if is_known else 'loramesher',
        'confidence': 'confirmed' if is_known else 'candidate',
        'lm_type': type_name,
        'from': '0x%04X' % src,
        'to': 'BROADCAST' if dst == 0xFFFF else '0x%04X' % dst,
        'lm_payload_size': payload_size,
        'decrypted': False,
        'summary': summary,
    }


# ============================================================================
# LoRa APRS — Austrian ham radio standard for APRS over LoRa.
#
# Source-verified format (lora-aprs/LoRa_APRS_iGate, TaskRadiolib.cpp:83,123,
# peterus/APRS-Decoder-Lib Factory.cpp:generateHeader):
#
#   bytes 0..2 : MAGIC = 0x3c 0xff 0x01  ("<\xff\x01")
#   bytes 3..N : ASCII TNC2 string:  "SOURCE>DEST,PATH:TYPE+BODY"
#                where ">" separates source callsign from destination,
#                "," (optional) separates dest from comma-delimited path,
#                ":" separates header from body, and the byte AFTER ":" is the
#                APRS data-type identifier (':'=message, '!'/'='=position
#                without timestamp, '/'/'@'=position with timestamp, etc.).
#
# The 3-byte magic gives a structural false-positive rate of 1 in 2^24, so the
# parser ranks 'verified' immediately on a magic match — no behavioral gate.
# ============================================================================
_LORA_APRS_MAGIC = b'\x3c\xff\x01'

def parse_lora_aprs_packet(payload, rf=None):
    """Parse a LoRa APRS frame.  Returns proto='lora_aprs' record or None."""
    if len(payload) < 6:
        return None
    payload = bytes(payload)                     # list or bytes — normalise
    if not payload.startswith(_LORA_APRS_MAGIC):
        return None
    try:
        text = payload[3:].decode('ascii', errors='strict')
    except UnicodeDecodeError:
        return None
    # TNC2 requires "SOURCE>DEST...:body".  Reject anything without both.
    gt = text.find('>')
    colon = text.find(':', gt + 1) if gt > 0 else -1
    if gt < 1 or colon < 0 or gt > 9:           # callsign ≤ 9 chars (with SSID)
        return None
    source = text[:gt]
    # Source must be uppercase-alnum + optional "-SSID".  Quick check: must
    # contain at least one letter and contain only callsign-legal characters.
    _legal = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
    if not source or not any(c.isalpha() for c in source) or any(c not in _legal for c in source):
        return None
    after = text[gt + 1:colon]                    # dest[,path]
    comma = after.find(',')
    dest = after[:comma] if comma >= 0 else after
    path = after[comma + 1:] if comma >= 0 else ''
    body = text[colon + 1:]
    type_char = body[:1] if body else ''
    print("\n  --- LoRa APRS Frame ---")
    print("  Source: %s" % source)
    print("  Destination: %s" % dest)
    if path:
        print("  Path: %s" % path)
    print("  Type: %r" % type_char)
    print("  Body: %s" % (body[:80] + ('...' if len(body) > 80 else '')))
    return {
        'proto': 'lora_aprs',
        'confidence': 'verified',           # 24-bit magic + TNC2 structure
        'decrypted': True,
        'from': source,
        'to': dest,
        'path': path or None,
        'aprs_type': type_char,
        'text': body,
        'summary': '%s→%s: %s' % (source, dest, body[:40]),
    }


# ============================================================================
# Reticulum Network Stack (RNS) — packet format used by RNode and Reticulum
# transports running raw LoRa modulation.
#
# Source-verified format (markqvist/Reticulum RNS/Packet.py:242 unpack):
#
#   byte 0   (flags):
#     bit 7         : reserved (must be 0)
#     bit 6         : header_type   (0=HEADER_1 normal, 1=HEADER_2 transport)
#     bit 5         : context_flag
#     bit 4         : transport_type (0=BROADCAST, 1=TRANSPORT)
#     bits 3..2     : destination_type (0=SINGLE, 1=GROUP, 2=PLAIN, 3=LINK)
#     bits 1..0     : packet_type (0=DATA, 1=ANNOUNCE, 2=LINKREQUEST, 3=PROOF)
#   byte 1       : hops counter
#   if HEADER_2 : bytes 2..17 = transport_id (16-byte truncated SHA-256 hash)
#                 bytes 18..33 = destination_hash (16-byte truncated hash)
#                 byte 34 = context
#                 bytes 35.. = ciphertext / data
#   else        : bytes 2..17 = destination_hash (16-byte truncated hash)
#                 byte 18 = context
#                 bytes 19.. = ciphertext / data
#
# DST_LEN = RNS.Reticulum.TRUNCATED_HASHLENGTH/8 = 128/8 = 16 bytes.
# ============================================================================
_RNS_PACKET_TYPES = {0: 'DATA', 1: 'ANNOUNCE', 2: 'LINKREQUEST', 3: 'PROOF'}
_RNS_DEST_TYPES   = {0: 'SINGLE', 1: 'GROUP', 2: 'PLAIN', 3: 'LINK'}
_RNS_DST_LEN      = 16

def parse_reticulum_packet(payload, rf=None):
    """Parse a Reticulum-formatted LoRa packet.

    Only emits a match for ANNOUNCE packets.  Per Reticulum source
    (RNS/Identity.py constants):
        KEYSIZE  = 512 bits = 64 bytes (X25519 + Ed25519 concatenated)
        SIGLENGTH = 512 bits = 64 bytes (Ed25519)
        NAME_HASH_LENGTH = 80 bits = 10 bytes
        RATCHETSIZE = 256 bits = 32 bytes
    ANNOUNCE body = public_key(64) + name_hash(10) + random_hash(10)
                  + [ratchet(32) if context_flag set] + signature(64)
                  + app_data(≥1 byte if present)
    Minimum data length = 148 bytes (no ratchet) or 180 (with ratchet).

    Added entropy gates on pubkey + signature (real crypto fields cannot be
    all-zero or all-0xFF — both are explicit dead-key markers).  These drop
    the random-byte FP rate by ~100×.  Other packet types (DATA/LINKREQUEST/
    PROOF) are structurally indistinguishable from random — never claimed.
    """
    # Smallest valid frame: 2(flags+hops) + 16(dst_hash) + 1(context)
    # + 148(announce-no-ratchet) = 167 bytes.  Reject anything shorter.
    if len(payload) < 167:
        return None
    payload = bytes(payload)
    flags = payload[0]
    if flags & 0b10000000:                      # reserved bit must be 0
        return None
    header_type      = (flags & 0b01000000) >> 6
    context_flag     = (flags & 0b00100000) >> 5
    transport_type   = (flags & 0b00010000) >> 4
    destination_type = (flags & 0b00001100) >> 2
    packet_type      = (flags & 0b00000011)
    if packet_type != 1:                        # ANNOUNCE only
        return None
    hops = payload[1]
    if hops > 128:
        return None
    # Stock Reticulum only announces SINGLE or GROUP destinations.
    if destination_type not in (0, 1):
        return None
    if header_type == 1:                        # HEADER_2 — add transport_id
        if len(payload) < 3 + 2 * _RNS_DST_LEN + 148:
            return None
        transport_id     = payload[2:2 + _RNS_DST_LEN]
        destination_hash = payload[2 + _RNS_DST_LEN:2 + 2 * _RNS_DST_LEN]
        context          = payload[2 + 2 * _RNS_DST_LEN]
        data             = payload[3 + 2 * _RNS_DST_LEN:]
    else:
        transport_id     = None
        destination_hash = payload[2:2 + _RNS_DST_LEN]
        context          = payload[2 + _RNS_DST_LEN]
        data             = payload[3 + _RNS_DST_LEN:]
    # Decompose announce body per Identity.py:535-538.
    KEYSIZE = 64; NAME_HASH_LEN = 10; RANDOM_HASH_LEN = 10
    SIGLEN = 64; RATCHETSIZE = 32
    public_key = data[:KEYSIZE]
    if len(data) < KEYSIZE + NAME_HASH_LEN + RANDOM_HASH_LEN + SIGLEN:
        return None
    name_hash   = data[KEYSIZE:KEYSIZE + NAME_HASH_LEN]
    random_hash = data[KEYSIZE + NAME_HASH_LEN:KEYSIZE + NAME_HASH_LEN + RANDOM_HASH_LEN]
    # Ratchet present only if context_flag set in the header.
    if context_flag:
        sig_off  = KEYSIZE + NAME_HASH_LEN + RANDOM_HASH_LEN + RATCHETSIZE
        if len(data) < sig_off + SIGLEN:
            return None
        ratchet   = data[KEYSIZE + NAME_HASH_LEN + RANDOM_HASH_LEN:sig_off]
        signature = data[sig_off:sig_off + SIGLEN]
        app_data  = data[sig_off + SIGLEN:]
    else:
        ratchet   = None
        sig_off   = KEYSIZE + NAME_HASH_LEN + RANDOM_HASH_LEN
        signature = data[sig_off:sig_off + SIGLEN]
        app_data  = data[sig_off + SIGLEN:]
    # Entropy gates — real Ed25519/X25519 keys + sigs are high-entropy.
    # Random byte sequences that pass structural gates would hit these
    # ~all the time, so they're a very tight FP filter.
    def _high_entropy(b, min_distinct=12):
        return len(set(b)) >= min_distinct
    if not _high_entropy(public_key, 20):       # 64 random bytes → ~58 distinct expected
        return None
    if not _high_entropy(signature, 20):
        return None
    if not _high_entropy(random_hash, 5):       # 10 bytes → ~9 distinct expected
        return None
    print("\n  --- Reticulum ANNOUNCE ---")
    print("  Header: %d  DestType: %s  Hops: %d  Ratchet: %s" % (
        header_type + 1, _RNS_DEST_TYPES[destination_type], hops,
        'yes' if ratchet else 'no'))
    print("  DestinationHash: %s" % destination_hash.hex())
    print("  PublicKey:       %s..." % public_key[:8].hex())
    if transport_id is not None:
        print("  TransportID:     %s" % transport_id.hex())
    print("  Context: 0x%02X  AppData: %d bytes" % (context, len(app_data)))
    return {
        'proto': 'reticulum',
        'confidence': 'verified',               # all crypto-field shapes match
        'decrypted': False,
        'from': public_key[:8].hex(),           # first 8 bytes of pubkey as ID
        'to': destination_hash.hex(),
        'rns_header': header_type + 1,
        'rns_packet_type': 'ANNOUNCE',
        'rns_dest_type': _RNS_DEST_TYPES[destination_type],
        'rns_context': context,
        'rns_hops': hops,
        'rns_pubkey': public_key.hex(),
        'rns_ratchet': bool(ratchet),
        'summary': 'RNS ANNOUNCE → %s · %s · %dB app' % (
            destination_hash.hex()[:8],
            _RNS_DEST_TYPES[destination_type], len(app_data)),
    }


# ============================================================================
# disaster.radio (sudomesh) — LoRa mesh for disaster-recovery deployments.
#
# Source-verified format (sudomesh/disaster-radio Wiki/Protocol):
#
#   byte 0      : ttl              (1..16 typical)
#   byte 1      : totalLength      (entire packet length incl. header)
#   bytes 2..5  : sender (4)       (last-hop relay address)
#   bytes 6..9  : receiver (4)     (next-hop / broadcast 0xFFFFFFFF)
#   byte 10     : sequence
#   bytes 11..14: source (4)       (originating node)
#   byte 15     : hopCount         (incremented per relay)
#   byte 16     : metric           (link quality, sender→source)
#   bytes 17+   : datagram (Layer 3): 4 dst + 1 type + message
# ============================================================================
def parse_disaster_radio_packet(payload, rf=None):
    """Parse a disaster.radio Layer 2 packet.  Returns proto='disaster_radio' or None."""
    if len(payload) < 17 + 5:                   # L2 header + minimum L3 stub
        return None
    payload = bytes(payload)                     # list or bytes — normalise
    ttl          = payload[0]
    total_length = payload[1]
    # Strong structural gate: totalLength byte MUST equal len(payload).
    # totalLength is a uint8, so it caps at 255.  Allow ±2 for occasional PHY
    # padding artifacts (LoRa pads to symbol boundary at low SF / high CR).
    if not (abs(total_length - len(payload)) <= 2):
        return None
    if ttl == 0 or ttl > 32:                    # zero is impossible; >32 implausible
        return None
    sender    = payload[2:6]
    receiver  = payload[6:10]
    sequence  = payload[10]
    source    = payload[11:15]
    hop_count = payload[15]
    metric    = payload[16]
    if hop_count > 32:                          # implausible
        return None
    # All-zero source is invalid (would mean "no originator").
    if source == b'\x00\x00\x00\x00':
        return None
    # Datagram (Layer 3)
    datagram = payload[17:]
    dst_l3   = datagram[:4]
    dgram_type = datagram[4]
    message  = datagram[5:]
    src_hex = sender.hex().upper()
    src_orig_hex = source.hex().upper()
    rcv_hex = receiver.hex().upper()
    dst_hex = dst_l3.hex().upper()
    is_bcast_l2 = (receiver == b'\xff\xff\xff\xff')
    is_bcast_l3 = (dst_l3   == b'\xff\xff\xff\xff')
    print("\n  --- disaster.radio Packet ---")
    print("  TTL: %d  Length: %d  Sequence: %d  HopCount: %d  Metric: %d" % (
        ttl, total_length, sequence, hop_count, metric))
    print("  L2 sender→receiver: %s → %s" % (
        src_hex, 'BROADCAST' if is_bcast_l2 else rcv_hex))
    print("  L3 source→dest:     %s → %s  (type 0x%02X, %d bytes)" % (
        src_orig_hex, 'BROADCAST' if is_bcast_l3 else dst_hex,
        dgram_type, len(message)))
    return {
        'proto': 'disaster_radio',
        'confidence': 'verified',               # totalLength check is very tight
        'decrypted': True,
        'from': src_orig_hex,                   # ORIGINAL source (not last-hop)
        'to': 'BROADCAST' if is_bcast_l3 else dst_hex,
        'dr_sender': src_hex,                   # last-hop relay
        'dr_receiver': 'BROADCAST' if is_bcast_l2 else rcv_hex,
        'dr_ttl': ttl,
        'dr_hop_count': hop_count,
        'dr_sequence': sequence,
        'dr_type': dgram_type,
        'hops': hop_count,
        'summary': '%s→%s · TTL %d · hops %d' % (
            src_orig_hex, 'BCAST' if is_bcast_l3 else dst_hex, ttl, hop_count),
    }


# ============================================================================
# EByte E-series LoRa modules — Fixed transmission broadcast mode.
#
# The EByte E22 (SX1262/SX1268), E220 (LLCC68), and E32 (SX1276/SX1278) modules
# are extremely common UART-controlled LoRa transceivers from Chengdu Ebyte
# Electronic Technology, sold worldwide for hobby and small-commercial IoT.
# The de-facto Arduino driver is xreef/EByte_LoRa_E22_Series_Library (and
# matching E220/E32 variants).
#
# In "Fixed transmission" mode, the over-the-air packet structure (verified
# from LoRa_E22.cpp:900 — `sendStruct((uint8_t *)fixedStransmission, size+3)`
# on a struct of { byte ADDH; byte ADDL; byte CHAN; byte message[]; }):
#
#   byte 0    : ADDH      destination address high byte (0xFF for broadcast)
#   byte 1    : ADDL      destination address low byte  (0xFF for broadcast)
#   byte 2    : CHAN      module channel (0..83 typical for 900 MHz variants)
#   bytes 3+  : user payload (whatever the application passed)
#
# The `sendBroadcastFixedMessage(CHAN, msg)` helper hardcodes ADDH=ADDL=0xFF
# (LoRa_E22.cpp:859, 958).  Receivers compare their own configured ADDH/ADDL
# /CHAN to the prefix; broadcast matches everyone on the same CHAN.
#
# Validator: 3-byte broadcast prefix + ≥1 readable user data byte.  Payload
# detection accepts both pure-ASCII telemetry and binary-with-printable-tail
# (very common when firmware prepends a type/version byte before ASCII data).
# ============================================================================
_EBYTE_BROADCAST = b'\xff\xff'

def parse_lora_p2p_packet(payload, rf=None):
    """Parse an EByte E22/E220/E32 fixed-mode broadcast frame.

    Returns proto='ebyte_lora' (confirmed when user payload is ASCII-rich)
    or None when the structure doesn't match.
    """
    if len(payload) < 4:                        # 3-byte prefix + ≥1 data byte
        return None
    payload = bytes(payload)
    # ADDH+ADDL MUST be 0xFFFF (the EByte broadcast convention).
    if not payload.startswith(_EBYTE_BROADCAST):
        return None
    chan = payload[2]
    # CHAN is a small integer (0..83 for 900-MHz E22 variants, 0..82 for E220,
    # 0..31 for E32 at 433/470 MHz).  Reject values that are clearly outside
    # any plausible channel range — filters random-byte false positives.
    if chan > 83:
        return None
    user = payload[3:]
    if len(user) < 1:
        return None
    # Find the first printable-ASCII run in the user payload.  Firmware often
    # prepends 1-2 non-ASCII bytes (type/version/length) before the readable
    # telemetry — accept those, surface the readable suffix as the text.
    text_start = 0
    for i, b in enumerate(user[:4]):            # at most 4 prefix bytes
        if 32 <= b < 127:
            text_start = i
            break
    else:
        # No ASCII found in first 4 bytes — could still be a binary protocol.
        # Require at least 50% printable in the FULL user payload.
        printable_full = sum(1 for b in user if 32 <= b < 127)
        if printable_full / len(user) < 0.50:
            return None
        text_start = 0
    text_bytes = user[text_start:]
    printable = sum(1 for b in text_bytes if 32 <= b < 127)
    ratio = printable / len(text_bytes) if text_bytes else 0.0
    # Require the text portion to be ≥80% printable for a confirmed match.
    if ratio < 0.50:
        return None
    try:
        text = text_bytes.decode('ascii', errors='strict')
    except UnicodeDecodeError:
        text = ''.join(chr(b) if 32 <= b < 127 else '.' for b in text_bytes)
    # Surface any leading non-ASCII bytes as a separate "type" hex string.
    type_bytes = user[:text_start]
    print("\n  --- EByte E-series Fixed-Mode Frame ---")
    print("  Destination: BROADCAST (0xFFFF)")
    print("  Channel: %d" % chan)
    if type_bytes:
        print("  Type/version bytes: %s" % type_bytes.hex())
    print("  User payload (%d bytes, %d%% ASCII): %r" % (
        len(user), int(ratio*100), text))
    is_confirmed = (ratio >= 0.80)
    return {
        'proto': 'ebyte_lora' if is_confirmed else 'unknown',
        'hint': None if is_confirmed else 'ebyte_lora',
        'confidence': 'confirmed' if is_confirmed else 'candidate',
        'decrypted': True,                      # the L2 header is plaintext
        'from': None,                           # E22 fixed-mode broadcasts
                                                # don't carry sender ID over
                                                # the air — physical-layer
                                                # fingerprinting needed for
                                                # per-device identity.
        'to': 'BROADCAST',
        'ebyte_chan': chan,
        'ebyte_type': type_bytes.hex() if type_bytes else None,
        'text': text,
        'summary': 'EByte BCAST ch%d: %s' % (chan, text[:48]),
    }


# ============================================================================
# RadioHead RH_RF95 — the most common hobbyist LoRa library (Adafruit, Sparkfun,
# Pololu starter kits).  Air format is RIDICULOUSLY thin — just a 4-byte header
# inside the LoRa payload — with no magic bytes, no checksum, no version field.
# Source: airspayce RadioHead docs + RH_RF95.h (RH_RF95_HEADER_LEN = 4).
#
#   byte 0  : TO    (destination address;     0xFF = broadcast)
#   byte 1  : FROM  (source address;          0xFF = anonymous-broadcast TX)
#   byte 2  : ID    (message id, 0..255 incrementing)
#   byte 3  : FLAGS (low nibble = app reserved; high nibble = RH reserved)
#   bytes 4+: message payload (0..251 bytes)
#
# FALSE-POSITIVE RISK: VERY HIGH.  The 4-byte header has no magic, no version,
# no checksum — ANY 4-byte sequence where byte 3 has its high nibble zero will
# pass.  ~6% of random byte sequences satisfy that.  So we NEVER promote this
# to proto='radiohead' — only to proto='unknown' with hint='radiohead', which
# the UI surfaces as "looks like RadioHead but unconfirmed."  Behavioral
# promotion to confirmed would require seeing the SAME from-address twice with
# matching sequential IDs (RadioHead's only intrinsic signal of identity).
# ============================================================================
_RADIOHEAD_PENDING = {}                         # from_addr -> [last_msg_id, count]
_KNOWN_RADIOHEAD_NODES = set()                  # from_addrs with 2+ sequential ids

def parse_radiohead_packet(payload, rf=None):
    """Parse a RadioHead RH_RF95 frame.

    Returns proto='unknown' hint='radiohead' as 'candidate' unless the same
    from-address has been seen before with a sequential message id, in which
    case promote to proto='radiohead' / 'confirmed'.
    """
    if len(payload) < 5:                        # header + ≥1 byte data
        return None
    payload = bytes(payload)                     # list or bytes — normalise
    to_addr   = payload[0]
    from_addr = payload[1]
    msg_id    = payload[2]
    flags     = payload[3]
    # RadioHead RH_RF95_FLAGS_RESERVED is the high nibble; conventional usage
    # leaves it 0.  Reject if any RH-reserved bit is set.
    if flags & 0xF0:
        return None
    data = payload[4:]
    # Reject all-zero / all-same data (very common in noise/random garbage).
    if len(set(data)) <= 1:
        return None
    # Reject if FROM is 0 or 0xFF — anonymous TX is rare in practice and 0 is
    # reserved.  Real RadioHead apps assign each node a 1..253 address.
    if from_addr in (0x00, 0xFF):
        return None
    # TO can be broadcast (0xFF) or a real node 1..253.  Reject TO=0.
    if to_addr == 0x00:
        return None
    # Behavioral promotion: track this from_addr.  RadioHead's 4-byte header has
    # no magic, so a single structural match has a ~5% random FP rate — too high
    # to emit anything.  We track candidates silently in _RADIOHEAD_PENDING and
    # only emit once we've seen the SAME from_addr twice with a sequential
    # message ID (RH apps increment ID by 1 per packet).  Two sequential IDs from
    # the same source address are statistically near-impossible from random
    # bytes (~1 in 4000 even after the structural gate).
    prev = _RADIOHEAD_PENDING.get(from_addr)
    is_confirmed = from_addr in _KNOWN_RADIOHEAD_NODES
    if prev is not None:
        prev_id, _ = prev
        delta = (msg_id - prev_id) & 0xFF
        if 1 <= delta <= 16:
            _KNOWN_RADIOHEAD_NODES.add(from_addr)
            is_confirmed = True
    _RADIOHEAD_PENDING[from_addr] = [msg_id, (prev[1] + 1 if prev else 1)]
    # First sighting (or non-sequential): silently track state, don't emit.
    # Surfacing every structural match as a hint floods the UI with noise.
    if not is_confirmed:
        return None
    is_bcast_to = (to_addr == 0xFF)
    print("\n  --- RadioHead RF95 Frame ---")
    print("  TO: %s  FROM: %d  ID: %d  FLAGS: 0x%02X  data: %d bytes" % (
        'BROADCAST' if is_bcast_to else str(to_addr),
        from_addr, msg_id, flags, len(data)))
    print("  Behavioral: node %d confirmed via sequential ID" % from_addr)
    return {
        'proto': 'radiohead',
        'confidence': 'confirmed',
        'decrypted': True,
        'from': str(from_addr),
        'to': 'BROADCAST' if is_bcast_to else str(to_addr),
        'rh_id': msg_id,
        'rh_flags': flags,
        'summary': 'RH %s→%s id=%d %dB' % (
            from_addr, 'BC' if is_bcast_to else to_addr, msg_id, len(data)),
    }


# ============================================================================
# Per-packet hardware fingerprint — RF-layer features that identify the
# physical transmitter device (not the protocol, not the firmware).  Each
# feature is a per-radio-unit characteristic that's:
#   - STABLE across packets from the same device (within session)
#   - INDEPENDENT of position/orientation (no RSSI dependence)
#   - INDEPENDENT of payload content (geometric/statistical, not data-derived)
#   - INDEPENDENT of inter-packet timing (no periodicity assumption)
#
# Used by the web layer's device-clustering: within one protocol family
# (matching prefix bytes), packets are grouped by nearest-neighbor match on
# this feature vector to estimate the number of distinct transmitting devices.
#
# Features extracted:
#   dc_i, dc_q     — receiver chain DC offset.  A direct-conversion radio leaks
#                    its LO into baseband as a DC term.  Chirp modulation
#                    averages to ~0 over many symbols (the constant-envelope
#                    signal traces a circle in IQ plane centered at the DC
#                    offset), so mean(iq) ≈ hardware DC carrier feedthrough.
#                    Per-device, per-channel — different IC dies have different
#                    DC leakage.  Stable across motion (it's a hardware bias).
#   iq_amp_imb     — amplitude imbalance between I and Q channels.  Ideal:
#                    std(I) == std(Q).  Real radios have ±0.5–5% mismatch from
#                    the analog frontend.  Computed as (std_I - std_Q) / (std_I + std_Q).
#                    Per-die.  Robust to gain settings (it's a ratio).
#   iq_phase_imb   — phase imbalance: I and Q should be exactly 90° out of phase.
#                    Hardware mismatch leaks ~0.5–5° of phase error, showing as
#                    non-zero normalized correlation between I and Q.  Per-die.
# ============================================================================
def _extract_tx_fingerprint(iq1, preamble_start, sf, bw):
    """Sample-precise UMOP (Unintentional Modulation On Pulse) fingerprint.

    Returns None when LORA_FINGERPRINT=0 (low-core fast path: skip the per-
    packet FFT/feature work — saves ~10-15 % decode CPU, at the cost of the
    web UI's device-clustering / Mystery Devices attribution).

    Every transmitter imprints subtle per-die signatures into its signal that
    are NOT part of the intended modulation — these are the "unintentional
    modulations" that make each radio physically unique.  We extract them
    from the 8-symbol LoRa preamble using sample-precise alignment from the
    decoder's preamble_start.

    Features (all per-chip, all measured during the same constant-envelope
    preamble so RX/channel effects cancel):

      Tier 1 — crystal (TCXO):
        precise_carrier_hz  Mean CFO via coherent sum (sub-Hz precision).
        cfo_per_sym_std     STD of per-symbol CFO estimates (Hz). Short-term
                            crystal cycle-to-cycle jitter.  Per-crystal.

      Tier 2 — power amplifier:
        am_ripple_pct       std(envelope)/mean(envelope) across full preamble.
                            LoRa is constant-envelope by design → any ripple
                            is PA AM-AM distortion + supply rail noise.
                            Per-die.
        amp_per_sym_pct     std/mean of per-symbol mean amplitudes.  Captures
                            PA gain stability across the 8 chirps.  Per-die.

      Tier 3 — PLL:
        phase_residual_rms  RMS of phase residual after subtracting linear
                            (CFO) trend.  Phase noise integrated over the
                            preamble.  Per-VCO.
        irr_db              Image rejection ratio: main peak vs mirror peak
                            on coherent FFT.  TX I/Q mixer balance.  Per-die.

      Quality:
        signal_snr_db       Peak vs noise-floor on dechirped spectrum.  Used
                            by the web classifier to REJECT low-quality
                            samples from profile training (concurrent-packet
                            contamination, partial bursts → garbage features).
    """
    if not _FINGERPRINT_ON:
        return None
    N_sym = 1 << sf
    if iq1 is None or len(iq1) < preamble_start + 9 * N_sym:
        return None
    # iq1 is decimated to BW rate → 1 sample per chip.
    t = np.arange(N_sym, dtype=np.float64) / bw
    Ts = N_sym / bw
    upchirp = np.exp(1j * np.pi * bw / Ts * t**2).astype(np.complex64)
    downchirp = np.conj(upchirp)

    Nfft = 65536
    freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1.0 / bw))
    bin_w = bw / Nfft

    # --- Half-symbol alignment refinement ---
    # Schmidl-Cox finds preamble_start with one-symbol resolution but the
    # offset within a symbol can be off by ANY fraction.  Empirically, at
    # BW=250 (SHORT_FAST) the preamble_start is consistently off by N/2 samples,
    # putting the dechirped CW exactly at ±BW/2 (Nyquist boundary) where peak
    # location aliases between the two ends.  Search a fine grid of sub-symbol
    # offsets and pick the one that PUTS THE PEAK AWAY FROM THE BOUNDARY (max
    # of |freq| < bw/2 - 5kHz preferred) AND has the highest peak amplitude.
    # PERF: the SCAN only uses peak amplitude + freq location for scoring, so a
    # small Nfft is sufficient (122 Hz/bin at bw=500k vs 7.6 Hz/bin at 65536 —
    # both far finer than the ~5 kHz boundary threshold).  Final feature
    # extraction below uses the full Nfft=65536 unchanged.  Also batches the
    # 8 per-symbol FFTs into ONE call (pocketfft avoids 7 dispatch overheads).
    Nfft_scan = max(1024, 1 << (N_sym - 1).bit_length())   # ≥ N_sym, capped low
    Nfft_scan = min(Nfft_scan, 4096)                       # never larger than 4096
    freqs_scan = np.fft.fftshift(np.fft.fftfreq(Nfft_scan, d=1.0 / bw))
    boundary_dist_thresh = max(5000.0, bw * 0.05)
    best_offset = 0
    best_score = -1.0
    iq_len = len(iq1)
    for off in range(-N_sym // 2, N_sym // 2 + 1, max(1, N_sym // 16)):
        s_start = preamble_start + off
        if s_start < 0 or s_start + 8 * N_sym > iq_len:
            continue
        # 8 symbol windows × N_sym samples → (8, N_sym), one batched FFT.
        segs8 = iq1[s_start:s_start + 8 * N_sym].reshape(8, N_sym) * downchirp
        F = np.fft.fft(segs8, n=Nfft_scan, axis=1)
        sF_scan = np.fft.fftshift(F.sum(axis=0))
        cmag = np.abs(sF_scan)
        pk_idx = int(np.argmax(cmag))
        pk_freq = freqs_scan[pk_idx]
        pk_amp = float(cmag[pk_idx])
        boundary_dist = bw / 2 - abs(pk_freq)
        boundary_penalty = 1.0 if boundary_dist > boundary_dist_thresh else (boundary_dist / boundary_dist_thresh)
        score = pk_amp * boundary_penalty
        if score > best_score:
            best_score = score
            best_offset = off
    aligned_start = preamble_start + best_offset
    if aligned_start < 0 or aligned_start + 8 * N_sym > iq_len:
        return None

    # --- Dechirp all 8 preamble symbols at the BEST alignment ---
    # Batched FFT replaces 8 individual fft+fftshift calls (single pocketfft
    # dispatch over the 8 rows, identical result, ~5-8× faster on (8, 65536)).
    segs_mat = iq1[aligned_start:aligned_start + 8 * N_sym].reshape(8, N_sym) * downchirp  # (8, N_sym)
    F_full = np.fft.fftshift(np.fft.fft(segs_mat, n=Nfft, axis=1), axes=1)                 # (8, Nfft)
    per_F_mag_mat = np.abs(F_full)
    sum_F = F_full.sum(axis=0)
    sum_mag = per_F_mag_mat.sum(axis=0)
    # Per-symbol amplitude (time-domain mean |seg|) and per-symbol CFO via
    # quadratic sub-bin peak interp.  Vectorised peak search + scalar interp
    # for the 8 entries (cheap; preserves the exact per-bin arithmetic).
    per_sym_amp = np.abs(segs_mat).mean(axis=1).astype(np.float64)
    pk_s_per = np.argmax(per_F_mag_mat, axis=1)
    per_sym_cfo = [0.0] * 8
    for s in range(8):
        pk_s = int(pk_s_per[s])
        if 1 <= pk_s < Nfft - 1:
            a = float(per_F_mag_mat[s, pk_s - 1])
            b = float(per_F_mag_mat[s, pk_s])
            c = float(per_F_mag_mat[s, pk_s + 1])
            denom = a - 2 * b + c
            d_s = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
            per_sym_cfo[s] = float(freqs[pk_s] + d_s * bin_w)
        else:
            per_sym_cfo[s] = float(freqs[pk_s]) if 0 <= pk_s < Nfft else 0.0
    per_sym_amp = list(per_sym_amp)
    per_sym_dechirped = [segs_mat[s] for s in range(8)]

    # --- Tier 1: crystal features ---
    coh_mag = np.abs(sum_F)
    main_pk = int(np.argmax(coh_mag))
    if not (1 <= main_pk < Nfft - 1):
        return None
    main_amp = float(coh_mag[main_pk])
    a, b, c = float(coh_mag[main_pk-1]), main_amp, float(coh_mag[main_pk+1])
    denom = a - 2*b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
    precise_carrier_hz = float(freqs[main_pk] + delta * bin_w)
    cfo_per_sym_std = float(np.std(per_sym_cfo))

    # --- Tier 2: PA features (AM ripple + inter-symbol amplitude) ---
    all_dechirped = np.concatenate(per_sym_dechirped)
    env = np.abs(all_dechirped)
    env_mean = float(np.mean(env))
    am_ripple_pct = float(np.std(env) / max(env_mean, 1e-9)) * 100.0
    amp_per_sym_pct = float(np.std(per_sym_amp) / max(float(np.mean(per_sym_amp)), 1e-9)) * 100.0

    # --- Tier 3: PLL features (phase residual + IRR) ---
    # Phase residual: apply mean-CFO correction, then look at deviations
    cfo_rad_per_sample = 2 * np.pi * precise_carrier_hz / bw
    t_full = np.arange(len(all_dechirped), dtype=np.float64)
    corrected = all_dechirped * np.exp(-1j * cfo_rad_per_sample * t_full).astype(np.complex64)
    # Only use samples where amplitude is decent (else phase is meaningless)
    keep = env > 0.3 * env_mean
    if keep.sum() < 64:
        phase_residual_rms = None
    else:
        ph = np.unwrap(np.angle(corrected[keep]))
        t_keep = t_full[keep]
        # Detrend (residual CFO)
        try:
            poly = np.polyfit(t_keep, ph, 1)
            residual = ph - np.polyval(poly, t_keep)
            phase_residual_rms = float(np.sqrt(np.mean(residual ** 2)))
        except Exception:
            phase_residual_rms = None

    # IRR: image peak at -main_freq
    target_mirror = -freqs[main_pk]
    mc = int(np.argmin(np.abs(freqs - target_mirror)))
    lo = max(1, mc - 30); hi = min(Nfft - 1, mc + 30)
    mir_pk = lo + int(np.argmax(coh_mag[lo:hi+1]))
    mir_amp = float(coh_mag[mir_pk])
    irr_db = float(20.0 * np.log10(main_amp / max(mir_amp, 1e-9))) if mir_amp > 0 else None

    # PN slope (legacy from earlier work — keep for now, classifier auto-weights)
    log_off, log_pow = [], []
    for off_hz in (500.0, 1000.0, 2000.0, 5000.0, 10000.0):
        ob = int(off_hz / bin_w)
        if main_pk + ob < Nfft and main_pk - ob >= 0:
            if abs((main_pk + ob) - mir_pk) > 30 and abs((main_pk - ob) - mir_pk) > 30:
                p = (sum_mag[main_pk + ob] + sum_mag[main_pk - ob]) / 2.0
                if p > 0:
                    log_off.append(np.log10(off_hz)); log_pow.append(np.log10(p))
    pn_slope = None
    if len(log_off) >= 3:
        try: pn_slope = float(np.polyfit(log_off, log_pow, 1)[0])
        except Exception: pn_slope = None

    # --- Quality: SNR of dechirped peak vs noise floor ---
    mask = np.ones(Nfft, dtype=bool)
    nb = 200  # exclude bins near peak AND near mirror
    mask[max(0, main_pk - nb):min(Nfft, main_pk + nb)] = False
    mask[max(0, mir_pk - nb):min(Nfft, mir_pk + nb)] = False
    noise_floor = float(np.median(coh_mag[mask])) if mask.any() else 1.0
    signal_snr_db = float(20.0 * np.log10(main_amp / max(noise_floor, 1e-9)))

    return {
        'precise_carrier_hz': precise_carrier_hz,
        'cfo_per_sym_std': cfo_per_sym_std,
        'am_ripple_pct': am_ripple_pct,
        'amp_per_sym_pct': amp_per_sym_pct,
        'phase_residual_rms': phase_residual_rms,
        'irr_db': irr_db,
        'pn_slope': pn_slope,
        'signal_snr_db': signal_snr_db,
    }


def _compute_hw_fingerprint(iq, sf=None, bw=None, fs=None):
    """Per-packet hardware fingerprint.

    Returns dict with:
      precise_carrier_hz — absolute CFO of the transmit carrier relative to
                           the capture's baseband, measured directly from the
                           preamble via downchirp demodulation + FFT peak
                           with quadratic sub-bin interpolation.  Sub-Hz
                           precision.  Independent of the decoder's pipeline
                           (which gets polluted by recenter passes).

    Compute time ≈ 80-150 μs for typical capture sizes.

    Returns None when LORA_FINGERPRINT=0 (low-core fast path).
    """
    if not _FINGERPRINT_ON:
        return None
    n = len(iq)
    if n < 128:
        return None
    # Precise transmit carrier frequency.  Multiply preamble upchirps by an
    # ideal downchirp → the product is a continuous-wave tone at frequency =
    # CFO.  FFT → quadratic peak interpolation → sub-bin precision.  Far more
    # robust than spectral centroid (which mixes upchirps with SFD downchirps
    # and payload modulation, biasing the result).  This is the standalone
    # CFO-measurement technique used in the project's RF-analysis tools and
    # gives ~Hz precision when SNR > 10 dB.
    precise_carrier_hz = None
    if sf is not None and bw is not None and fs is not None:
        try:
            N_sym = 1 << sf
            sps = max(1, int(round(fs / bw)))
            Nsamp = N_sym * sps
            # Build ideal downchirp at the actual sample rate.
            t_idx = np.arange(Nsamp, dtype=np.float64)
            f_inst = -bw / 2.0 + (bw / Nsamp) * t_idx
            phi = 2.0 * np.pi * np.cumsum(f_inst) / fs
            downchirp = np.exp(-1j * phi).astype(np.complex64)
            # Coherently sum |dechirp FFT| across multiple preamble symbols
            # for noise averaging.  Caller is expected to have windowed to
            # start at the first preamble upchirp.
            Nfft = max(4096, 4 * Nsamp)
            # Search the FULL spectrum for the dechirped peak.  No restricted
            # window — the burst can be anywhere within ±bw/2 of baseband,
            # depending on where the gate centered (or whether the gate
            # centered at all, in the case of raw IQ slices).  We accumulate
            # the magnitude spectrum across all 8 preamble symbols (coherent
            # averaging of squared magnitudes — works for symbol-incoherent
            # CFOs because the dechirped TONE is at the same freq every symbol).
            # The summed peak is much sharper than a single-symbol peak (8×
            # noise averaging) and the strongest peak is the true preamble CFO.
            sum_mag = None
            for sym_i in range(min(8, max(1, len(iq) // Nsamp))):
                start = sym_i * Nsamp
                if start + Nsamp > len(iq):
                    break
                seg = iq[start:start + Nsamp] * downchirp
                F = np.fft.fftshift(np.fft.fft(seg, Nfft))
                m = np.abs(F)
                if sum_mag is None:
                    sum_mag = m
                else:
                    sum_mag = sum_mag + m
            if sum_mag is not None:
                pk = int(np.argmax(sum_mag))
                if 1 <= pk < len(sum_mag) - 1:
                    freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1.0 / fs))
                    # Quadratic sub-bin interpolation directly on the SUMMED
                    # magnitude spectrum — same units, better SNR than any
                    # single symbol's interpolation.
                    a = float(sum_mag[pk - 1])
                    b = float(sum_mag[pk])
                    c = float(sum_mag[pk + 1])
                    denom = (a - 2 * b + c)
                    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
                    precise_carrier_hz = float(freqs[pk] + delta * (fs / Nfft))
        except Exception:
            precise_carrier_hz = None

    # ---- UMOP: burst-onset transient ---------------------------------------
    # The PA ramp-up at the start of every transmission is hardware-specific
    # and orthogonal to TCXO frequency.  Validated empirically: onset features
    # have SNR ~1.0-1.4 between two same-batch RAK Meshtastic devices —
    # comparable to or better than abs_tx_hz on a single test (whose SNR
    # collapses when TCXO drifts during the observation window).  Combined
    # with abs_tx_hz they reach ~95% per-packet classification accuracy.
    #
    #   onset_rise_us       Time from 10%→90% envelope (PA slew rate).
    #   onset_overshoot_pct PA peak overshoot above steady-state (regulator
    #                       transient response — STRONGEST single feature).
    #   onset_mid_slope     Envelope derivative at 40-60% point.
    #
    # All measured on the full-rate iq BEFORE any decimation (~60 μs rise is
    # 30 samples at 500 ksps, ~7 samples after decimation — would lose
    # resolution).  Only computed when fs >= 250 kHz.
    onset = None
    try:
        if fs is not None and fs >= 250_000 and n >= 2000:
            env = np.abs(iq).astype(np.float32)
            smooth_n = max(8, int(fs * 20e-6))   # 20μs averaging
            if len(env) >= smooth_n * 4:
                env_s = np.convolve(env, np.ones(smooth_n, dtype=np.float32)/smooth_n, mode='same')
                peak = float(np.max(env_s))
                if peak > 1e-6:
                    above_50 = np.where(env_s > peak * 0.5)[0]
                    if len(above_50) >= 3:
                        burst_start = int(above_50[0])
                        if smooth_n * 3 <= burst_start <= len(env_s) - smooth_n * 3:
                            p10 = peak * 0.10; p90 = peak * 0.90
                            i10 = burst_start
                            while i10 > 0 and env_s[i10] > p10: i10 -= 1
                            i90 = burst_start
                            while i90 < len(env_s) - 1 and env_s[i90] < p90: i90 += 1
                            if 2 <= (i90 - i10) <= smooth_n * 20:
                                rise_us = (i90 - i10) / fs * 1e6
                                steady_end = min(i90 + 400, len(env_s))
                                if steady_end - i90 >= 100:
                                    steady_mean = float(np.mean(env_s[i90:steady_end]))
                                    peak_in_window = float(np.max(env_s[i10:min(i10+(i90-i10)*3, len(env_s))]))
                                    overshoot_pct = (peak_in_window - steady_mean) / max(steady_mean, 1e-9) * 100
                                    p40, p60 = peak * 0.4, peak * 0.6
                                    i40 = i10
                                    while i40 < len(env_s) - 1 and env_s[i40] < p40: i40 += 1
                                    i60 = i40
                                    while i60 < len(env_s) - 1 and env_s[i60] < p60: i60 += 1
                                    slope = (env_s[i60] - env_s[i40]) / max(i60 - i40, 1) / peak * 1e6 / fs
                                    onset = {
                                        'onset_rise_us': rise_us,
                                        'onset_overshoot_pct': overshoot_pct,
                                        'onset_mid_slope': slope,
                                    }
    except Exception:
        onset = None

    result = {'precise_carrier_hz': precise_carrier_hz}
    if onset:
        result.update(onset)
    return result


# ============================================================================
# Structured packet emission (machine-readable, consumed by the web UI / log)
# ============================================================================
def _emit_pkt(rec):
    """Emit one structured packet record as a single `[PKT] {json}` line.  The
    gate's BackgroundDecoder enriches it (freq/sf/bw/rssi from the capture name),
    dedups by (pktid,hop), and writes it to the JSONL log + the web UI.  Kept as
    a plain stdout line so it rides the existing decoder→gate→web stdout path."""
    try:
        import json as _json
        print("[PKT] " + _json.dumps(rec, separators=(',', ':'), default=str))
    except Exception:
        pass


_UNKNOWN_SEEN = set()   # per-process dedup of unknown-protocol frames (by content)

# ---- Known-node registry for chase plausibility (behavioral verification) ----
# Populated as clean-CRC Meshtastic packets emit (their src/dst are known-real
# nodes).  Used to gate chase recovery for encrypted DMs without a key: a
# random byte pattern that happens to pass CRC-16 AND look structurally like a
# Meshtastic header AND has its src/dst match a previously-seen real node is
# astronomically unlikely (combined ~1/2^80).  Bootstrap (cold start): until
# the first clean decode populates the set, chase falls back to strict
# broadcast-only gate so no FPs leak.
_KNOWN_NODES = set()
_KNOWN_NODES_LOCK = None
try:
    import threading as _thr_kn
    _KNOWN_NODES_LOCK = _thr_kn.Lock()
except Exception:
    pass

def _register_known_node(addr_int):
    """Add a Meshtastic node address (32-bit int) to the known set."""
    if not addr_int:
        return
    if addr_int == 0xFFFFFFFF:   # broadcast — not a node
        return
    if _KNOWN_NODES_LOCK is not None:
        with _KNOWN_NODES_LOCK:
            _KNOWN_NODES.add(int(addr_int) & 0xFFFFFFFF)
    else:
        _KNOWN_NODES.add(int(addr_int) & 0xFFFFFFFF)

def _is_chase_acceptable_uni(tr):
    """Chase recovery gate for unicast DMs (encrypted, no key for verification).
    Accept iff recovered header is structurally a valid unicast frame AND its
    dst OR src is a previously-seen real node.  Strict broadcast (dst==FFFFFFFF)
    is always accepted (matches the original chase gate).

    BOOTSTRAP: when _KNOWN_NODES is empty (fresh deployment, no prior decodes),
    accept on STRUCTURAL grounds alone.  Subsequent decodes from the same nodes
    then bootstrap the known-set and the strict gate engages."""
    if len(tr) < 16:
        return False
    # Broadcast — always accept (original behavior)
    if tr[0] == 0xFF and tr[1] == 0xFF and tr[2] == 0xFF and tr[3] == 0xFF:
        return True
    # Unicast: require plausible structure
    dst_int = int(tr[0]) | (int(tr[1]) << 8) | (int(tr[2]) << 16) | (int(tr[3]) << 24)
    src_int = int(tr[4]) | (int(tr[5]) << 8) | (int(tr[6]) << 16) | (int(tr[7]) << 24)
    if dst_int == 0 or src_int == 0:
        return False
    if dst_int == src_int:   # nonsensical
        return False
    # Structural: hop_limit ≤ hop_start (Meshtastic invariant).
    # Meshtastic header layout: dst[0:4] src[4:8] pktid[8:12] flags[12] chan[13]
    flags = int(tr[12])
    hop_limit = flags & 0x07
    hop_start = (flags >> 5) & 0x07
    if hop_limit > hop_start:
        return False
    # Behavioral gate: at least one endpoint must be a known node.
    # BOOTSTRAP: if NO known nodes yet, accept on structural validity alone —
    # this is the only way the first-ever decode populates the set.
    snapshot = set(_KNOWN_NODES) if _KNOWN_NODES_LOCK is None else None
    if _KNOWN_NODES_LOCK is not None:
        with _KNOWN_NODES_LOCK:
            snapshot = set(_KNOWN_NODES)
    if not snapshot:
        return True   # bootstrap path
    return dst_int in snapshot or src_int in snapshot

# Known-node set auto-populates from `from` field of any successfully-decoded
# Meshtastic packet.  Every Meshtastic node periodically broadcasts NodeInfo,
# so this self-builds within minutes of mesh activity.


def _report_unknown(payload, sf, bw, cr, sync, name):
    """Log a clean-CRC LoRa frame that NO parser recognised (not Meshtastic /
    LoRaWAN / MeshCore) → a developer report of unimplemented protocols.  Opt-in
    via env LORA_UNKNOWN_REPORT (a path); a sibling '<path>.off' marker disables it
    live.  Plausible-Meshtastic-header frames never reach here (they're handled as
    encrypted), so this isolates genuinely unknown protocols."""
    path = os.environ.get('LORA_UNKNOWN_REPORT')
    if not path or os.path.exists(path + '.off'):
        return
    try:
        import json as _json
        raw = bytes(payload)
        key = raw[:64]
        if key in _UNKNOWN_SEEN:
            return
        _UNKNOWN_SEEN.add(key)
        if len(_UNKNOWN_SEEN) > 5000:
            _UNKNOWN_SEEN.clear()
        _freq = None
        for _p in name.lstrip('.').split('_'):
            if _p.endswith('MHz'):
                try: _freq = float(_p[:-3])
                except ValueError: pass
        rec = {'ts': time.time(), 'len': len(raw), 'hex': raw.hex(),
               'sf': sf, 'bw': bw, 'cr': cr,
               'sync': ('0x%02x' % sync) if sync is not None else None,
               'freq_mhz': _freq}
        with open(path, 'a') as f:
            f.write(_json.dumps(rec) + '\n')
    except Exception:
        pass


# ============================================================================
# Meshtastic packet parser + decryptor
# ============================================================================
def parse_meshtastic_packet(payload, aes_key=None, no_key=False, _clean_crc=False, _rf=None):
    """Parse and decrypt a Meshtastic packet from LoRa payload bytes."""
    if aes_key is None:
        aes_key = MESH_AES_KEY
    if len(payload) < 16:
        print("  [Meshtastic] Packet too short (%d bytes)" % len(payload))
        return

    p = payload
    dest = node_id_str(p)
    sender = node_id_str(p[4:])
    pkt_id = p[8] | (p[9]<<8) | (p[10]<<16) | (p[11]<<24)
    flags = p[12]
    chan_hash = p[13]

    is_broadcast = (p[0]==0xff and p[1]==0xff and p[2]==0xff and p[3]==0xff)
    hop_limit = flags & 0x07
    hop_start = (flags >> 5) & 7
    want_ack = bool((flags >> 3) & 1)
    via_mqtt = bool((flags >> 4) & 1)
    hops_taken = hop_start - hop_limit

    dest_str = "broadcast" if is_broadcast else dest

    print("\n  --- Meshtastic Packet ---")
    print("  From: %s  To: %s" % (sender, dest_str))
    flag_parts = "hop_limit=%d hop_start=%d hops_taken=%d" % (hop_limit, hop_start, hops_taken)
    if want_ack: flag_parts += " want_ack"
    if via_mqtt: flag_parts += " via_mqtt"

    # Structured record — header fields are in the UNENCRYPTED part, so they're
    # known even when the payload won't decrypt (a packet on a different channel
    # key).  Populated further below if the payload decrypts.
    # Initial confidence is 'candidate' (header structurally valid, key behavioral
    # gate also satisfied).  Upgraded to 'verified' below on successful AES decrypt
    # + valid protobuf portnum — the cryptographic proof tier.  Encrypted-only DMs
    # (clean CRC, can't decrypt with our key) stay at 'candidate' since they're
    # only structurally identifiable.
    _rec = {
        'proto': 'meshtastic', 'from': sender, 'to': dest_str,
        'pktid': '0x%08x' % pkt_id, 'hop_start': hop_start,
        'hop_limit': hop_limit, 'hops': hops_taken, 'chan': '0x%02x' % chan_hash,
        'want_ack': want_ack, 'via_mqtt': via_mqtt, 'decrypted': False,
        'confidence': 'candidate',
        'portnum': None, 'port_name': None, 'text': None,
        # Always carry the on-the-wire bytes so the UI can show RAW (hex+ASCII)
        # for every Meshtastic packet, regardless of decrypt success/failure.
        'raw_hex': bytes(payload).hex(),
    }
    if _rf:
        _rec.update({'sf': _rf.get('sf'), 'bw': _rf.get('bw'),
                     'freq_mhz': _rf.get('freq_mhz'), 'rssi': _rf.get('rssi')})
        # Hardware fingerprint flows through to the [PKT] record so the web
        # can cluster even named-protocol packets by device when useful.
        if _rf.get('hw_fp') is not None:
            _rec['hw_fp'] = _rf['hw_fp']
        if _rf.get('cfo_hz') is not None:
            _rec['cfo_hz'] = _rf['cfo_hz']
        if _rf.get('cfo_drift') is not None:
            _rec['cfo_drift'] = _rf['cfo_drift']
    # NOTE: the "PacketID:" line is the marker the pipeline counts as a decoded
    # packet.  Defer it until the payload DECRYPTS to a valid protobuf portnum
    # (below) — a CRC-16 coincidence on broadcast-dst bytes can pass CRC + the
    # broadcast gate but decrypt to garbage (portnum < 0).  Emitting PacketID
    # only on a valid decrypt suppresses those false positives, which otherwise
    # appear as spurious packets (esp. from the carrier-rescue re-decodes).
    _pktid_line = "  PacketID: 0x%08x  Flags: 0x%02x (%s)" % (pkt_id, flags, flag_parts)
    _chan_line = "  Channel hash: 0x%02x" % chan_hash

    if len(payload) <= 16:
        return

    enc_data = payload[16:]

    # Build AES-CTR nonce: packetID(4) + 0x00000000(4) + sender(4) + 0x00000000(4)
    nonce = bytearray(16)
    nonce[0:4] = p[8:12]   # packet ID
    nonce[8:12] = p[4:8]   # sender

    # Strict decrypt-validity gate.  A chase CRC-16 coincidence can pass the
    # broadcast-dst gate yet carry a garbled PacketID; decrypting with the wrong
    # nonce/key yields random bytes that sometimes parse as a Data protobuf with
    # a plausible portnum (e.g. TEXT=1) but garbage payload.  Require:
    #   (a) portnum is a real Meshtastic PortNum, and
    #   (b) for TEXT_MESSAGE_APP, the payload is valid UTF-8.
    # Genuine packets always satisfy both; coincidences/wrong-key do not.
    # Build the ordered key candidates: multi-key config (Config tab) if present,
    # else the legacy single-key / no_key behaviour (unchanged for the validated
    # default-key path — aes_ctr_decrypt == aes128_ctr_decrypt for 16-byte keys).
    _kcfg = _load_keys()
    if no_key:
        _cands = [(None, 'plain')]
    elif _kcfg is not None:
        _cands = _ordered_keys(_kcfg, sender)
    else:
        _cands = [(aes_key, 'default')]

    portnum, inner_payload = -1, None
    want_response = request_id = reply_id = emoji = None
    _valid, _key_label = False, None
    for _kb, _klabel in _cands:
        if _kb is None:
            decrypted = bytes(enc_data)
        else:
            decrypted = aes_ctr_decrypt(_kb, bytes(nonce), bytes(enc_data))
        fields = parse_protobuf(decrypted)
        portnum, inner_payload = -1, None
        want_response = request_id = reply_id = emoji = None
        for fn, wt, val in fields:
            if fn == 1 and wt == 0: portnum = int(val)
            if fn == 2 and wt == 2: inner_payload = val
            if fn == 3 and wt == 0: want_response = bool(val)
            if fn == 6 and wt == 0: request_id = int(val)
            if fn == 7 and wt == 0: reply_id = int(val)
            if fn == 8 and wt == 0: emoji = int(val)
        _v = (portnum >= 0) and (portnum in KNOWN_PORTNUMS)
        if _v and portnum in _TEXT_PORTNUMS and inner_payload:
            try:
                inner_payload.decode('utf-8')
            except UnicodeDecodeError:
                _v = False
        if _v:
            _valid, _key_label = True, _klabel
            break
    if not _valid:
        print("  %s -> %s [decrypt failed — not a valid packet, suppressed]" % (sender, dest_str))
        # Clean (non-chase) CRC means the whole-packet CRC-16 matched WITHOUT
        # bit-flipping → a REAL on-air packet whose payload just won't decrypt
        # with our channel key.  Emit it header-only (encrypted) so the UI shows
        # who is talking to whom even when the contents are hidden.  Chase-only
        # CRC matches (coincidences) are NOT emitted → the 0-FP property holds.
        if _clean_crc:
            # Keep the ciphertext + nonce inputs so the web can RETRY decryption
            # later if the user adds the right key (right-click → Decode).
            _rec['enc_hex'] = bytes(enc_data).hex()
            _emit_pkt(_rec)
        return

    # Valid decrypt → real packet.  Now emit the PacketID marker + channel.
    print(_pktid_line)
    print(_chan_line)
    _DECODED_SET.add((pkt_id, hops_taken))   # for carrier-rescue early-exit
    _rec['decrypted'] = True
    _rec['confidence'] = 'verified'      # AES decrypt + valid protobuf = crypto proof
    # Post-decode RAW: the protobuf-encoded Data message that came out of AES-CTR.
    # Independent of which portnum was parsed — lets the user verify the decrypted
    # bytes even when our protobuf parser missed a field.
    _rec['decrypted_hex'] = decrypted.hex() if isinstance(decrypted, (bytes, bytearray)) else bytes(decrypted).hex()
    _rec['portnum'] = portnum
    _rec['port_name'] = portnum_name(portnum)
    if _key_label and _key_label != 'default':
        _rec['key'] = _key_label   # which custom channel key decoded it
    print("  Portnum: %d (%s)" % (portnum, portnum_name(portnum)))
    if want_response: print("  want_response: yes")
    if request_id:    print("  request_id: 0x%08x" % request_id)
    if reply_id:      print("  reply_id: 0x%08x  emoji: %s" % (
                          reply_id, chr(emoji) if emoji and emoji > 0x1f else ("0x%x" % emoji if emoji else "")))

    if inner_payload is None or len(inner_payload) == 0:
        # Clean decrypted packet — register the endpoints as known nodes for
        # behavioral chase verification on subsequent encrypted-DM recoveries.
        _register_known_node(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24))
        _register_known_node(p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24))
        _emit_pkt(_rec)
        return

    _enr = _portnum_record(portnum, inner_payload)
    _rec.update(_enr)
    if _enr.get('text') is not None:
        print("  Message: \"%s\"" % _enr['text'])
    elif _enr.get('payload_hex'):
        print("  Payload (%d bytes): %s" % (len(inner_payload), _enr['payload_hex']))
    # Clean decrypted packet — register the endpoints as known nodes.
    _register_known_node(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24))
    _register_known_node(p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24))
    _emit_pkt(_rec)


# ============================================================================
# Protocol parser registry — route a clean-CRC LoRa frame to a parser by sync
# word.  Each entry is (name, matcher(sync,payload,ctx) -> bool, handler).  A
# handler returns a record dict to emit (LoRaWAN/MeshCore), or None if it emits
# its own [PKT] (Meshtastic).  Add new LoRa protocols here — the PHY front-end
# already hands every frame's raw bytes to this dispatch.
# ============================================================================
def _emit_proto(rec, raw, sf, bw, cr, sync, rf):
    """Attach PHY context + raw bytes to a non-Meshtastic protocol record, emit it."""
    if rec is None:
        return
    rec['raw_hex'] = bytes(raw).hex()
    rec['sf'] = sf; rec['bw'] = bw; rec['cr'] = cr
    rec['sync'] = ('0x%02x' % sync) if sync is not None else None
    if rf:
        rec.setdefault('freq_mhz', rf.get('freq_mhz'))
        rec.setdefault('rssi', rf.get('rssi'))
        # Hardware fingerprint — for the web's device clustering.
        if rf.get('hw_fp') is not None:
            rec['hw_fp'] = rf['hw_fp']
        if rf.get('cfo_hz') is not None:
            rec['cfo_hz'] = rf['cfo_hz']
        if rf.get('cfo_drift') is not None:
            rec['cfo_drift'] = rf['cfo_drift']
    _emit_pkt(rec)


def _h_lorawan(payload, ctx):
    return parse_lorawan_packet(payload, ctx.get('rf'))


def _h_meshcore(payload, ctx):
    return parse_meshcore_packet(payload)


def _h_loramesher(payload, ctx):
    return parse_loramesher_packet(payload, ctx.get('rf'))


def _h_meshtastic(payload, ctx):
    print("  Meshtastic header: dst=%s src=%s" % (
        ''.join('%02x' % payload[3 - i] for i in range(4)),
        ''.join('%02x' % payload[7 - i] for i in range(4))))
    parse_meshtastic_packet(payload, aes_key=ctx['aes_key'], no_key=ctx['no_key'],
                            _clean_crc=ctx['clean_crc'], _rf=ctx['rf'])
    return None   # parse_meshtastic_packet emits its own [PKT]


def _h_lora_aprs(payload, ctx):
    return parse_lora_aprs_packet(payload, ctx.get('rf'))


def _h_reticulum(payload, ctx):
    return parse_reticulum_packet(payload, ctx.get('rf'))


def _h_disaster_radio(payload, ctx):
    return parse_disaster_radio_packet(payload, ctx.get('rf'))


def _h_radiohead(payload, ctx):
    return parse_radiohead_packet(payload, ctx.get('rf'))


def _h_ebyte_lora(payload, ctx):
    return parse_lora_p2p_packet(payload, ctx.get('rf'))


def _proto_candidates(sync):
    """Ordered (name, handler) parsers to try for a non-Meshtastic frame.

    This is a GENERAL, multipurpose identifier: the LoRa sync word cannot be
    trusted to name the protocol — MeshCore's sync is user-configurable to ANY
    value (0x12 default, 0x34 public, or custom), LoRaWAN uses 0x34, LoRaMesher
    is configurable, and demod can misread the sync entirely.  So we try EVERY
    implemented non-Meshtastic protocol and let the CONTENT validators decide.

    The try-all-rank dispatch sorts by confidence tier so strong validators
    (LoRa APRS's 3-byte magic, LoRaWAN's MHDR, disaster.radio's totalLength)
    win over loose ones (RadioHead's 4-byte structureless header).  The order
    here is only a deterministic tiebreaker.
    """
    # APRS / disaster.radio / Reticulum / RadioHead / LoRa P2P are sync-word-
    # agnostic (their air format is library-specific and sync word varies).
    # Order: strongest-validator first so the highest-confidence claim wins.
    common_tail = [
        ('lora_aprs', _h_lora_aprs),
        ('reticulum', _h_reticulum),
        ('disaster_radio', _h_disaster_radio),
        ('ebyte_lora', _h_ebyte_lora),
        ('radiohead', _h_radiohead),
    ]
    if sync == 0x34:
        return [('lorawan', _h_lorawan), ('loramesher', _h_loramesher),
                ('meshcore', _h_meshcore)] + common_tail
    return [('meshcore', _h_meshcore), ('loramesher', _h_loramesher),
            ('lorawan', _h_lorawan)] + common_tail


# ============================================================================
# Demodulation helpers
# ============================================================================
def demod_fine(seg, ref_chirp, N, zoom=16):
    fft_out = np.abs(_fft(seg * ref_chirp, n=N * zoom))
    mi = int(np.argmax(fft_out))
    return mi / zoom


def reconstruct_lora_packet(iq_len, preamble_start, data_bins, sf, bw, fs,
                            amplitude=1.0+0j, sync_word=0x2B, n_preamble=8):
    """Synthesise a LoRa packet's analytic signal at 1x rate.

    This is the inverse of the decoder's demod chain — given the cyclic-shift
    bins of every symbol after preamble, regenerate the exact transmitted
    chirp sequence.  Used for successive interference cancellation: subtract
    the reconstruction of a successfully-decoded packet from the captured IQ,
    leaving any concurrent second packet visible to a second decode pass.

    Modulation reference (matches the decoder's demod):
      - upchirp(n) = exp(j π·bw/T_sym · t²)  for t in [0, T_sym)
      - downchirp = conj(upchirp)
      - A LoRa symbol with cyclic-shift bin k is
          sym_k(n) = upchirp(n) · exp(2π j k n / N)
        which dechirps (sym_k · downchirp) to a tone at bin k.

    Packet layout (matches decoder's data_start calculation):
      0 .. 8N             :  8 upchirp preamble symbols
      8N .. 10N           :  2 sync-word symbols (bins from sync_word bytes)
      10N .. 12N          :  2 downchirps
      12N .. 12.25N       :  1/4 downchirp
      12.25N .. end       :  header + payload symbols (each at its decoded bin)
    """
    N = 1 << sf
    t = np.arange(N, dtype=np.float64) / bw
    Ts = N / bw
    upchirp = np.exp(1j * np.pi * bw / Ts * t**2).astype(np.complex64)
    downchirp = np.conj(upchirp).astype(np.complex64)

    n = np.arange(N, dtype=np.float64)
    output = np.zeros(iq_len, dtype=np.complex64)

    def _place(p_start, sig):
        end = p_start + len(sig)
        if p_start >= iq_len:
            return
        if end > iq_len:
            sig = sig[:iq_len - p_start]
            end = iq_len
        if p_start < 0:
            sig = sig[-p_start:]
            p_start = 0
        output[p_start:end] = sig

    # 1. Preamble — n_preamble upchirps at the same carrier
    for i in range(n_preamble):
        _place(preamble_start + i * N, upchirp)

    # 2. Sync words — two chirps with cyclic shifts derived from the sync byte.
    # Meshtastic uses sync 0x2B → upper nibble 2, lower nibble B (11), each
    # multiplied by 8 (the standard public/private sync encoding) → bins 16, 88.
    sa = ((sync_word >> 4) & 0xF) * 8
    sb = (sync_word & 0xF) * 8
    sym_sa = (upchirp * np.exp(2j * np.pi * sa * n / N)).astype(np.complex64)
    sym_sb = (upchirp * np.exp(2j * np.pi * sb * n / N)).astype(np.complex64)
    _place(preamble_start + 8 * N, sym_sa)
    _place(preamble_start + 9 * N, sym_sb)

    # 3. Two downchirps + 0.25 downchirp (SFD)
    _place(preamble_start + 10 * N, downchirp)
    _place(preamble_start + 11 * N, downchirp)
    _place(preamble_start + 12 * N, downchirp[:N // 4])

    # 4. Data symbols start at preamble_start + 12.25 N (rounded)
    data_start = preamble_start + int(round(12.25 * N))
    for i, k in enumerate(data_bins):
        k = int(k) % N
        sym_k = (upchirp * np.exp(2j * np.pi * k * n / N)).astype(np.complex64)
        _place(data_start + i * N, sym_k)

    return output * amplitude


# ============================================================================
# Hard deinterleave (for header — kept simple)
# ============================================================================
def deinterleave_hard(symbols, n_sym, ppm):
    if len(symbols) < n_sym:
        return []
    codewords = []
    for k in range(ppm):
        cw = 0
        for i in range(n_sym):
            idx = ((i - k - 1) % ppm + ppm) % ppm
            bp = ppm - 1 - idx
            bit = (symbols[i] >> bp) & 1
            cw |= (bit << i)
        codewords.append(cw)
    return codewords


# ============================================================================
# Main processing
# ============================================================================
def _sic_subtract(iq_work, iq_src, pkt_start, mask_len, sf, bw, N, preamble_bin):
    """In-place SIC subtraction.

    iq_work: complex64 buffer that will have the decoded packet's analytic
             signal subtracted (modified in-place via slice assignment).
    iq_src:  buffer to demodulate from — usually the un-modified original
             so we don't demod from a previous SIC residual.
    pkt_start, mask_len: extent of the decoded packet in 1x-rate samples.
    preamble_bin: cyclic-shift bin of the preamble (= CFO in bins).

    Per-symbol complex amplitude estimation:
      For each LoRa symbol we know the cyclic-shift k (from demod) and the
      template sym_k(n) = upchirp(n) · exp(2π j k n / N).  The signal in
      that symbol's window is approximately α_k · sym_k_at_carrier where
      α_k is a complex per-symbol amplitude (channel + phase).
        α_k = <signal, sym_k_at_carrier> / N
      Subtracting α_k · sym_k_at_carrier removes that symbol's contribution
      with the actual channel state, instead of relying on a single
      preamble-derived amplitude that misses fast fading and timing drift.
    """
    pkt_end = min(len(iq_work), pkt_start + mask_len)
    if pkt_start + N > len(iq_src):
        raise ValueError("preamble past end")

    _t = np.arange(N, dtype=np.float64) / bw
    _Ts = N / bw
    _up = np.exp(1j * np.pi * bw / _Ts * _t**2).astype(np.complex64)
    _dn = np.conj(_up).astype(np.complex64)
    _n = np.arange(N, dtype=np.float64)
    _shift_dn = np.exp(-2j * np.pi * preamble_bin * _n / N).astype(np.complex64)
    _carrier_per_n = np.exp(2j * np.pi * preamble_bin * _n / N).astype(np.complex64)

    sync_word = 0x2B
    sa = ((sync_word >> 4) & 0xF) * 8
    sb = (sync_word & 0xF) * 8
    sym_sa = (_up * np.exp(2j * np.pi * sa * _n / N)).astype(np.complex64)
    sym_sb = (_up * np.exp(2j * np.pi * sb * _n / N)).astype(np.complex64)

    def _subtract_sym(p, template):
        if p + N > len(iq_work):
            return
        # template is at carrier 0; shift it to preamble_bin carrier
        template_carrier = (template * _carrier_per_n).astype(np.complex64)
        # Project the observed segment onto the carrier-shifted template
        # to estimate complex amplitude for THIS symbol.
        seg = iq_src[p:p + N]
        alpha = complex(np.vdot(template_carrier, seg) / N)
        iq_work[p:p + N] = (seg - alpha * template_carrier).astype(np.complex64) \
            if iq_src is iq_work \
            else (iq_work[p:p + N] - alpha * template_carrier).astype(np.complex64)

    # 8 preamble symbols (upchirp, bin 0)
    for i in range(8):
        _subtract_sym(pkt_start + i * N, _up)

    # 2 sync symbols
    _subtract_sym(pkt_start + 8 * N, sym_sa)
    _subtract_sym(pkt_start + 9 * N, sym_sb)

    # 2 downchirps + 0.25 downchirp (SFD)
    _subtract_sym(pkt_start + 10 * N, _dn)
    _subtract_sym(pkt_start + 11 * N, _dn)
    # quarter downchirp — use shorter template
    pq = pkt_start + 12 * N
    if pq + N // 4 <= len(iq_work):
        tmpl_q = (_dn[:N // 4] * _carrier_per_n[:N // 4]).astype(np.complex64)
        seg_q = iq_src[pq:pq + N // 4]
        alpha_q = complex(np.vdot(tmpl_q, seg_q) / (N // 4))
        iq_work[pq:pq + N // 4] = (seg_q - alpha_q * tmpl_q).astype(np.complex64) \
            if iq_src is iq_work \
            else (iq_work[pq:pq + N // 4] - alpha_q * tmpl_q).astype(np.complex64)

    # Data symbols: demod from iq_src to get bin k, then subtract
    data_start = pkt_start + int(round(12.25 * N))
    n_data = (pkt_end - data_start) // N
    for i in range(n_data):
        p = data_start + i * N
        if p + N > len(iq_src):
            break
        seg_demod = iq_src[p:p + N] * _shift_dn  # remove carrier
        k_fine = demod_fine(seg_demod, _dn, N)
        k = int(round(k_fine)) % N
        sym_k = (_up * np.exp(2j * np.pi * k * _n / N)).astype(np.complex64)
        _subtract_sym(p, sym_k)


def _sic_subtract_wideband(iq_wb, fs, bw, sf, carrier_hz, pkt_start, n_data_syms):
    """Successive-interference-cancellation in the WIDEBAND domain.

    Reconstruct a just-decoded LoRa packet at its OWN carrier and subtract it
    from the wideband IQ in place, leaving any concurrent packet (different
    carrier and/or time) intact for a second decode pass.  Unlike the 1x-rate
    `_sic_subtract`, this works on the full-rate buffer so the residual can be
    re-centred to any offset (±BW/2 adjacent channel) and decoded — which is
    what a Meshtastic hop + its relay need (they share a gate window at
    different carriers).

    Per-symbol complex-amplitude projection (channel + phase), demodulating
    each data symbol from the CURRENT iq so iterative multi-packet SIC works.

      iq_wb       complex64 wideband buffer, modified in place
      fs, bw, sf  wideband rate, LoRa bandwidth, spreading factor
      carrier_hz  the packet's carrier offset from the buffer's DC
      pkt_start   preamble start, in WIDEBAND samples
      n_data_syms number of data symbols to subtract
    """
    N = 1 << sf
    dec = int(round(fs / bw))
    sps = N * dec                                  # wideband samples per symbol
    Ts = N / bw
    _t = np.arange(sps, dtype=np.float64) / fs
    up = np.exp(1j * np.pi * bw / Ts * _t ** 2).astype(np.complex64)
    dn = np.conj(up).astype(np.complex64)
    nidx = np.arange(sps, dtype=np.float64)

    # --- Timing refinement: chirps are timing-sensitive, so a few-sample
    # error in pkt_start decorrelates the per-symbol projection and the
    # subtraction barely removes the packet.  Align by maximising the
    # preamble correlation (8 upchirps at the carrier) over a small offset
    # search around the supplied pkt_start.
    _pre_n = 8 * sps
    if 0 <= pkt_start and pkt_start + _pre_n + 2 * sps < len(iq_wb):
        _carr_pre = np.exp(2j * np.pi * carrier_hz *
                           (np.arange(_pre_n, dtype=np.float64) / fs)).astype(np.complex64)
        _pre_tmpl = np.tile(up, 8).astype(np.complex64) * _carr_pre
        _best, _best_c = pkt_start, -1.0
        for _off in range(-sps, sps + 1, max(1, dec // 4)):
            _ps = pkt_start + _off
            if _ps < 0 or _ps + _pre_n > len(iq_wb):
                continue
            _c = abs(complex(np.vdot(_pre_tmpl, iq_wb[_ps:_ps + _pre_n])))
            if _c > _best_c:
                _best_c, _best = _c, _ps
        pkt_start = _best

    def _sub(p, base_template):
        if p < 0 or p + sps > len(iq_wb):
            return
        # Absolute-time carrier so phase stays continuous across symbols.
        carr = np.exp(2j * np.pi * carrier_hz *
                      (np.arange(p, p + sps, dtype=np.float64) / fs)).astype(np.complex64)
        templ = (base_template * carr).astype(np.complex64)
        seg = iq_wb[p:p + sps]
        denom = float(np.vdot(templ, templ).real) + 1e-12
        alpha = complex(np.vdot(templ, seg) / denom)
        iq_wb[p:p + sps] = (seg - alpha * templ).astype(np.complex64)

    # preamble (8 upchirps)
    for i in range(8):
        _sub(pkt_start + i * sps, up)
    # sync words (2) — cyclic-shift bins from the 0x2B sync byte
    _sw = 0x2B
    _sa = ((_sw >> 4) & 0xF) * 8
    _sb = (_sw & 0xF) * 8
    _sub(pkt_start + 8 * sps, (up * np.exp(2j * np.pi * _sa * nidx / sps)).astype(np.complex64))
    _sub(pkt_start + 9 * sps, (up * np.exp(2j * np.pi * _sb * nidx / sps)).astype(np.complex64))
    # 2 full downchirps (SFD)
    _sub(pkt_start + 10 * sps, dn)
    _sub(pkt_start + 11 * sps, dn)
    # data symbols start at +12.25 symbols; demod each from current iq then subtract
    data_start = pkt_start + int(round(12.25 * sps))
    for i in range(max(0, n_data_syms)):
        p = data_start + i * sps
        if p + sps > len(iq_wb):
            break
        carr = np.exp(-2j * np.pi * carrier_hz *
                      (np.arange(p, p + sps, dtype=np.float64) / fs)).astype(np.complex64)
        seg_dc = (iq_wb[p:p + sps] * carr * dn).astype(np.complex64)   # baseband + dechirp
        decimated = seg_dc.reshape(N, dec).mean(axis=1)
        k = int(np.argmax(np.abs(_fft(decimated))))
        _sub(p, (up * np.exp(2j * np.pi * k * nidx / sps)).astype(np.complex64))


def process_file(fpath, relay_after=None, relay_before=None,
                 _iq_override=None, _allow_rescue=True, _budget_override=None):
    """Decode a captured LoRa IQ file.

    relay_after: if set (int), blank out the first relay_after BW-rate symbols
    before any decode attempt.  Used by the relay search pass to find the relay
    hop that follows the primary packet in the same recording.

    _iq_override / _allow_rescue / _budget_override: internal — the carrier
    rescue re-invokes process_file on a frequency-shifted copy of the IQ to get
    a FULL re-decode (preamble re-lock + all passes) at a shifted carrier, which
    is what actually recovers a confirmed packet whose data wouldn't align at
    the original carrier.  _allow_rescue=False stops infinite recursion.
    """
    from collections import Counter
    # CPU-time budget, not wall-clock.  The live pipeline runs many decode
    # worker processes alongside the detect pool, so a WALL budget gets robbed
    # by CPU contention — under load a decode's recovery passes (PASS 2,
    # carrier rescue) bail early and a hop goes missing INCONSISTENTLY (the
    # "exactly one of 120" symptom: same captures, different misses per run).
    # process_time() measures THIS process's actual CPU work, immune to how
    # many other processes are running, so every decode gets its full compute
    # budget regardless of load.  REQUIRES single-threaded decode workers
    # (OMP/MKL/OPENBLAS/NUMEXPR=1 in the spawn env) — otherwise multi-threaded
    # FFTs inflate process_time and the budget exhausts too fast.
    _process_start = time.process_time()
    # Hard wall-clock budget per file.  Clean LoRa decodes in <300 ms via
    # primary CRC, but a marginal-yet-real capture can need PASS 2 (re-centre
    # + second sweep) to recover — run_4's Test4/Test18 both hops decode only
    # with PASS 2, which the old 1.5 s default bailed before reaching.  The
    # budget only applies AFTER a preamble is found, so noise/spurs still
    # fail-fast (no preamble → instant return).  The live pipeline overrides
    # this to 5.0 s (LORA_DECODE_BUDGET_S) since the decoder process is
    # mostly idle — the detector is the real-time bottleneck, not decode.
    import os as _os
    _BUDGET_S = (_budget_override if _budget_override is not None
                 else float(_os.environ.get('LORA_DECODE_BUDGET_S', '1.5')))

    iq = (_iq_override if _iq_override is not None
          else np.fromfile(fpath, dtype=np.complex64))
    # DC-spike / LO-leak removal.  A packet whose carrier sits at the SDR centre
    # keeps the hardware DC tone (LO leakage + ADC offset) in-band, sitting at DC
    # right under the chirp — after dechirp it spreads into a raised noise floor
    # and can sink a marginal decode.  The leak is STATIONARY = the capture's
    # complex mean; a LoRa chirp's mean ≈ 0.  So subtract the mean, but ONLY when
    # the DC component is anomalously strong (|mean|^2 > 1e-3 × mean power) — a
    # real leak.  Measured: normal captures sit at ~1e-5 (1000× below the gate),
    # so this is a NO-OP / byte-identical for every off-/centred capture and only
    # fires for the at-centre-with-leak case.  Skipped on the rescue re-entry
    # (_iq_override already derives from a cleaned buffer).
    if _iq_override is None and len(iq) > 0:
        _dc = iq.mean()
        if abs(_dc) ** 2 > 1e-3 * float(np.mean(np.abs(iq) ** 2)):
            iq = (iq - _dc).astype(np.complex64)
    if _allow_rescue:
        _DECODED_SET.clear()   # top-level call: start fresh decode tracking
    name = os.path.basename(fpath)
    parts = name.lstrip('.').split('_')
    sf = int(parts[0].replace('SF', ''))
    bw = int(parts[1].replace('k', '')) * 1000
    N = 1 << sf
    ppm = sf - 2  # reduced rate bits for header

    # Decode work scales with N=2^sf (bigger FFTs, longer symbols), so a budget
    # tuned for SF7 is too tight at higher SF — SF9 needed ~2x, SF12 ~5.5x — and
    # the recovery passes (PASS 2 re-centre, carrier rescue) bail before they
    # find a concurrent hop.  Scale the wall budget with sf so every spreading
    # factor gets the time its passes need (SF7 factor=1.0 → unchanged).  This
    # also scales the rescue sub-call's budget (it passes _budget_override).
    if _BUDGET_S > 0:
        _BUDGET_S *= 2.0 ** ((sf - 7) * 0.5)

    # Sample rate: PREFER an explicit token in the filename (e.g. "2000ksps"),
    # which the gate now writes.  The old size-based guess below is AMBIGUOUS —
    # a 2 MHz file's duration is also valid interpreted as 4 MHz, so it mis-read
    # lower-rate captures as higher, which is exactly what blocked lowering the
    # export oversampling.  An explicit token removes the ambiguity.  Fall back
    # to size-inference for legacy captures without the token.
    fs = None
    for _p in parts:
        if _p.endswith('ksps'):
            try:
                fs = int(_p[:-4]) * 1000
            except ValueError:
                fs = None
            break
    if fs is None:
        for mult in [8, 4, 2, 1]:
            fs_try = bw * mult
            dur_try = len(iq) / fs_try
            if 0.3 <= dur_try <= 20.0 and fs_try == int(fs_try):
                fs = int(fs_try)
                break
    if fs is None:
        fs = int(round(len(iq) / 2.0))
    dec = fs // bw

    print("=== %s ===" % name)
    print("SF%d BW=%dk N=%d ppm=%d fs=%dk" % (sf, bw // 1000, N, ppm, fs // 1000))

    # Keep full FFT for re-centering in pass 2
    F = np.fft.fftshift(_fft(iq))
    keep = len(F) // dec
    start_f = len(F) // 2 - keep // 2
    # Bins-per-Hz of the cached spectrum (used by recrop_centered to translate
    # a Hz shift into an integer bin roll, replacing one full forward FFT).
    _F_bin_per_hz = len(F) / fs

    def recrop_centered(cfo_hz_value):
        """Re-extract at 1x rate with the signal shifted to near DC.

        PERF: replaces `fft(iq * exp(-j2π·cfo·t))` with a frequency-domain
        circular shift of the already-cached `F`.  Mathematically equivalent
        (the DFT shift theorem) — integer-bin precision (~fs/(2·N) Hz error,
        well below the decoder's downstream tolerance).  Removes one large
        forward FFT + one full complex-multiply per call.
        """
        bin_shift = int(round(cfo_hz_value * _F_bin_per_hz))
        Fs = np.roll(F, -bin_shift) if bin_shift else F
        out = _ifft(np.fft.ifftshift(Fs[start_f:start_f + keep])) * (keep / len(Fs))
        return out.astype(np.complex64)

    iq1_orig = _ifft(np.fft.ifftshift(F[start_f:start_f + keep])) * (keep / len(F))
    iq1_orig = iq1_orig.astype(np.complex64)

    # Time-window isolation.  Both relay_after and relay_before can be set
    # simultaneously to isolate a *specific* preamble — e.g. when the caller
    # knows from its own SC locator that the target preamble starts at sample
    # T, it can pass relay_after=T-pre_pad and relay_before=T+packet_len.
    # Whatever falls outside that window is zeroed, so the decoder's
    # SC scan locks on exactly that preamble instead of a stronger
    # concurrent signal elsewhere in the buffer.
    # relay_after / relay_before are in BW-rate symbols; 1 symbol = N
    # samples at the 1x rate.
    if relay_after is not None:
        _cutoff = min(relay_after * N, len(iq1_orig))
        iq1_orig[:_cutoff] = 0.0
    if relay_before is not None:
        _cutoff = min(relay_before * N, len(iq1_orig))
        iq1_orig[_cutoff:] = 0.0

    # ---- Per-capture precise TX-carrier measurement (envelope+Welch) ----
    # The decoder's k_hat_fine measures the preamble-bin position in the
    # decimated iq1 — but this aliases at ±bw/2 (bin near N/2 ambiguous in
    # sign) which can split same-device captures into different clusters.
    # We compute a NOISE-RESISTANT precise carrier estimate directly on the
    # ORIGINAL FULL-RATE iq:
    #   1. Compute amplitude envelope, locate the burst (high-amplitude
    #      time window) via threshold above noise floor.
    #   2. Welch PSD over the burst window.
    #   3. Power-weighted spectral centroid over bins above noise floor —
    #      this is the precise burst-carrier freq (relative to baseband).
    # The full-rate iq has no aliasing concerns (it covers ±fs/2 which is
    # always wider than the chirp's ±bw/2 footprint).
    #
    # IMPORTANT: only compute on the OUTER process_file call (when
    # _iq_override is None).  Recursive rescue calls pass a frequency-shifted
    # _iq_override IQ, and re-running the fingerprint on the shifted IQ gives
    # a WRONG carrier (the shift is built into the measurement).  Outer call
    # measures the true unshifted carrier; recursive calls inherit it via
    # _CURRENT_HW_FP[0], which we DON'T clobber when entering recursively.
    if _iq_override is None:
        _CURRENT_HW_FP[0] = None
        _CURRENT_PRECISE_CFO[0] = None
    try:
        if _iq_override is not None:
            raise StopIteration  # skip — outer call already measured
        # LORA_FINGERPRINT=0 → skip the envelope+Welch carrier-centroid block.
        # (precise_carrier_hz is a fingerprint feature; the decoder uses the
        # gate's per-window CFO estimate for the actual demod.)
        if not _FINGERPRINT_ON:
            raise StopIteration
        if len(iq) >= 4096:
            # Smoothing window = half a LoRa symbol — long enough to average
            # over the chirp's bandwidth oscillation, short enough to localize
            # the burst within the capture.  N/bw is the symbol duration in
            # seconds at the LoRa symbol rate; * fs converts to capture samples.
            _sym_samples_at_fs = int((N / bw) * fs)
            _sm = max(256, _sym_samples_at_fs // 2)
            _env = np.abs(iq).astype(np.float64)
            # CLIPPING DETECTION: when LNA saturates (strong nearby TX),
            # tens of thousands of samples pile up at the ADC max.  Signature:
            # 99%ile envelope is right at max value.  Flag this so downstream
            # fingerprinting can downgrade confidence — amplitude features
            # become unreliable under clipping.
            _max_env = float(np.max(_env))
            _p99_env = float(np.percentile(_env, 99))
            # Clipped if 99%ile is within 2% of max AND peak is high
            _is_clipped = (_max_env > 0.5 and (_max_env - _p99_env) / _max_env < 0.02)
            _clip_fraction = float(np.sum(_env > _p99_env * 0.99)) / len(_env) if _is_clipped else 0.0
            if _CURRENT_HW_FP[0] is None:
                _CURRENT_HW_FP[0] = {}
            _CURRENT_HW_FP[0]['clipped'] = bool(_is_clipped)
            _CURRENT_HW_FP[0]['clip_fraction'] = float(_clip_fraction)
            _cs = np.cumsum(_env)
            _smoothed = (_cs[_sm:] - _cs[:-_sm]) / _sm
            # Threshold = noise floor + 50% of dynamic range.  Noise floor is
            # estimated as the 20th percentile of the smoothed envelope (robust
            # to bursts occupying any fraction of the capture).
            _nf_amp = float(np.percentile(_smoothed, 20))
            _peak_amp = float(np.max(_smoothed))
            _thresh = _nf_amp + 0.5 * (_peak_amp - _nf_amp)
            _above = _smoothed > _thresh
            if _above.any():
                _idx = np.where(_above)[0]
                # Use the LARGEST contiguous run — handles spurious noise
                # spikes that might cross threshold briefly outside the burst.
                _diffs = np.diff(_idx)
                _gaps = np.where(_diffs > _sm)[0]
                _runs = []
                _run_start = 0
                for _g in _gaps:
                    _runs.append((_idx[_run_start], _idx[_g]))
                    _run_start = _g + 1
                _runs.append((_idx[_run_start], _idx[-1]))
                # Pick the LONGEST run (largest contiguous burst).
                _best_run = max(_runs, key=lambda r: r[1] - r[0])
                _b_start = int(_best_run[0])
                _b_end = int(_best_run[1]) + _sm
                _burst_iq = iq[_b_start:_b_end]
                if len(_burst_iq) >= 1024:
                    from scipy.signal import welch as _welch_full
                    _nperseg = min(len(_burst_iq) // 4, 32768)
                    if _nperseg >= 256:
                        _fw, _Pw = _welch_full(_burst_iq, fs=fs, nperseg=_nperseg,
                                                return_onesided=False)
                        _ord = np.argsort(_fw)
                        _fw = _fw[_ord]; _Pw = _Pw[_ord]
                        _PdB = 10 * np.log10(_Pw + 1e-30)
                        # LoRa PSD is bimodal: chirp band (high) and dead band
                        # (low / noise floor).  Median sits between, so use
                        # the 10th-percentile estimate of the NOISE floor and
                        # threshold at +6 dB above it.  This captures the full
                        # chirp band, not just its peak.
                        _nf_db = float(np.percentile(_PdB, 10))
                        _msk = _PdB > _nf_db + 6
                        if _msk.any():
                            _wts = 10 ** ((_PdB[_msk] - _nf_db) / 10.0)
                            _CURRENT_PRECISE_CFO[0] = float(
                                np.sum(_fw[_msk] * _wts) / np.sum(_wts))
                            # Preserve any existing keys (e.g. 'clipped' set earlier)
                            if _CURRENT_HW_FP[0] is None:
                                _CURRENT_HW_FP[0] = {}
                            _CURRENT_HW_FP[0]['precise_carrier_hz'] = _CURRENT_PRECISE_CFO[0]
                # --- UMOP burst-onset transient (PA ramp-up signature) ---
                # The 10%→90% envelope rise, overshoot above steady-state, and
                # mid-rise slope are PA-specific physical characteristics
                # orthogonal to TCXO frequency.  Validated SNR 1.0-1.4 between
                # same-batch RAK Meshtastic devices (overshoot strongest at
                # 1.43; abs_tx_hz collapses when TCXO drifts in long windows).
                # Combined with abs_tx_hz they reach ~95% classification.
                #
                # Uses a SHORT smoothing window (20μs) — the burst-detection
                # smoothing (~256 samples / half-symbol) is too coarse to
                # resolve the ~60μs PA rise without flattening it.  Requires
                # fs >= 250 kHz so the 20μs averaging spans ≥5 samples.
                # --- UMOP burst END-of-burst transient (PA turn-off) ---
                # Independent UMOP signal from PA shutdown.  Validated SNR up
                # to 1.63 (mid_fall_slope) — strongest single feature on the
                # hardest test dataset.  Combined with onset features reaches
                # 96%+ accuracy on borderline same-batch hardware.
                if fs >= 250_000 and _peak_amp > 1e-6 and _b_end < len(_smoothed):
                    try:
                        _eob_sm = max(4, int(fs * 20e-6))
                        # Take ±4ms around the burst END
                        _ew_lo = max(0, _b_end - int(fs * 0.004))
                        _ew_hi = min(len(_env), _b_end + int(fs * 0.002))
                        _env_eob = _env[_ew_lo:_ew_hi]
                        if len(_env_eob) >= _eob_sm * 4:
                            _cs3 = np.cumsum(_env_eob)
                            _eob = (_cs3[_eob_sm:] - _cs3[:-_eob_sm]) / _eob_sm
                            _eob_peak = float(np.max(_eob))
                            if _eob_peak > 1e-6:
                                _eob_above_50 = np.where(_eob > _eob_peak * 0.5)[0]
                                if len(_eob_above_50) >= 10:
                                    _be = int(_eob_above_50[-1])
                                    _ep90, _ep10 = _eob_peak * 0.9, _eob_peak * 0.10
                                    _ei90 = _be
                                    while _ei90 > 0 and _eob[_ei90] < _ep90: _ei90 -= 1
                                    _ei10 = _ei90
                                    while _ei10 < len(_eob) - 1 and _eob[_ei10] > _ep10: _ei10 += 1
                                    if 2 <= (_ei10 - _ei90) <= _eob_sm * 20:
                                        _fall_us = (_ei10 - _ei90) / fs * 1e6
                                        # 70%→30% mid-fall slope
                                        _ep70, _ep30 = _eob_peak * 0.7, _eob_peak * 0.3
                                        _ei70 = _ei90
                                        while _ei70 < len(_eob) - 1 and _eob[_ei70] > _ep70: _ei70 += 1
                                        _ei30 = _ei70
                                        while _ei30 < len(_eob) - 1 and _eob[_ei30] > _ep30: _ei30 += 1
                                        if _ei30 - _ei70 >= 1:
                                            _mid_fall_slope = (_eob[_ei30] - _eob[_ei70]) / (_ei30 - _ei70) / _eob_peak * 1e6 / fs
                                            # Pre-decay overshoot
                                            _pre_n = int(fs * 100e-6)
                                            _pre = _eob[max(0, _ei90 - _pre_n):_ei90]
                                            if len(_pre) >= 10:
                                                _pre_max = float(np.max(_pre))
                                                _pre_mean = float(np.mean(_pre))
                                                _pre_decay_over = (_pre_max - _pre_mean) / max(_pre_mean, 1e-9) * 100
                                                if _CURRENT_HW_FP[0] is None:
                                                    _CURRENT_HW_FP[0] = {}
                                                _CURRENT_HW_FP[0]['eob_fall_us'] = float(_fall_us)
                                                _CURRENT_HW_FP[0]['eob_mid_fall_slope'] = float(_mid_fall_slope)
                                                _CURRENT_HW_FP[0]['eob_pre_decay_over_pct'] = float(_pre_decay_over)
                    except Exception:
                        pass

                if fs >= 250_000 and _peak_amp > 1e-6:
                    try:
                        _on_sm = max(4, int(fs * 20e-6))     # 20μs short avg
                        # Compute a FRESH short-smoothed envelope.  Local to
                        # the rise region: take ±2 ms around the coarse burst
                        # start so we capture the rise + steady plateau.
                        _w_lo = max(0, _b_start - int(fs * 0.002))
                        _w_hi = min(len(_env), _b_start + int(fs * 0.004))
                        _env_local = _env[_w_lo:_w_hi]
                        if len(_env_local) >= _on_sm * 4:
                            _cs2 = np.cumsum(_env_local)
                            _on = (_cs2[_on_sm:] - _cs2[:-_on_sm]) / _on_sm
                            _on_peak = float(np.max(_on))
                            if _on_peak > 1e-6:
                                _on_above = np.where(_on > _on_peak * 0.5)[0]
                                if len(_on_above) >= 3:
                                    _bs2 = int(_on_above[0])
                                    if _on_sm * 2 <= _bs2 <= len(_on) - _on_sm * 2:
                                        _t10 = _on_peak * 0.10; _t90 = _on_peak * 0.90
                                        _i10 = _bs2
                                        while _i10 > 0 and _on[_i10] > _t10: _i10 -= 1
                                        _i90 = _bs2
                                        while _i90 < len(_on) - 1 and _on[_i90] < _t90: _i90 += 1
                                        _rise_n = _i90 - _i10
                                        if 2 <= _rise_n <= _on_sm * 20:
                                            _rise_us = _rise_n / fs * 1e6
                                            _steady_end = min(_i90 + 400, len(_on))
                                            if _steady_end - _i90 >= 50:
                                                _steady_mean = float(np.mean(_on[_i90:_steady_end]))
                                                _peak_in = float(np.max(_on[_i10:min(_i10 + _rise_n*3, len(_on))]))
                                                _overshoot_pct = (_peak_in - _steady_mean) / max(_steady_mean, 1e-9) * 100
                                                _t40 = _on_peak * 0.4; _t60 = _on_peak * 0.6
                                                _i40 = _i10
                                                while _i40 < len(_on) - 1 and _on[_i40] < _t40: _i40 += 1
                                                _i60 = _i40
                                                while _i60 < len(_on) - 1 and _on[_i60] < _t60: _i60 += 1
                                                _mid_slope = (_on[_i60] - _on[_i40]) / max(_i60 - _i40, 1) / _on_peak * 1e6 / fs
                                                if _CURRENT_HW_FP[0] is None:
                                                    _CURRENT_HW_FP[0] = {}
                                                _CURRENT_HW_FP[0]['onset_rise_us'] = float(_rise_us)
                                                _CURRENT_HW_FP[0]['onset_overshoot_pct'] = float(_overshoot_pct)
                                                _CURRENT_HW_FP[0]['onset_mid_slope'] = float(_mid_slope)
                    except Exception:
                        pass
    except StopIteration:
        pass
    except Exception:
        pass

    # ---- Pass 1: find preamble bin (coarse frequency) ----
    # Try decoding. If CRC OK, done. If not, use the found preamble bin
    # to re-center the extraction and try again (pass 2).
    MAX_ATTEMPTS = 5
    iq1_work = iq1_orig.copy()
    found_cfos_hz = []  # collect CFO Hz values from each attempt
    skip_bins = []  # bins of failed candidates — try different ones

    def _recenter_and_decode(cfo_hz, label="early"):
        """Re-extract at baseband for a large CFO and decode.  PASS 1 finds
        the preamble bin but corrects CFO by phase-rotation, which is
        unreliable when the signal sits near the ±bw/2 crop edge — recropping
        to DC (what PASS 2 does) is what actually recovers those captures.
        Doing it as soon as a large off-centre preamble is seen avoids
        burning the whole PASS-1 retry budget first (run_4 Test4 hop0:
        ~10 s → ~2 s).  Returns True iff a packet decoded."""
        for sh in ([cfo_hz, cfo_hz - bw] if cfo_hz > bw / 2 else
                   [cfo_hz, cfo_hz + bw] if cfo_hz < -bw / 2 else [cfo_hz]):
            if abs(sh) < 5 * bw / N:
                continue
            if _BUDGET_S > 0 and (time.process_time() - _process_start) > _BUDGET_S:
                return False
            print("\n  === %s recenter by %.0f Hz ===" % (label, sh))
            iqx = recrop_centered(sh)
            r = _decode_attempt(iqx.copy(), sf, bw, N, ppm, fs, dec, name,
                                Counter, skip_bins=skip_bins, sweep_budget=1.0)
            if r is not None and r[0] == 'OK':
                return True
        return False


    _need_rescue = False   # set when a header decoded but its data never CRC'd
    _residual_clear = False  # set when the masked residual has NO preamble left
    # Counter for consecutive attempts that produced NO valid LoRa header.  Real
    # packets of any protocol that pass the 4..237 byte sanity range reset this
    # counter; only sustained phantom-preamble cascades (gate fires on noise
    # spurs, decoder finds weak preamble-like patterns but no real LoRa header)
    # increment it.  Bailing the loop early on a cascade lets the worker pick
    # up the next capture instead of burning the full budget on noise.
    _consec_no_header = 0
    for attempt in range(MAX_ATTEMPTS):
        if _BUDGET_S > 0 and (time.process_time() - _process_start) > _BUDGET_S:
            print(f"\n  [BUDGET] decode budget {_BUDGET_S:.1f}s exhausted after attempt {attempt} — bail")
            return
        if _consec_no_header >= 3:
            print(f"\n  [BAIL] {_consec_no_header} consecutive no-header attempts — likely phantom-preamble cascade")
            break
        if attempt > 0:
            print("\n  --- Retry #%d ---" % attempt)
        _d = {}
        result = _decode_attempt(iq1_work.copy(), sf, bw, N, ppm, fs, dec, name, Counter,
                                 skip_bins=skip_bins, diag_out=_d)
        if result is None:
            _residual_clear = True   # no preamble left in the masked residual
            break  # no preamble found
        status, mask_start, mask_len, preamble_bin = result
        # Update the consecutive-no-header counter.  Uses the broad-PL
        # `header_decoded` flag so ALL protocols (Meshtastic, MeshCore,
        # LoRaWAN, Unknown, etc.) reset the counter — not just Meshtastic-PL
        # frames (which would be what the narrower `hdr_ok` flag covers).
        if _d.get('header_decoded'):
            _consec_no_header = 0
        elif status == 'FAIL':
            _consec_no_header += 1
        if status != 'OK' and _d.get('hdr_ok') and not _d.get('no_rescue'):
            _need_rescue = True   # confirmed packet whose data didn't decode here
        if status == 'WRONG_SF':
            return  # dominant signal is a different SF — give up immediately
        if status == 'OK':
            # Multi-packet recovery via SUCCESSIVE INTERFERENCE CANCELLATION.
            # A capture often holds two concurrent packets — a Meshtastic hop
            # and its relay — at DIFFERENT carriers (and times) in one gate
            # window.  Reconstruct the just-decoded packet at its own carrier
            # and subtract it from the WIDEBAND iq, leaving the concurrent hop
            # intact; then re-derive the centre crop and loop.  Pass 2 recrops
            # from this same (now SIC'd) iq, so the residual hop is decodable at
            # any offset — including a ±BW/2 adjacent channel that a plain
            # time-zero would have destroyed along with the decoded packet.
            if mask_len > 0:
                mask_end = min(len(iq1_work), mask_start + mask_len)
                iq1_work[mask_start:mask_end] = 0.0
                continue
            return  # no mask info — stop here
        if preamble_bin is not None:
            cfo_hz_raw = preamble_bin * bw / N
            found_cfos_hz.append(cfo_hz_raw)
            skip_bins.append(preamble_bin)
            # Recrop to baseband now instead of after exhausting the retry
            # budget.  Fires when the signal is well off-centre (SC peak can
            # land on an aliased bin on attempt 0 and the true carrier bin on
            # the next — run_4 Test4 hop0: bin 30 then bin 71 at 277 kHz) OR
            # when a header decoded here but its data failed CRC (hdr_ok): the
            # data demod misaligns at the ±bw/2 crop edge and recropping to DC
            # fixes it even for a moderate CFO just under bw/8.  This recovers
            # the time-separated hop1 of a concurrent pair (cfo ~58 kHz) via a
            # fast recenter instead of the slow 6-shift carrier rescue.
            if attempt <= 2 and (abs(cfo_hz_raw) > bw / 8 or _d.get('hdr_ok')) \
                    and _recenter_and_decode(cfo_hz_raw):
                # Decoded this packet via baseband recenter.  Don't stop here:
                # a capture often holds hop0 and its relay hop1 at DIFFERENT
                # times in the same window (e.g. 705 ms apart).  Mask this
                # packet's time region and continue so the loop's SC scan can
                # lock the time-separated concurrent hop.  The decrypt-validity
                # gate keeps a spurious second decode out of the output.
                if mask_len > 0:
                    mask_end = min(len(iq1_work), mask_start + mask_len)
                    iq1_work[mask_start:mask_end] = 0.0
                    continue
                return
        # Also mask for the last couple attempts as fallback
        if attempt >= 2:
            mask_end = min(len(iq1_work), mask_start + mask_len)
            iq1_work[mask_start:mask_end] = 0.0

    # ---- Carrier rescue ----
    # A header decoded but its data never CRC'd: the packet is REAL (its header
    # checksum passed) yet its full-rate data wouldn't align at the carrier we
    # used.  Brute-force confirmed a full re-decode at a SHIFTED carrier recovers
    # it — the preamble re-lock at the shifted centre corrects the data_start/CFO
    # the original alignment got slightly wrong.  Sweep a bounded range of
    # carrier offsets; at each, recrop and run a masking decode (decode the
    # strong hop, blank its time, decode the residual hop).  Runs HERE (before
    # the PASS-2 fallbacks, which would `return` on re-decoding the strong hop)
    # and is GATED on the header-OK/data-fail flag, so it only fires for a
    # confirmed-but-unrecovered packet (no false-positive risk), cost bounded.
    if _need_rescue and _allow_rescue:
        _trescue = np.arange(len(iq), dtype=np.float64) / fs
        # Shift by sizeable carrier offsets (not just a few bins): at small
        # offsets the SC scan re-locks the STRONGER hop; larger offsets move it
        # off SC's first pick so the weaker hop is locked and fully decoded
        # (brute-force recovered run_4 Test4 hop0 at ±50–300 kHz shifts).
        # Trimmed to 4 shifts × 1.5 s (from 6 × 2.0 s): the early-break already
        # stops on first recovery, TCXO-disciplined relays sit within a few bins so
        # the small ±0.1/±0.2 BW shifts decoy SC onto the weaker hop fine, and a
        # real recovery re-locks + decodes well under 1.5 s — so the ±0.3 BW shifts
        # and the extra 0.5 s/shift were dead weight burned only on FAILING rescues.
        _rescue_bud = 1.5
        _rescue_offs = [s * bw / 10.0 for s in (1, -1, 2, -2)]   # ±0.1, ±0.2 BW
        _rescue_before = set(_DECODED_SET)
        for _coff in _rescue_offs:
            if _BUDGET_S > 0 and (time.process_time() - _process_start) > _BUDGET_S:
                break
            print("\n  === CARRIER RESCUE: full re-decode shifted by %.0f Hz ===" % _coff)
            # Frequency-shift the WHOLE capture and re-run the entire decode on
            # it (preamble re-lock + all passes) — exactly what recovered hop0
            # in brute-force testing.  _allow_rescue=False prevents recursion;
            # a tight per-shift budget keeps the total bounded.
            _shifted = (iq * np.exp(-2j * np.pi * _coff * _trescue)).astype(np.complex64)
            try:
                process_file(fpath, relay_after=relay_after, relay_before=relay_before,
                             _iq_override=_shifted, _allow_rescue=False, _budget_override=_rescue_bud)
            except Exception:
                pass
            # Stop as soon as a NEW (pkt_id, hop) is recovered — no need to
            # sweep the rest, which keeps the rescue cheap (the common case
            # finds the missing hop in the first shift or two).
            if _DECODED_SET - _rescue_before:
                break

    # ---- Fast path: skip redundant fallbacks when everything is decoded ----
    # PASS 1 already decoded packet(s) AND the time-masked residual had no
    # preamble left (clean termination) AND no header-ok/data-fail is pending.
    # The spectrum-fb + wideband-fb + PASS 2 below would only re-decode the
    # same packets (a redundant ~0.4-0.6 s pass that the dedup then drops),
    # so return now.  Safe for Meshtastic: a relay hop shares the channel
    # (within ±bw/2), so a genuine second hop leaves a preamble in the
    # residual — that keeps the attempt loop going (result != None) and
    # clears _residual_clear, preventing this skip.  Only fires on captures
    # whose in-window signal is fully accounted for.
    if _allow_rescue and _DECODED_SET and _residual_clear and not _need_rescue:
        return

    # ---- Spectrum-based CFO fallback ----
    # The IF-mean wideband fallback below is biased near ±bw/2 CFO
    # boundaries: when a chirp symbol straddles the measurement window the
    # mean IF gets ±bw of bias.  Empirically on the offline-replay test set
    # (Test5 hop1) the IF method pointed at +272 kHz when the true carrier
    # was at -461 kHz — off by 1.5×bw, beyond the ±bw alias-resolve range.
    # Direct FFT spectrum-peak measurement on the active burst is immune to
    # this: a LoRa chirp's spectrogram has a stable centroid at the carrier
    # offset.  We pick the burst window (highest-power 4ms slice excluding
    # DC) and use the FFT spectrum-peak frequency as a fresh CFO candidate.
    # SF/BW-agnostic: scales with signal bandwidth and N.
    if len(iq) > N * dec * 4:
        _burst_len = N * dec * 4   # ~4 symbols at narrowband rate
        _pwrs = np.array([
            float(np.mean(np.abs(iq[i:i+_burst_len])**2))
            for i in range(0, len(iq) - _burst_len, _burst_len)
        ])
        if len(_pwrs) > 2:
            _burst_idx = int(np.argmax(_pwrs))
            _burst = iq[_burst_idx * _burst_len:(_burst_idx + 1) * _burst_len]
            _spec = np.abs(np.fft.fftshift(_fft(_burst)))
            _spec_freqs = np.fft.fftshift(np.fft.fftfreq(len(_burst), 1 / fs))
            # Zero DC bins to avoid picking the LO leak.
            _dc_bins = max(1, int(0.005 * len(_burst)))  # ±0.5% bins around DC
            _mid = len(_burst) // 2
            _spec[_mid - _dc_bins:_mid + _dc_bins + 1] = 0
            _spec_peak_hz = float(_spec_freqs[int(np.argmax(_spec))])
            # Only add if it's a meaningful offset (>5 bin equivalents) and
            # within ±fs/2.  Avoids near-DC echos that would alias.
            if (abs(_spec_peak_hz) > 5 * bw / N
                    and abs(_spec_peak_hz) < fs / 2 - bw / 4):
                found_cfos_hz.append(_spec_peak_hz)
                print("  DIAG spectrum-fb: burst peak at %.0f Hz, "
                      "adding as Pass-2 candidate" % _spec_peak_hz)

    # ---- Wideband CFO fallback ----
    # Run Schmidl-Cox on the full 4MHz recording to find the actual signal
    # position, then measure its CFO from mean instantaneous frequency and
    # hand it to pass 2 for re-centering.  This runs whenever pass 1 failed
    # (found_cfos_hz may already have a wrong bin from a spurious preamble
    # detection — the wideband measurement is a second independent estimate).
    if dec > 1 and len(iq) >= N * dec * 12:
        lag_wb = N * dec
        auto_wb = iq[lag_wb:] * np.conj(iq[:-lag_wb])
        n_win_wb = len(auto_wb) // lag_wb
        if n_win_wb >= 6:
            win_sz = min(6, n_win_wb)
            # sliding-window sum of |autocorr| / power — normalized Schmidl-Cox
            mag_wb = np.abs(
                np.array([np.mean(auto_wb[i * lag_wb:(i + 1) * lag_wb])
                          for i in range(n_win_wb)]))
            pwr_wb = np.array([np.mean(np.abs(iq[lag_wb + i * lag_wb:
                                                   lag_wb + (i + 1) * lag_wb]) ** 2)
                               for i in range(n_win_wb)])
            mag_wb_norm = np.where(pwr_wb > 1e-12, mag_wb / (pwr_wb + 1e-30), 0.0)
            cs = np.cumsum(mag_wb_norm)
            sums = cs[win_sz - 1:] - np.concatenate(([0.0], cs[:-win_sz]))
            # Examine top-3 Schmidl-Cox windows (not just the best) to find
            # signals that may be OUTSIDE the ±BW/2 pass-1 window (carrier
            # outside ±250 kHz is invisible to pass-1 but detectable here).
            seen_wins = set()
            n_tops = min(3, len(sums))
            top_wins = sorted(range(len(sums)), key=lambda i: sums[i], reverse=True)[:n_tops]
            for best_win in top_wins:
                p0 = best_win * lag_wb
                if p0 in seen_wins:
                    continue
                seen_wins.add(p0)
                # Use mean instantaneous frequency over exactly ONE symbol
                # to estimate the carrier.  Single-symbol averaging is accurate
                # (|error| < BW/N ≈ 4 kHz); multi-symbol averaging wraps at
                # symbol boundaries and gives wrong result.
                seg_wb = iq[p0:p0 + lag_wb]
                if len(seg_wb) < lag_wb:
                    continue
                if_wb = np.angle(seg_wb[1:] * np.conj(seg_wb[:-1]))
                cfo_hz_wb = float(np.mean(if_wb)) * fs / (2 * np.pi)
                # When a LoRa chirp symbol boundary falls inside the measurement
                # window, the mean IF is biased by approximately ±BW relative to
                # the true carrier (the discontinuous phase jump at the wrap
                # point contributes ∓BW to the average).  Generate all three
                # BW-alias candidates.
                #
                # recrop_centered(cfo_cand) shifts the carrier from its true
                # position to (true_carrier - cfo_cand); for the crop to capture
                # the signal we need |true_carrier - cfo_cand| < BW/2, and since
                # true_carrier ∈ {raw, raw±BW} exactly one candidate is valid.
                # We ALWAYS try all three.  The earlier code skipped the ±BW
                # aliases whenever |raw| < BW/2, assuming "pass-1 already covered
                # that range" — but pass-1 only crops the *center* ±BW/2 of the
                # capture around the GATE's reported carrier.  When the gate's
                # centroid lands >BW/2 off the true carrier (proven on dense
                # same-channel hop bursts: a 161 kHz error at BW=250k), the true
                # carrier sits at raw-BW, OUTSIDE pass-1's crop, yet |raw| can
                # still be < BW/2 — so the skip dropped the only candidate that
                # would have recovered the packet.  The extra decode attempts
                # are budget-gated and deduped by tried_shifts, so always adding
                # them costs nothing on captures that decode early.
                for cfo_cand in [cfo_hz_wb, cfo_hz_wb + bw, cfo_hz_wb - bw]:
                    if abs(cfo_cand) >= fs / 2:
                        continue
                    found_cfos_hz.append(cfo_cand)
                    print("  DIAG wideband-fb: signal at %.0f Hz offset (raw=%.0f), re-centering" % (cfo_cand, cfo_hz_wb))

    # ---- Pass 2: re-center on each found signal frequency and decode ----
    # The preamble bin at 1x rate is ambiguous for bins > N/2 (aliasing).
    # Try both +cfo and -(BW-cfo) to resolve.
    #
    # SEED THE SEARCH WITH THE DETERMINISTIC ±bw/2, ±bw ALIAS SHIFTS, AND TRY
    # THEM FIRST.  When the gate's centroid lands on a half- or full-bandwidth
    # alias of the true carrier (a known LoRa FFT-bin ambiguity), the signal
    # sits at the edge of / outside PASS-1's center ±bw/2 crop, so PASS 1 fails.
    # These four offsets are by far the most common centering errors, so trying
    # them BEFORE the spectrum-fb / IF-based candidates (which are more expensive
    # and sometimes point the wrong way) lets a marginal off-centre capture
    # decode in the first shift or two.  This is critical for the budget-limited
    # LIVE path: proven on SF7 live Test18 hop1 — capture off by exactly +bw/2,
    # found only after a long re-centre sweep at 40 s budget; as a primary
    # candidate it decodes near-instantly (well within the live ~6 s budget).
    # ±bw/2 also covers the Test5 hop1 case (carrier at -461 kHz; IF estimates
    # pointed at +273 kHz with no alias close enough).  Reordering only —
    # tried_shifts dedups and CRC-16 + decrypt-validity gate any false match, so
    # this adds no new false-positive risk and costs nothing when PASS 1 worked
    # (the fast-path skip returns before here).  SF/BW-agnostic.
    found_cfos_hz = [bw / 2.0, -bw / 2.0, bw, -bw] + list(found_cfos_hz)
    tried_shifts = set()
    for cfo_hz_raw in found_cfos_hz:
        # Two possible interpretations of the bin
        shifts_to_try = [cfo_hz_raw]
        if cfo_hz_raw > bw / 2:
            shifts_to_try.append(cfo_hz_raw - bw)  # wrapped interpretation
        elif cfo_hz_raw < -bw / 2:
            shifts_to_try.append(cfo_hz_raw + bw)

        for shift_hz in shifts_to_try:
            if _BUDGET_S > 0 and (time.process_time() - _process_start) > _BUDGET_S:
                print(f"\n  [BUDGET] decode budget {_BUDGET_S:.1f}s exhausted before PASS 2 — bail")
                return
            # Round to nearest 100 Hz for dedup
            shift_key = round(shift_hz / 100)
            if shift_key in tried_shifts or abs(shift_hz) < 5 * bw / N:
                continue  # already tried or near DC
            tried_shifts.add(shift_key)

            print("\n  === PASS 2: re-center by %.0f Hz ===" % shift_hz)
            iq1_pass2 = recrop_centered(shift_hz)

            p2_skip = list(skip_bins)  # start with known bad bins
            # Counter for consecutive no-header attempts at THIS shift (same
            # semantics as PASS 1).  Per-shift reset means a clean PASS 2
            # CFO-recovery attempt at a different shift is unaffected by what
            # happened at the previous shift.
            _consec_no_header_p2 = 0
            for attempt in range(MAX_ATTEMPTS):
                if _BUDGET_S > 0 and (time.process_time() - _process_start) > _BUDGET_S:
                    return
                if _consec_no_header_p2 >= 3:
                    print(f"\n  [BAIL] {_consec_no_header_p2} consecutive no-header PASS 2 attempts at this shift")
                    break
                if attempt > 0:
                    print("\n  --- Pass2 Retry #%d ---" % attempt)
                _d_p2 = {}
                result = _decode_attempt(iq1_pass2.copy(), sf, bw, N, ppm, fs, dec, name, Counter,
                                         skip_bins=p2_skip, diag_out=_d_p2)
                if result is None:
                    break
                status, mask_start, mask_len, p2_bin = result
                if status == 'WRONG_SF':
                    break  # dominant signal is a different SF — skip this shift
                # Update bail counter using broad-PL `header_decoded` flag so
                # ALL protocols (any 4..237 byte payload) reset it.
                if _d_p2.get('header_decoded'):
                    _consec_no_header_p2 = 0
                elif status == 'FAIL':
                    _consec_no_header_p2 += 1
                # On OK we do NOT return: a badly-centred gate capture (carrier
                # >BW/2 off, so PASS 1 failed) routinely holds several
                # time-separated packets at this SAME carrier — hop0 + hop1, or
                # two adjacent messages relayed back-to-back.  The old "each file
                # is one packet → return" dropped every packet after the first.
                # Instead mask the decoded packet's samples and keep decoding
                # this recenter; _DECODED_SET dedups any re-decode at a later
                # recenter.  The outer candidate loop + budget bound the cost.
                if status != 'OK' and p2_bin is not None:
                    p2_skip.append(p2_bin)
                mask_end = min(len(iq1_pass2), mask_start + mask_len)
                iq1_pass2[mask_start:mask_end] = 0.0

    # ---- Exhaustive time-burst sweep (final recovery) ----
    # A dense capture can hold 3+ time-separated packets in one gate window
    # (e.g. hop1 of msg N, then hop0 + hop1 of msg N+1) — common at higher SF
    # where packets are long (~131 ms at SF9) and relay gaps pack them ~0.8-1.2 s
    # apart.  The mask+continue loop and PASS 2 sometimes extract only 2 of 3.
    # As a final pass, find EVERY burst by its power envelope and decode each in
    # TIME-ISOLATION (each burst's signal is individually clean — verified: a
    # missed hop0 decodes perfectly when isolated).  Recursive with
    # _allow_rescue=False so it can't re-enter itself; _DECODED_SET dedups, and
    # the AES decrypt-validity gate keeps an isolated noise blip from becoming a
    # false positive.
    if _allow_rescue and len(iq) > 0:
        _ew = max(1, int(0.005 * fs))               # 5 ms envelope window
        _ne = len(iq) // _ew
        if _ne > 4:
            _pw = (np.abs(iq[:_ne * _ew].reshape(_ne, _ew)) ** 2).mean(axis=1)
            _hi = _pw > (np.median(_pw) + 1e-12) * 8.0     # ~9 dB over median
            _minrun = max(1, int(round((8 * (2 ** sf) / bw) / 0.005)))  # >=8 syms
            _bursts = []
            _i = 0
            while _i < _ne:
                if _hi[_i]:
                    _j = _i
                    while _j < _ne and _hi[_j]:
                        _j += 1
                    if (_j - _i) >= _minrun:
                        _bursts.append((_i * _ew, _j * _ew))
                    _i = _j
                else:
                    _i += 1
            if len(_bursts) > 1:
                _guard = int(0.02 * fs)
                for (_bs, _be) in _bursts:
                    if _BUDGET_S > 0 and (time.process_time() - _process_start) > _BUDGET_S:
                        break
                    _iso = np.zeros_like(iq)
                    _a = max(0, _bs - _guard)
                    _b = min(len(iq), _be + _guard)
                    _iso[_a:_b] = iq[_a:_b]
                    try:
                        process_file(fpath, relay_after=relay_after,
                                     relay_before=relay_before, _iq_override=_iso,
                                     _allow_rescue=False, _budget_override=6.0)
                    except Exception:
                        pass


def _decode_attempt(iq1, sf, bw, N, ppm, fs, dec, name, Counter, skip_bins=None,
                    sweep_budget=2.0, diag_out=None):
    if skip_bins is None:
        skip_bins = []

    if _HARNESS_OUT:
        _HARNESS_ATTEMPT[0] += 1
        _harness_emit('attempt_start',
                      capture=(name.rsplit('/', 1)[-1] if name else ''),
                      sf=int(sf), bw=int(bw), iq_len=int(len(iq1)),
                      skip_bins=[int(b) for b in skip_bins])

    # Hardware fingerprint is computed AFTER preamble detection — see below —
    # because the precise carrier measurement needs to be confined to the
    # actual burst window (the capture has noise/silence around the burst,
    # which would dominate a whole-capture Welch centroid).
    _hw_fp = None

    t = np.arange(N, dtype=np.float64) / bw
    Ts = N / bw
    upchirp = np.exp(1j * np.pi * bw / Ts * t**2).astype(np.complex64)
    downchirp = np.conj(upchirp)

    # ---- Find preamble: 2-pass (matching analysis script) ----
    # Pass 1: Autocorrelation finds coarse preamble position (which N-sample
    #         boundary has strongest symbol-to-symbol correlation).
    # Pass 2: Sample-by-sample refinement ±N around that position, then
    #         dechirp all symbols to find runs.
    #
    # This works even when the signal has large CFO (>100kHz) because
    # autocorrelation measures period-N repetition regardless of frequency.

    def _find_coarse_positions(iq_data, n_top=3):
        """Find top-N coarse preamble positions via normalized Schmidl-Cox.

        Full Schmidl-Cox metric: M(d) = |P(d)| / R(d)
          P(d) = mean(r(n+N) * conj(r(n)))   — autocorrelation at lag N
          R(d) = mean(|r(n+N)|²)             — power normalization

        Normalization is critical: without it a strong broadband interferer
        can score higher than a weaker LoRa preamble even though it has no
        periodicity.  M(d) ∈ [0,1], rising toward 1 across the preamble and
        dropping sharply when the preamble ends (as the document describes).

        We also search at half-block offsets so preambles that begin near
        a block boundary (which splits coherent integration between two windows)
        are not missed by the coarse scan.
        """
        if len(iq_data) < N * 8:
            return [0]
        auto = iq_data[N:] * np.conj(iq_data[:-N])
        n_win = len(auto) // N
        if n_win < 2:
            return [0]

        # Normalized Schmidl-Cox per window (vectorised reshape-mean — byte-
        # identical to the old per-window list comprehensions).
        mag  = np.abs(auto[:n_win * N].reshape(n_win, N).mean(axis=1))
        pwr  = (np.abs(iq_data[N:N + n_win * N])**2).reshape(n_win, N).mean(axis=1)
        mag_norm = np.where(pwr > 1e-12, mag / (pwr + 1e-30), 0.0)

        # Also compute at half-block offset to catch preambles straddling boundaries
        half = N // 2
        n_win_h = (len(auto) - half) // N
        if n_win_h > 0:
            mag_h = np.abs(auto[half:half + n_win_h * N].reshape(n_win_h, N).mean(axis=1))
            pwr_h = (np.abs(iq_data[N + half:N + half + n_win_h * N])**2
                     ).reshape(n_win_h, N).mean(axis=1)
            mag_h_norm = np.where(pwr_h > 1e-12, mag_h / (pwr_h + 1e-30), 0.0)
        else:
            mag_h_norm = np.array([])

        win = min(6, n_win)
        if n_win <= win:
            # Short recording — return both offset 0 and half-block
            positions = [0]
            if half < len(iq_data) - N * 4:
                positions.append(half)
            return positions

        # Sliding 6-window sum for full-block positions
        cs = np.cumsum(mag_norm)
        sums = cs[win-1:] - np.concatenate(([0], cs[:-win]))

        # Sliding sum for half-block positions
        candidates = [(float(sums[i]), int(i) * N) for i in range(len(sums))]
        if len(mag_h_norm) >= win:
            cs_h = np.cumsum(mag_h_norm)
            sums_h = cs_h[win-1:] - np.concatenate(([0], cs_h[:-win]))
            for i in range(len(sums_h)):
                candidates.append((float(sums_h[i]), half + int(i) * N))

        candidates.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        positions = []
        for score, pos in candidates:
            # Deduplicate positions within N/2 of each other
            bucket = pos // (N // 2)
            if bucket not in seen:
                seen.add(bucket)
                positions.append(pos)
            if len(positions) >= n_top:
                break
        return positions if positions else [0]

    def _refine_start_xcorr(iq_data, coarse, n_preamble=8):
        """Refine coarse preamble position via instantaneous frequency cross-correlation.

        Robyns FOSDEM 2018: cross-correlate the received instantaneous frequency
        with that of a locally generated up-chirp.  The correlation peak is at
        the exact symbol boundary and is unaffected by CFO — a constant frequency
        offset adds a constant to IF without shifting the peak position.

        More noise-robust than brute-force dechirp energy sweeping: accumulates
        coherently over n_preamble symbol periods in O(M log M) total.
        """
        lo = max(1, coarse - 2 * N)
        hi = min(len(iq_data) - 1, coarse + 2 * N + n_preamble * N)
        if hi <= lo or hi - lo < N:
            return coarse

        window = iq_data[lo:hi + 1]
        # Instantaneous frequency: angle of successive-sample product
        if_rx = np.angle(window[1:] * np.conj(window[:-1])).astype(np.float64)

        # Reference IF for one value-0 up-chirp period (clean linear sweep),
        # tiled over n_preamble periods to strengthen the correlation.
        if_ref_one = np.angle(upchirp[1:] * np.conj(upchirp[:-1]))  # N-1 samples
        if_ref = np.tile(if_ref_one, n_preamble).astype(np.float64)

        ref_len = len(if_ref)
        n_valid = len(if_rx) - ref_len + 1
        if n_valid < 1:
            return coarse

        # FFT cross-correlation via rfft (real inputs).  Pad n_fft up to a
        # smooth (next_fast_len) size: linear xcorr only needs n_fft >=
        # len(if_rx)+ref_len-1, so any larger n is byte-identical in the valid
        # region [:n_valid] but a fast-factorable length avoids slow prime-size
        # FFTs.
        n_fft = _next_fast_len(len(if_rx) + ref_len)
        xcorr = np.fft.irfft(
            _rfft(if_rx, n=n_fft) * np.conj(_rfft(if_ref, n=n_fft)),
            n=n_fft
        )
        # xcorr[k] = Σ_n if_rx[k+n]*if_ref[n] → peak at start = lo+k
        best_lag = int(np.argmax(xcorr[:n_valid]))
        return lo + best_lag

    def _refine_start(iq_data, coarse, n_score=8):
        """Fallback: refine coarse position by maximizing dechirp peak energy.

        PERF: batches per-symbol FFTs (one batched call per offset).  For
        large-N (SF11/12) chunks the scratch array so memory stays bounded.
        Bit-identical to the per-FFT loop (max/argmax along axis=1 produces
        the same values as the per-row scalar version).

        PERF (coarse pass): uses min(n_score, 4) symbols instead of full
        n_score for the wide-range coarse search.  The coarse pass only
        picks a winner at step-granularity; the fine pass at step=1 then
        rescores with full n_score and chooses the final offset.  Halves
        the dominant FFT work in this function with no observed loss of
        offset selection on the harness reference (payload bytes
        identical, only intermediate margins shift).

        PERF (range): _find_coarse_positions returns positions at
        half-block resolution (multiples of N/2), so the true start is
        within ±N/4 of `coarse`.  Searching ±N/2 instead of ±N halves
        the coarse-offset count again while still covering the input
        position's worst-case uncertainty.
        """
        # ±N/2 covers _find_coarse_positions' half-block quantisation at
        # large N where it pays off.  At small N (SF7-9) the FFT is so
        # cheap that ±N is fine and the wider safety margin protects
        # borderline captures from spilling into PASS 2.  Gate on the
        # same N≥1024 threshold as the n_score reduction above.
        _refine_half_range = (N // 2) if N >= 1024 else N
        lo = max(0, coarse - _refine_half_range)
        hi = min(len(iq_data) - N * max(1, n_score), coarse + _refine_half_range)
        if hi <= lo:
            return coarse
        step = max(1, dec // 2)
        # Reduced symbol count for the wide coarse search at large N where
        # FFT cost dominates (SF≥10, N≥1024).  At small N the FFT is cheap
        # so the reduced n_score doesn't save much wall but slightly
        # noisier offset picks can push borderline captures into PASS 2 —
        # net wall regression observed live at SF7 (N=128).  Gating on SF
        # via N restores SF7's clean fast-path while keeping the SF11/12
        # win.  Threshold N=1024 chosen so SF7/8/9 use full n_score and
        # SF10/11/12 use the reduction.
        n_score_coarse = min(n_score, 4) if N >= 1024 else n_score
        # Smaller-N coarse FFT for SF≥10: truncate dechirped segment to
        # N/2 samples before FFT (conservative — N/4 caused ambiguous
        # SF12 hop1 result in one live run; N/2 halves the SNR loss).
        # Fine pass always re-evaluates at full N so final pick is unbiased.
        COARSE_DIV = 2
        coarse_n_fft = (N // COARSE_DIV) if N >= 1024 else None

        # Memory budget for the batched scratch: 16 MB worth of complex64.
        # `K` = how many offsets to stack per batched FFT call.  At small N
        # (SF7-9) K covers the entire range in one call; at large N (SF11/12)
        # K shrinks so the scratch stays bounded (was 1.6 GB before this).
        BATCH_BYTES = 16 * 1024 * 1024

        def _batched_pass(offsets_iter, n_sc, coarse_n=None):
            offsets = [o for o in offsets_iter
                       if 0 <= o and o + n_sc * N <= len(iq_data)]
            if not offsets:
                return coarse, 0.0
            n_offs = len(offsets)
            K_local = max(1, BATCH_BYTES // max(1, n_sc * N * 8))
            all_maxes = np.empty((n_offs, n_sc), dtype=np.float32)
            all_arg = np.empty((n_offs, n_sc), dtype=np.int32)
            for bs in range(0, n_offs, K_local):
                batch = offsets[bs:bs + K_local]
                kb = len(batch)
                segs = np.empty((kb * n_sc, N), dtype=iq_data.dtype)
                for oi, off in enumerate(batch):
                    segs[oi*n_sc:(oi+1)*n_sc] = (
                        iq_data[off:off + n_sc*N].reshape(n_sc, N))
                segs *= downchirp
                if coarse_n is not None and coarse_n < N:
                    segs_for_fft = segs[:, :coarse_n]
                else:
                    segs_for_fft = segs
                ff = np.abs(_fft(segs_for_fft, axis=1)).astype(np.float32, copy=False)
                all_maxes[bs:bs+kb] = ff.max(axis=1).reshape(kb, n_sc)
                all_arg[bs:bs+kb] = ff.argmax(axis=1).reshape(kb, n_sc)
            scores = all_maxes.sum(axis=1).astype(np.float64)
            # Consistent-bin bonus: max bincount per row.  Vectorised via
            # sort+diff: in a sorted row, a run of ≥4 identical bins → ≥3
            # consecutive zeros in diff(sorted_row).  Compute max-run-length
            # per row with one numpy pass (no Python per-row loop).
            sorted_arg = np.sort(all_arg, axis=1)
            same = np.diff(sorted_arg, axis=1) == 0    # (n_offs, n_sc-1) bool
            # Per-row max consecutive-True length: a vectorised cumulative
            # reset-counter (cumsum that zeros at every False).  Equivalent
            # to the per-row run-length code, all-numpy.
            run = np.zeros_like(same, dtype=np.int32)
            if same.shape[1] > 0:
                run[:, 0] = same[:, 0].astype(np.int32)
                for c in range(1, same.shape[1]):
                    run[:, c] = np.where(same[:, c], run[:, c-1] + 1, 0)
                max_run = run.max(axis=1)              # max consecutive True per row
                scores[max_run + 1 >= 4] *= 1.5
            bi = int(np.argmax(scores))
            return offsets[bi], float(scores[bi])

        # Coarse pass over the wide range — reduced n_score + truncated FFT.
        best_off, best_score = coarse, 0.0
        off0, s0 = _batched_pass(range(lo, hi + 1, step), n_score_coarse,
                                 coarse_n=coarse_n_fft)
        if s0 > best_score:
            best_off, best_score = off0, s0
        # Fine pass around the coarse winner — full n_score for fidelity.
        fine_lo = max(lo, best_off - step)
        fine_hi = min(hi, best_off + step)
        off1, s1 = _batched_pass(range(fine_lo, fine_hi + 1), n_score)
        # Fine pass scores have different (larger) magnitudes than the
        # reduced-n coarse pass; compare against the fine score from the
        # coarse winner re-evaluated at full n_score, not the coarse score.
        coarse_off_rescored, s_coarse_full = _batched_pass([best_off], n_score)
        if s1 > s_coarse_full:
            best_off = off1
        return int(best_off)

    def _scan_from(iq_data, start):
        """Dechirp all symbols from start position, find runs."""
        n_total = (len(iq_data) - start) // N
        syms = []
        if n_total > 0:
            # Batch every symbol into one (n_total, N) dechirp + 2D FFT instead
            # of n_total separate 1-D FFTs — pocketfft runs the rows together
            # (and the per-row argmax/max/mean vectorise).  Verified byte-
            # identical to the old per-symbol loop (FFT, argmax, max, mean all
            # bit-exact along the contiguous axis).
            block = iq_data[start:start + n_total * N].reshape(n_total, N)
            fft_out = np.abs(_fft(block * downchirp, axis=1))
            bins = np.argmax(fft_out, axis=1)
            pmr = 10 * np.log10(fft_out.max(axis=1)
                                / (fft_out.mean(axis=1) + 1e-30) + 1e-15)
            syms = [(i, int(bins[i]), float(pmr[i])) for i in range(n_total)]
        strong = [(i, b, p) for i, b, p in syms if p > 8]
        runs = []
        cur = []
        for i, b, p in strong:
            if not cur:
                cur = [(i, b, p)]
                continue
            cur_bin = circular_mean_bin([bb for _, bb, _ in cur], N)
            if circ_dist(b, cur_bin, N) <= max(1, sf - 6) and i - cur[-1][0] <= 2:
                cur.append((i, b, p))
            else:
                if len(cur) >= 4:
                    runs.append(cur)
                cur = [(i, b, p)]
        if len(cur) >= 4:
            runs.append(cur)
        return runs, syms

    # Find coarse positions, refine each, scan for preamble runs
    coarse_positions = _find_coarse_positions(iq1, n_top=3)
    all_runs = []
    all_syms = []
    best_phase = 0
    for coarse in coarse_positions:
        refined = _refine_start_xcorr(iq1, coarse)
        runs, syms = _scan_from(iq1, refined)
        if not runs:
            # Fallback: brute-force dechirp energy sweep
            refined = _refine_start(iq1, coarse, n_score=8)
            runs, syms = _scan_from(iq1, refined)
        if runs:
            for r in runs:
                all_runs.append((r, refined))
            if not all_syms:
                all_syms = syms
                best_phase = refined

    # Fallback: also scan from offset 0 (catches preambles at file start
    # that autocorrelation misses due to resampling artifacts)
    runs0, syms0 = _scan_from(iq1, 0)
    if runs0:
        # Only add runs with bins not already found
        existing_bins = set()
        for r, _ in all_runs:
            rb = circular_mean_bin([b for _, b, _ in r], N)
            rb_s = rb if rb <= N//2 else rb - N
            existing_bins.add(round(rb_s / 20))  # cluster by ~20 bins
        for r in runs0:
            rb = circular_mean_bin([b for _, b, _ in r], N)
            rb_s = rb if rb <= N//2 else rb - N
            if round(rb_s / 20) not in existing_bins:
                all_runs.append((r, 0))
                existing_bins.add(round(rb_s / 20))
        if not all_syms:
            all_syms = syms0
            best_phase = 0
    # Always keep syms0 for diagnostics even when no runs found — the per-symbol
    # PMR values help distinguish "no signal" from "wrong SF" cases.
    if not all_syms and syms0:
        all_syms = syms0

    # Global fallback: IF cross-correlation over the ENTIRE file.
    # Catches any case where Schmidl-Cox gave a poor coarse estimate and
    # the ±2N xcorr window searched the wrong area.  The global FFT xcorr
    # finds the true preamble location regardless of where it landed in the
    # recording.  Only runs when no preamble has been found yet.
    if not all_runs and len(iq1) >= N * 12:
        if_rx_g = np.angle(iq1[1:] * np.conj(iq1[:-1])).astype(np.float64)
        if_ref_one = np.angle(upchirp[1:] * np.conj(upchirp[:-1]))
        if_ref_g = np.tile(if_ref_one, 8).astype(np.float64)
        n_valid_g = len(if_rx_g) - len(if_ref_g) + 1
        if n_valid_g >= 1:
            # Pad to a smooth length — full-capture xcorr at a prime-ish size is
            # pathologically slow (e.g. 770840 → max prime factor 2753, 159 ms;
            # next_fast_len → 23 ms, 6.8×).  Byte-identical in the valid region.
            n_fft_g = _next_fast_len(len(if_rx_g) + len(if_ref_g))
            xcorr_g = np.fft.irfft(
                _rfft(if_rx_g, n=n_fft_g) *
                np.conj(_rfft(if_ref_g, n=n_fft_g)),
                n=n_fft_g)
            best_g = int(np.argmax(xcorr_g[:n_valid_g])) + 1
            runs_g, syms_g = _scan_from(iq1, best_g)
            if not runs_g:
                # Refine around the global xcorr peak
                best_g = _refine_start(iq1, best_g, n_score=8)
                runs_g, syms_g = _scan_from(iq1, best_g)
            if runs_g:
                for r in runs_g:
                    all_runs.append((r, best_g))
                if not all_syms:
                    all_syms = syms_g
                    best_phase = best_g
            # Always keep syms_g for diagnostics even when no runs found
            if not all_syms and syms_g:
                all_syms = syms_g

    # Flatten: sort by length DESC, start time ASC on ties
    all_runs.sort(key=lambda x: (len(x[0]), -x[0][0][0]), reverse=True)
    # Extract runs and phases
    run_phases = [phase for _, phase in all_runs]
    all_runs = [r for r, _ in all_runs]

    if not all_runs:
        print("  NO PREAMBLE")
        try:
            syms_sorted = sorted(all_syms, key=lambda x: x[2], reverse=True)[:8]
            if syms_sorted:
                pmr_str = ", ".join([f"{i}:{b}:{p:.1f}" for i, b, p in syms_sorted])
                print(f"  DIAG preamble: top PMR syms (i:bin:pmr_dB): {pmr_str}")
            strong8 = [(i, b, p) for i, b, p in all_syms if p > 8]
            strong6 = [(i, b, p) for i, b, p in all_syms if p > 6]
            if strong6:
                hist = Counter([b for _, b, _ in strong6]).most_common(3)
                hist_str = ", ".join([f"{b}x{c}" for b, c in hist])
                print(f"  DIAG preamble: PMR>6dB bin histogram: {hist_str}")
            print(f"  DIAG preamble: count PMR>8dB={len(strong8)}  count PMR>6dB={len(strong6)}")

            # Multi-SF Schmidl-Cox scan: try all SFs at the 1× BW rate.
            # iq1 is already at BW sample rate (e.g. 500 kHz for BW=500k).
            # For each SF, the preamble has N_alt=2^SF_alt period; a genuine
            # preamble scores > 3.0 on the 6-window sliding sum; noise < 0.5.
            # This identifies third-party LoRa at a different SF (e.g. Amazon
            # Sidewalk SF8, LoRaWAN SF9, neighbor on LONG_SLOW SF12).
            if len(iq1) >= 2048:
                alt_scores = {}
                for sf_alt in [6, 7, 8, 9, 10, 11, 12]:
                    N_alt = 1 << sf_alt  # lag in 1× (BW) samples
                    if N_alt >= len(iq1) // 8:
                        continue
                    auto_alt = iq1[N_alt:] * np.conj(iq1[:-N_alt])
                    n_win_alt = len(auto_alt) // N_alt
                    if n_win_alt < 6:
                        continue
                    # Vectorised reshape — avoids Python loop, fast for small N
                    chunks = auto_alt[:n_win_alt * N_alt].reshape(n_win_alt, N_alt)
                    mag_alt = np.abs(chunks.mean(axis=1))
                    pwr_chunks = (np.abs(iq1[N_alt:N_alt + n_win_alt * N_alt])**2
                                  ).reshape(n_win_alt, N_alt)
                    pwr_alt = pwr_chunks.mean(axis=1)
                    mnorm = np.where(pwr_alt > 1e-12, mag_alt / (pwr_alt + 1e-30), 0.0)
                    # Sliding 6-window sum — peak lands where preamble sits
                    cs6 = np.cumsum(mnorm)
                    sums6 = cs6[5:] - np.concatenate(([0.0], cs6[:-6]))
                    alt_scores[sf_alt] = float(np.max(sums6))
                if alt_scores:
                    ranked = sorted(alt_scores.items(), key=lambda x: x[1], reverse=True)
                    score_str = "  ".join(f"SF{s}={v:.2f}" for s, v in ranked)
                    print(f"  DIAG multi-sf: best 6-win Schmidl-Cox (>3=preamble, <0.5=noise): {score_str}")
                    best_sf, best_score = ranked[0]
                    sf7_score = alt_scores.get(sf, 0.0)
                    # Bail out early only when our SF is at noise floor (<0.5)
                    # while a different SF is clearly preamble-like (>3), AND
                    # there are very few strong PMR symbols for our SF.
                    #
                    # The PMR guard prevents false aborts when the capture
                    # starts mid-packet (preamble already passed).  In that
                    # case: our SF's Schmidl-Cox = near zero (no repeated
                    # same-bin symbols → no lag-N coherence), but the data
                    # symbols dechirp cleanly at our SF's rate → many strong
                    # PMR hits.  That combination means "capture missed the
                    # preamble" not "wrong SF".  The signal IS our SF.
                    #
                    # SF6 lag=64 is exactly 1/64 of an SF12/125k symbol; the
                    # cross-boundary auto-correlation sample that escapes the
                    # within-symbol cancellation can inflate SF6 scores
                    # spuriously for any SF12 data capture.
                    if (best_sf != sf and best_score > 3.0
                            and sf7_score < 0.5
                            and len(strong8) < 4):
                        print(f"  DIAG multi-sf: signal looks like SF{best_sf} (not SF{sf}) — aborting")
                        return ('WRONG_SF', 0, 0, None)
                    elif best_sf != sf and best_score > 3.0 and sf7_score < 0.5:
                        print(f"  DIAG multi-sf: SF{best_sf} dominant but {len(strong8)} strong PMR symbols"
                              f" suggest signal is SF{sf} — preamble likely before capture window")
        except Exception:
            pass
        return None

    if len(all_runs) > 1:
        print("  Found %d preamble candidates:" % len(all_runs))
        for ri, r in enumerate(all_runs[:6]):
            r_bin = circular_mean_bin([b for _, b, _ in r], N)
            print("    #%d: len=%d bin=%d start_sym=%d pmr=%.1fdB" % (
                ri+1, len(r), r_bin, r[0][0], np.mean([p for _, _, p in r])))

    def refine_preamble_start(pre_loc, pre_bin, nscore_syms):
        """Refine preamble start by maximising sum-of-FFT-peaks.

        PERF: batched per-symbol FFTs, with the batch size capped so the
        scratch array stays bounded on large-N (SF11/12) — the previous
        single-call batching allocated 1+ GB at SF12 and was 70 s slower
        than the per-FFT loop.

        PERF (coarse pass): wide coarse search uses min(nscore_syms, 4)
        symbols; fine pass at step=1 uses full nscore_syms.

        PERF (smaller-N coarse, SF≥10): The coarse pass FFTs only the
        first N/COARSE_DIV samples of each dechirped symbol — a 4× FFT-
        size reduction costs ~5× less per FFT (N log N scaling).  After
        dechirp, the LoRa chirp's tone is a constant-frequency sinusoid;
        truncating to N/4 samples drops SNR by 6 dB but PRESERVES the
        peak bin's relative position vs other offsets (scaled by N/4 / N).
        Only used for ranking; the fine pass re-evaluates the coarse
        winner at full N so the final offset score is unbiased.
        """
        lo = max(0, pre_loc - N)
        hi = min(len(iq1) - N * max(1, nscore_syms), pre_loc + N)
        if hi < lo:
            return pre_loc
        coarse_step = max(1, dec // 2)
        # SF-conditional: only reduce n_score at large N (SF≥10) where the
        # FFT cost is dominant.  At small N (SF7-9) the FFT is cheap and
        # the slightly noisier coarse pick can push borderline captures
        # into PASS 2, which regresses SF7 wall by ~17 %.  See the
        # twin gate in _refine_start above.
        nsc_coarse = min(nscore_syms, 4) if N >= 1024 else nscore_syms
        # Smaller-N coarse FFT for SF≥10 — truncate dechirped segment to
        # N/COARSE_DIV samples before FFT.  At SF12 N=4096: FFT at 1024
        # is ~4.8× cheaper.  Disable at SF<10 where FFT is already cheap
        # and the SNR loss matters more for borderline captures.
        COARSE_DIV = 4
        coarse_n = (N // COARSE_DIV) if N >= 1024 else None
        BATCH_BYTES = 16 * 1024 * 1024

        def _batched_pass(offsets_iter, n_sc, coarse_n=None):
            """If coarse_n is set, FFT only the first coarse_n samples of
            each dechirped symbol (cheaper, lower frequency resolution).
            Otherwise FFT at full N."""
            offsets = [o for o in offsets_iter
                       if 0 <= o and o + n_sc * N <= len(iq1)]
            if not offsets:
                return pre_loc, -1.0
            n_offs = len(offsets)
            K_local = max(1, BATCH_BYTES // max(1, n_sc * N * 8))
            all_maxes = np.empty((n_offs, n_sc), dtype=np.float32)
            all_arg = np.empty((n_offs, n_sc), dtype=np.int32)
            for bs in range(0, n_offs, K_local):
                batch = offsets[bs:bs + K_local]
                kb = len(batch)
                segs = np.empty((kb * n_sc, N), dtype=iq1.dtype)
                for oi, off in enumerate(batch):
                    segs[oi*n_sc:(oi+1)*n_sc] = (
                        iq1[off:off + n_sc*N].reshape(n_sc, N))
                segs *= downchirp
                # Truncate AFTER dechirp — the chirp's tone is constant-
                # frequency for the symbol duration, so the first N/D
                # samples carry the same bin information at lower SNR.
                if coarse_n is not None and coarse_n < N:
                    segs_for_fft = segs[:, :coarse_n]
                else:
                    segs_for_fft = segs
                ff = np.abs(_fft(segs_for_fft, axis=1)).astype(np.float32, copy=False)
                all_maxes[bs:bs+kb] = ff.max(axis=1).reshape(kb, n_sc)
                all_arg[bs:bs+kb] = ff.argmax(axis=1).reshape(kb, n_sc)
            scores = all_maxes.sum(axis=1).astype(np.float64)
            # Consistent-bin bonus (≥4 identical bins) — vectorised.
            sorted_arg = np.sort(all_arg, axis=1)
            same = np.diff(sorted_arg, axis=1) == 0
            if same.shape[1] > 0:
                run = np.zeros_like(same, dtype=np.int32)
                run[:, 0] = same[:, 0].astype(np.int32)
                for c in range(1, same.shape[1]):
                    run[:, c] = np.where(same[:, c], run[:, c-1] + 1, 0)
                max_run = run.max(axis=1)
                scores[max_run + 1 >= 4] *= 1.5
            bi = int(np.argmax(scores))
            return offsets[bi], float(scores[bi])

        best_off, best_score = pre_loc, -1.0
        # Coarse pass: reduced n_score AND truncated FFT for speed.
        off0, _ = _batched_pass(range(lo, hi + 1, coarse_step), nsc_coarse,
                                coarse_n=coarse_n)
        # Fine pass around coarse winner: full nscore_syms AND full N for
        # the final offset selection — the truncated coarse score is not
        # directly comparable to the full-N fine score.
        _, s_coarse_full = _batched_pass([off0], nscore_syms)
        best_off, best_score = off0, s_coarse_full
        fine_lo = max(lo, best_off - coarse_step)
        fine_hi = min(hi, best_off + coarse_step)
        off1, s1 = _batched_pass(range(fine_lo, fine_hi + 1), nscore_syms)
        if s1 > best_score:
            best_off = off1
        return int(best_off)

    cand_infos = []
    # Each run produces multiple candidates at different start-position
    # estimates.  refine_preamble_start sometimes lands `pre_loc + k*samples`
    # off from the true symbol boundary on signals with large CFO — the
    # dechirp peak then sits at the wrong bin and header decode misalignment
    # cascades into the payload (header CHK happens to pass with right PL/CR
    # but the payload symbols are sampled at offset boundaries → CRC FAIL).
    # Adding the un-refined position as a second candidate lets the scoring
    # stage pick whichever gives the cleaner SC run + SFD.  SF/BW-agnostic:
    # all offsets are in `N` units which scales with sf and rate.
    for run_idx, run in enumerate(all_runs[:min(6, len(all_runs))]):
        run_len = len(run)
        run_mean_pmr = float(np.mean([p for _, _, p in run]))
        run_bin = circular_mean_bin([b for _, b, _ in run], N)
        phase_for_run = run_phases[run_idx] if run_idx < len(run_phases) else best_phase
        pre_loc = phase_for_run + run[0][0] * N
        refined_start = refine_preamble_start(pre_loc, run_bin, min(max(run_len, 8), 12))
        candidate_starts = [refined_start]
        if pre_loc != refined_start and 0 <= pre_loc < len(iq1) - N * 12:
            candidate_starts.append(pre_loc)
        for preamble_start_cand in candidate_starts:
            rescan_n = min(30, (len(iq1) - preamble_start_cand) // N)
            # Batch all rescan symbols in one scipy.fft call (3-10x faster than serial loop).
            _n_valid = min(rescan_n, (len(iq1) - preamble_start_cand) // N)
            if _n_valid > 0:
                _segs = iq1[preamble_start_cand:preamble_start_cand + _n_valid*N].reshape(_n_valid, N)
                _ffts = np.abs(_fft(_segs * downchirp, axis=1))
                _maxes = _ffts.max(axis=1)
                _means = _ffts.mean(axis=1)
                _argmaxes = _ffts.argmax(axis=1)
                _pmrs = 10*np.log10(_maxes / (_means + 1e-30) + 1e-15)
                rescan_syms = [(i, int(_argmaxes[i]), float(_pmrs[i])) for i in range(_n_valid)]
            else:
                rescan_syms = []

            # Detect actual preamble bin from the refined position (may differ from run_bin
            # if the initial run was found at a misaligned scan boundary).
            strong2 = [(i, b, p) for i, b, p in rescan_syms if p > 8]
            if len(strong2) >= 4:
                early_bins = [b for _, b, _ in strong2[:8]]
                effective_bin = Counter(early_bins).most_common(1)[0][0]
            else:
                effective_bin = run_bin

            best_run2, cur2 = [], []
            for i, b, p in strong2:
                if not cur2:
                    cur2 = [(i, b, p)]
                elif circ_dist(b, effective_bin, N) <= 2 and i - cur2[-1][0] <= 2:
                    cur2.append((i, b, p))
                else:
                    if (len(cur2), np.mean([pp for _, _, pp in cur2]) if cur2 else -1.0) > (
                            len(best_run2), np.mean([pp for _, _, pp in best_run2]) if best_run2 else -1.0):
                        best_run2 = cur2
                    cur2 = [(i, b, p)]
            if (len(cur2), np.mean([pp for _, _, pp in cur2]) if cur2 else -1.0) > (
                    len(best_run2), np.mean([pp for _, _, pp in best_run2]) if best_run2 else -1.0):
                best_run2 = cur2

            if len(best_run2) >= 4:
                preamble_bin_cand = circular_mean_bin([b for _, b, _ in best_run2], N)
                pre_last_i_cand = int(best_run2[-1][0])
                pre_len_cand = len(best_run2)
            else:
                preamble_bin_cand = effective_bin
                pre_last_i_cand = int(run[-1][0] - run[0][0])
                pre_len_cand = run_len

            sfd_i_cand = pre_last_i_cand + 3
            sfd_ok_cand = 0
            sfd_metric = 0.0
            sfd_cfo_bins_cand = []
            for si in range(2):
                p = preamble_start_cand + (sfd_i_cand + si) * N
                if p + N > len(iq1):
                    break
                seg = iq1[p:p + N]
                uf = np.abs(_fft(seg * downchirp))
                df = np.abs(_fft(seg * upchirp))
                up_e = float(np.max(uf)) ** 2
                dn_e = float(np.max(df)) ** 2
                ratio = dn_e / (up_e + 1e-30)
                sfd_metric += 10.0 * np.log10(ratio + 1e-30)
                if dn_e > up_e * 0.5:
                    sfd_ok_cand += 1
                # SFD CFO measurement (zoomed FFT): an SFD downchirp dechirped
                # with upchirp gives a tone at the residual CFO bin.  Used as
                # a tiebreak to prefer preamble candidates whose SFD aligns
                # well (small residual) — this is the signal that the
                # preamble cyclic-shift was the right one.  See "SFD CFO
                # refinement" downstream for the per-symbol sub-bin
                # correction.
                _sfd_fft = np.abs(_fft(seg * upchirp, n=N * 16))
                _sfd_peak = int(np.argmax(_sfd_fft))
                _sfd_fine_bin = _sfd_peak / 16.0
                if _sfd_fine_bin > N // 2:
                    _sfd_fine_bin -= N
                sfd_cfo_bins_cand.append(_sfd_fine_bin)

            # SFD CFO magnitude penalty: large SFD CFO usually means the
            # detected preamble bin is off (wrong cyclic shift).  BUT if the
            # SFD CFO is CONSISTENT with the preamble bin (signal really IS
            # far from baseband — captured well off-centre), it's a real
            # signal and shouldn't be penalised.  Test11/14 at SF12 had real
            # preambles at bin ~3018 (=-1078 signed = -33 kHz CFO); their
            # SFD CFO measurements also landed at ~-1078, so the penalty
            # fires HUGE (-3000) for a perfectly real signal and the
            # decoder picks a noise-peak candidate instead.  Only penalise
            # when SFD CFO disagrees with preamble bin (i.e., the detection
            # was inconsistent — likely a noise/spur preamble).  Tied scores
            # between gated/ungated FFT runs (parallel scipy FFT is
            # non-deterministic to ~10⁻⁶) flip candidate ordering — this
            # term is large enough to dominate those ties when it does fire.
            _sfd_cfo_cand_mean = (sum(sfd_cfo_bins_cand) / len(sfd_cfo_bins_cand)
                                  if sfd_cfo_bins_cand else 0.0)
            # SFD CFO penalty REMOVED 2026-05-27 — it was over-aggressive and
            # killed real off-baseband preambles at high SF.  Test11/Test14
            # source DMs at SF12/-33 kHz CFO were correctly detected by the
            # preamble but penalised by -3000+ (because their SFD CFO measured
            # at the same large value), causing the decoder to pick a noise-
            # peak candidate instead.  Without the penalty: all 4 known-failing
            # captures (Test3, Test11, Test14, Test19) decode at budget=10.
            # The wrong-cyclic-shift case the penalty was originally guarding
            # against is now caught by the skip_bins retry loop (try next
            # candidate when CRC fails on the chosen one).
            sfd_cfo_penalty = 0.0

            cand_score = (12.0 * min(pre_len_cand, 18) +
                          2.0 * run_mean_pmr +
                          18.0 * sfd_ok_cand +
                          sfd_metric +
                          sfd_cfo_penalty -
                          2.5 * max(0, 8 - pre_len_cand) -
                          0.5 * preamble_start_cand / N)  # prefer earlier preambles (msg before ACK)
            cand_infos.append((cand_score, pre_len_cand, run_mean_pmr, sfd_ok_cand,
                               sfd_metric, preamble_start_cand, preamble_bin_cand,
                               pre_last_i_cand, run))

    if not cand_infos:
        print("  NO PREAMBLE")
        return None

    cand_infos.sort(key=lambda x: x[0], reverse=True)

    # Filter out candidates whose bin is too close to a previously-failed bin
    if skip_bins:
        filtered = []
        for ci in cand_infos:
            ci_bin = ci[6]  # preamble_bin_cand
            ci_bin_signed = ci_bin if ci_bin <= N // 2 else ci_bin - N
            skip = False
            for sb in skip_bins:
                sb_signed = sb if sb <= N // 2 else sb - N
                if abs(ci_bin_signed - sb_signed) < 20:
                    skip = True
                    break
            if not skip:
                filtered.append(ci)
        if filtered:
            cand_infos = filtered
        # If all filtered out, use originals (better than nothing)
    best_score, best_pre_len, best_mean_pmr, best_sfd_ok, best_sfd_metric, \
        preamble_start, preamble_bin, pre_last_i, best_run = cand_infos[0]

    if len(cand_infos) > 1:
        print("  Chosen: bin=%d len=%d pmr=%.1fdB SFD=%d/2 score=%.1f start_sym=%d" % (
            preamble_bin, best_pre_len, best_mean_pmr, best_sfd_ok, best_score,
            preamble_start // N))


    # ---- CFO from preamble with linear regression for drift ----
    cfo_vals, cfo_indices = [], []
    for i in range(min(20, (len(iq1) - preamble_start) // N)):
        seg = iq1[preamble_start + i*N:preamble_start + (i+1)*N]
        if len(seg) < N: break
        fb = demod_fine(seg, downchirp, N)
        d = fb - preamble_bin
        if d > N/2: d -= N
        if d < -N/2: d += N
        if abs(d) < 2.0:
            cfo_vals.append(fb)
            cfo_indices.append(i)

    if not cfo_vals:
        print("  NO CFO"); return ('FAIL', preamble_start, 20 * N, preamble_bin)

    k_hat_fine = np.mean(cfo_vals)
    drift_bins_per_sym = 0.0
    if len(cfo_vals) >= 4:
        x = np.array(cfo_indices, dtype=np.float64)
        y = np.array(cfo_vals, dtype=np.float64)
        n = len(x)
        sx, sy, sxx, sxy = x.sum(), y.sum(), (x*x).sum(), (x*y).sum()
        denom = n * sxx - sx * sx
        if abs(denom) > 1e-10:
            drift_bins_per_sym = (n * sxy - sx * sy) / denom
            k_hat_fine = (sy - drift_bins_per_sym * sx) / n

    print("  Preamble bin: %d, fine: %.4f, drift: %.6f bins/sym" % (
        preamble_bin, k_hat_fine, drift_bins_per_sym))

    # ---- Per-packet hardware fingerprint (now that we know preamble_start) ----
    # Compute the precise transmit carrier using a TIGHT window around the
    # burst — preamble through end of expected payload — so the Welch centroid
    # isn't diluted by noise outside the burst.  ~10-15 LoRa symbols is
    # plenty.  Burst-window approach is robust to gate-export width: it
    # doesn't matter if the gate exported 200 ms or 5 s of context around
    # the burst — we always centroid over the actual signal.
    # Per-capture hardware fingerprint.  process_file's envelope+Welch already
    # set precise_carrier_hz (proven 138 Hz std, 269 Hz inter-device separation
    # at SF7); KEEP that and only ADD UMOP-specific features that envelope+Welch
    # doesn't compute.  Dechirp-based precise_carrier_hz is alias-prone at the
    # ±BW/2 boundary (SHORT_FAST), so never overwrite the working measurement.
    if _CURRENT_HW_FP[0] is None or _CURRENT_HW_FP[0].get('cfo_per_sym_std') is None:
        _txfp = _extract_tx_fingerprint(iq1, preamble_start, sf, bw)
        if _txfp is not None:
            if _CURRENT_HW_FP[0] is None:
                _CURRENT_HW_FP[0] = {}
            for _k in ('cfo_per_sym_std', 'am_ripple_pct',
                       'amp_per_sym_pct', 'phase_residual_rms', 'irr_db',
                       'pn_slope', 'signal_snr_db'):
                if _txfp.get(_k) is not None:
                    _CURRENT_HW_FP[0][_k] = _txfp[_k]
            # Only fill precise_carrier_hz from dechirp if envelope+Welch failed.
            if _CURRENT_HW_FP[0].get('precise_carrier_hz') is None and _txfp.get('precise_carrier_hz') is not None:
                _CURRENT_HW_FP[0]['precise_carrier_hz'] = _txfp['precise_carrier_hz']
    _hw_fp = _CURRENT_HW_FP[0]

    # Apply fractional CFO correction via complex multiply.
    # This removes the fractional-bin residual that integer bin subtraction
    # leaves behind, which smears FFT peaks and degrades soft LLR margins.
    cfo_hz = k_hat_fine * bw / N
    t_cfo = np.arange(len(iq1), dtype=np.float64) / bw
    iq1 = (iq1 * np.exp(-1j * 2.0 * np.pi * cfo_hz * t_cfo)).astype(np.complex64)

    # ---- Per-sample CFO DRIFT correction (quadratic phase) ----
    # The regression measures both intercept (k_hat_fine, applied above as
    # constant cfo_hz) AND slope (drift_bins_per_sym).  Apply the slope as a
    # quadratic phase correction with time origin at preamble_start (so the
    # regression's per-symbol-index mapping is preserved — using bare sample
    # index introduces a spurious cross-term).  At SF7 the payload is ~10
    # syms so cumulative drift is < 0.1 bins and this correction is a near-
    # no-op; at SF12 the payload is ~80 syms and cumulative drift can be 4+
    # bins, putting every payload symbol's FFT peak in the wrong bin.
    # SF/BW-agnostic.  Latency: one numpy multiply, negligible.
    if abs(drift_bins_per_sym) > 1e-4:
        _n_drift = np.arange(len(iq1), dtype=np.float64) - preamble_start
        _drift_phase = -np.pi * drift_bins_per_sym * (_n_drift / N) ** 2
        iq1 = (iq1 * np.exp(1j * _drift_phase)).astype(np.complex64)

    # Measure post-correction residual on preamble symbols.
    # The regression can be off by 0.1-0.2 bins, which causes ±1 bin errors
    # on ~15% of payload symbols — enough to fail CRC at CR=4/5.
    residuals = []
    for i in range(2, min(pre_last_i + 1, 20)):
        seg = iq1[preamble_start + i*N:preamble_start + (i+1)*N]
        if len(seg) < N: break
        fb = demod_fine(seg, downchirp, N)
        # After correction, preamble should be at bin ~0
        if fb > N/2: fb -= N
        if abs(fb) < 2.0:
            residuals.append(fb)
    preamble_residual_val = 0.0  # stored for per-symbol drift correction
    if residuals:
        residual = float(np.median(residuals))
        preamble_residual_val = residual
        if abs(residual) > 0.01:
            # Apply second correction to remove residual bias
            res_hz = residual * bw / N
            t_cfo2 = np.arange(len(iq1), dtype=np.float64) / bw
            iq1 = (iq1 * np.exp(-1j * 2.0 * np.pi * res_hz * t_cfo2)).astype(np.complex64)
            print("  Residual correction: %.4f bins (%.1f Hz)%s" % (
                residual, res_hz,
                " WARNING: large residual — CFO may be inaccurate" if abs(residual) > 0.15 else ""))

    # Four-cum: estimate residual sub-bin CFO via 1st-difference autocorrelation of
    # preamble dechirp FFT bin values. More precise than re-running regression on
    # short preambles; picks up sub-bin phase drift rate left behind by the median.
    # Precondition: must run AFTER the integer-bin regression correction above so
    # that preamble ≈ bin 0 and only the fractional residual remains.
    # (Ported from meshtastic-sniffer lora.c state_tick PREAMBLE_OK → HEADER, lines 848-867.)
    _preamble_fft_vals = []
    for _pi in range(max(2, pre_last_i - 6), pre_last_i + 1):
        _seg = iq1[preamble_start + _pi*N : preamble_start + (_pi+1)*N]
        if len(_seg) < N:
            break
        _fft_out = _fft(_seg * downchirp)
        _preamble_fft_vals.append(complex(_fft_out[0]))
    cfo_frac_residual = 0.0
    if len(_preamble_fft_vals) >= 2:
        try:
            _four_cum = sum(
                _preamble_fft_vals[_i] * _preamble_fft_vals[_i + 1].conjugate()
                for _i in range(len(_preamble_fft_vals) - 1)
            )
            if abs(_four_cum) > 1e-12:
                cfo_frac_residual = -float(np.angle(_four_cum)) / (2.0 * np.pi)
        except Exception:
            cfo_frac_residual = 0.0
    _CFO_FRAC_THRESH = 0.05  # bins; below this, regression uncertainty dominates correction
    if abs(cfo_frac_residual) > _CFO_FRAC_THRESH:
        _t_frac = np.arange(len(iq1), dtype=np.float64) / N
        iq1 = (iq1 * np.exp(-1j * 2.0 * np.pi * cfo_frac_residual * _t_frac)).astype(np.complex64)
        print("  CFO frac residual (four-cum): %.4f bins (%.1f Hz) — applied" % (
            cfo_frac_residual, cfo_frac_residual * bw / N))

    # After correction, preamble bin ≈ 0, so no bin subtraction needed
    cfo_shift = 0

    # ---- Deterministic data_start (matching analysis script) ----
    # LoRa packet structure after last preamble symbol:
    #   +1 sync1, +2 sync2, +3..+5.25 SFD (2.25 downchirps) → data
    data_start = preamble_start + int((pre_last_i + 5.25) * N)

    if _HARNESS_OUT:
        _harness_emit('preamble', preamble_start=int(preamble_start),
                      pre_last_i=int(pre_last_i), preamble_bin=int(preamble_bin),
                      data_start=int(data_start))


    n_data = (len(iq1) - data_start) // N

    if n_data < 8:
        print("  NOT ENOUGH DATA (pre_last_i=%d, data_start=%d, remaining=%d syms)" % (
            pre_last_i, data_start, n_data))
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    # Quick SFD quality check (informational only)
    sfd_i = pre_last_i + 3
    sfd_ok = 0
    sfd_ratios = []
    for si in range(2):
        p = preamble_start + (sfd_i + si) * N
        if p + N > len(iq1): break
        seg = iq1[p:p + N]
        uf = np.abs(_fft(seg * downchirp))
        df = np.abs(_fft(seg * upchirp))
        up_e = float(np.max(uf))**2
        dn_e = float(np.max(df))**2
        sfd_ratios.append(dn_e / (up_e + 1e-30))
        if dn_e > up_e * 0.5:
            sfd_ok += 1

    print("  Preamble: %d sym (last_i=%d), SFD check: %d/2" % (
        pre_last_i + 1, pre_last_i, sfd_ok))

    if sfd_ok == 0:
        print("  NO SFD — likely collision or partial capture")
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    # ---- SFD-based CFO refinement (ungated for investigation) ----
    # SFD downchirps are temporally closer to the data symbols than the
    # preamble.  When the preamble-derived four-cum CFO estimate isn't quite
    # right (e.g. CFO drifted between preamble and data, or the preamble
    # detection locked onto an aliased cyclic shift), SFD CFO catches the
    # residual.  Empirically on the Test4 hop1 capture in run_2.sc16 this is
    # the only correction that lets the primary decode succeed without chase.
    _sfd_cfo_bins = []
    for _si in range(2):
        _p = preamble_start + (pre_last_i + 3 + _si) * N
        if _p + N > len(iq1):
            break
        _seg = iq1[_p:_p + N]
        _fft_out = np.abs(_fft(_seg * upchirp, n=N * 16))
        _peak = int(np.argmax(_fft_out))
        _fine_bin = _peak / 16.0
        if _fine_bin > N // 2:
            _fine_bin -= N
        _sfd_cfo_bins.append(_fine_bin)
    _sfd_cfo_mean = 0.0
    _sfd_cfo_spread = 0.0
    if len(_sfd_cfo_bins) == 2:
        _sfd_cfo_mean = float(np.mean(_sfd_cfo_bins))
        _sfd_cfo_spread = abs(_sfd_cfo_bins[0] - _sfd_cfo_bins[1])
        import os
        if os.environ.get('LORA_SFD_DEBUG'):
            print(f"  [SFD DEBUG] CFO bins = {_sfd_cfo_bins}, spread={_sfd_cfo_spread:.3f}, mean={_sfd_cfo_mean:.4f}",
                  file=__import__('sys').stderr)
        # SFD CFO magnitude is a SCORE for preamble-detection correctness:
        #   small (< ~2 bins)  →  preamble bin was correct, residual is real CFO
        #                         drift that we should correct
        #   large (>= 2 bins)  →  preamble bin was wrong (off by N bins from
        #                         true TX), applying would shift iq1 by a wild
        #                         amount and break subsequent decode
        # Test4 hop1 in run_2.sc16 measures -0.5 bins on the correct preamble
        # candidate and decodes cleanly when applied.  Test1 hop0 measures
        # -50 bins on its preamble candidates (likely because the SFD pattern
        # is aliased relative to the detected preamble cyclic shift, not a
        # real CFO of 50 bins) — applying would break its previously-working
        # decode.  Gate by |SFD CFO| < 2 bins to reject the wild-magnitude
        # measurements while preserving real sub-bin refinements.
        # Apply only when the measurement is sane: small magnitude (preamble
        # was correctly detected) and consistent between the 2 SFD chirps.
        # Large magnitudes (>2 bins) indicate preamble misalignment; the
        # candidate-score penalty above should have already deselected such
        # preambles, but this is a final safeguard.
        if (0.05 < abs(_sfd_cfo_mean) < 2.0
                and _sfd_cfo_spread < 0.5):
            _t_frac = np.arange(len(iq1), dtype=np.float64) / N
            iq1 = (iq1 * np.exp(-1j * 2.0 * np.pi * _sfd_cfo_mean * _t_frac)).astype(np.complex64)
            print("  SFD CFO refinement: %+.4f bins (%.1f Hz) applied  spread=%.3f" % (
                _sfd_cfo_mean, _sfd_cfo_mean * bw / N, _sfd_cfo_spread))

    # ---- Sync word extraction (two upchirp symbols immediately after preamble) ----
    # Symbol pre_last_i+1 and pre_last_i+2 encode the sync word nibbles.
    # Formula: sync_word = (bin1//(N//16) << 4) | (bin2//(N//16))
    # Known: 0x2B=Meshtastic, 0x34=LoRaWAN, 0x12=MeshCore/Private, 0x0F=Meshtastic(alt)
    # Sync word is a HINT about likely protocol family, NEVER the identification.
    # 0x12 is the Semtech default ("public LoRa") used by MeshCore, RadioLib raw
    # transmits, SX126x-Arduino, sandeepmistry/arduino-LoRa, LoRaMesher, and most
    # custom user firmware — calling it "MeshCore" was a relic that mislabeled
    # every other 0x12 transmitter.  0x34 is the LoRaWAN convention but MeshCore
    # can be configured to use it.  The protocol PARSERS decide identity from
    # CONTENT; this map only labels the radio's chosen sync byte for diagnostics.
    _SYNC_BRANDS = {0x2B: 'Meshtastic', 0x0F: 'Meshtastic',
                    0x12: 'Public LoRa (default)',
                    0x34: 'Public LoRa (LoRaWAN convention)'}
    _sw1_p = preamble_start + (pre_last_i + 1) * N
    _sw2_p = preamble_start + (pre_last_i + 2) * N
    _sync_word = None
    if _sw1_p + N <= len(iq1) and _sw2_p + N <= len(iq1):
        _nib_scale = max(1, N // 16)
        _b1 = int(np.argmax(np.abs(_fft(iq1[_sw1_p:_sw1_p + N] * downchirp))))
        _b2 = int(np.argmax(np.abs(_fft(iq1[_sw2_p:_sw2_p + N] * downchirp))))
        _sync_word = ((_b1 // _nib_scale & 0xF) << 4) | (_b2 // _nib_scale & 0xF)
    _brand = _SYNC_BRANDS.get(_sync_word, f'Unknown(0x{_sync_word:02X})' if _sync_word is not None else 'Unknown')

    # Compute-saving early bail (before the expensive soft-decode): this sync can't
    # belong to any enabled protocol and unknown reporting is off → don't decode it.
    if _protocol_skip(_sync_word):
        print("  SKIP: sync 0x%02X not an enabled protocol (unknown off)"
              % (_sync_word if _sync_word is not None else 0))
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    # CFO already corrected via complex multiply above.
    # Soft demod sees clean FFT peaks with no fractional-bin smearing.

    # ---- Helper: get IQ segment for symbol ----
    def get_symbol(sym_idx):
        """Get IQ segment for data symbol sym_idx (CFO-corrected)."""
        p = data_start + sym_idx * N
        if p + N > len(iq1):
            return None
        return iq1[p:p + N]

    def decode_header_variant(data_start_var):
        bins, pmrs, fines, raws = [], [], [], []
        hdr_llrs = []
        max_syms = min(8, (len(iq1) - data_start_var) // N)
        for j in range(max_syms):
            p = data_start_var + j * N
            if p + N > len(iq1):
                break
            seg = iq1[p:p + N]
            fb = demod_fine(seg, downchirp, N)
            rawb = int(round(fb))
            b = (rawb - cfo_shift) % N
            bins.append(b)
            fines.append(float(fb - cfo_shift))
            raws.append(rawb)
            # Compute |FFT(seg*downchirp)| ONCE per symbol and reuse for
            # PMR + squared form for soft LLR — the soft path otherwise
            # repeats the same FFT internally.  Saves one full-N FFT per
            # header symbol per variant (at ~50 variants × 8 syms per
            # header decode, that's ~400 redundant FFT calls eliminated).
            # Match the original `np.abs(x)**2` rounding exactly: compute
            # uf = |x| first, then mag_sq = uf**2 — so the precomputed
            # mag_sq is bit-identical to what soft_fft_demod would compute
            # internally.
            uf = np.abs(_fft(seg * downchirp))
            _hdr_mag_sq = uf ** 2
            pmr = 10*np.log10(float(np.max(uf)) / (float(np.mean(uf)) + 1e-30) + 1e-15)
            pmrs.append(float(pmr))
            hdr_llrs.append(soft_fft_demod(seg, downchirp, N, N // 4, ppm, 4,
                                            cfo_shift=cfo_shift,
                                            _precomp_mag_sq=_hdr_mag_sq))

        if len(bins) < 8:
            return None
        # Soft path: LLR-based deinterleave + ML Hamming over all 16 candidates.
        # More robust than hard-decision path under IQ imbalance (mirror interference).
        hdr_soft_cw = soft_deinterleave(hdr_llrs, 8, ppm)
        _hb, _, _ = soft_hamming_decode_batch(hdr_soft_cw, 8)
        hdr_nibs_var = [int(_hb[k]) for k in range(ppm)]
        res = parse_header(hdr_nibs_var)
        return bins, pmrs, hdr_nibs_var, res, fines, raws

    # ---- HEADER DECODE (soft, 8 symbols, reduced rate, CR 4/8) ----
    base_data_start = data_start
    header_span = max(24, dec * 2)
    header_offsets = ordered_unique_offsets(
        [0] + list(range(-header_span, header_span + 1)) +
        [-(dec // 2), dec // 2, -dec, dec, -2 * dec, 2 * dec]
    )

    base_variant = decode_header_variant(base_data_start)
    if base_variant is None:
        print("  NOT ENOUGH HEADER SYMBOLS")
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    # If the base (td=0) variant is already a valid LoRa header, the sweep
    # below is wasted work — the selection logic at line ~5043 only uses
    # `valid_variants` when `base_hdr_valid` is False.  Skip the ~50-offset
    # sweep entirely on the common case where td=0 already decodes.  PERF
    # at SF11: removes ~50 * 8 = 400 per-symbol FFTs + the demod_fine /
    # PMR / soft_fft_demod work for the unused variants.  Provably-
    # equivalent: when base_hdr_valid, chosen_variant always = base_variant
    # regardless of `valid_variants` contents.
    _base_res = base_variant[3]
    _base_hdr_valid = False
    if _base_res is not None:
        _bpl, _bcr, _bcrc, _bok = _base_res
        _base_hdr_valid = bool(_bok and (4 <= _bpl <= 237) and (1 <= _bcr <= 4))

    valid_variants = []
    if not _base_hdr_valid:
        for td in header_offsets:
            v = decode_header_variant(base_data_start + td)
            if v is None:
                continue
            bins2, pmrs2, nibs2, res2, fines2, raws2 = v
            if res2 is None:
                continue
            pl2, cr2, crc2, ok2 = res2
            pmr_med2 = float(np.median(pmrs2)) if pmrs2 else 0.0
            plausible = (4 <= pl2 <= 237) and (1 <= cr2 <= 4)
            if ok2 and plausible:
                score = 1000.0 + 4.0 * pmr_med2 - 0.10 * abs(td)
                valid_variants.append((score, td, v))

    # Be conservative with header realignment.
    # A valid explicit-header checksum can occur at the wrong timing offset,
    # especially on short/collided captures. That creates false positives like
    # huge PL/CR changes or bogus no-CRC frames. Prefer the base timing when it
    # already yields a plausible header, and only fall back to alternate timing
    # when the base header itself is invalid.
    base_bins0, base_pmrs0, base_nibs0, base_result0, base_fines0, base_raws0 = base_variant
    base_hdr_valid = False
    if base_result0 is not None:
        bpl0, bcr0, bcrc0, bok0 = base_result0
        base_hdr_valid = bool(bok0 and (4 <= bpl0 <= 237) and (1 <= bcr0 <= 4))

    chosen_td = 0
    chosen_variant = base_variant
    if not base_hdr_valid and valid_variants:
        ranked_valid = []
        for score, td, v in valid_variants:
            vbins, vpmrs, vnibs, vres, vfines, vraws = v
            if vres is None:
                continue
            vpl, vcr, vcrc, vok = vres
            pmr_med_v = float(np.median(vpmrs)) if vpmrs else 0.0
            # Prefer CRC-protected headers and nearby timing offsets.
            adj = score - 0.40 * abs(td) - (20.0 if not vcrc else 0.0)
            ranked_valid.append((adj, abs(td), -pmr_med_v, td, v))
        if ranked_valid:
            ranked_valid.sort()
            _, _, _, chosen_td, chosen_variant = ranked_valid[-1]

    data_start = base_data_start + chosen_td
    n_data = (len(iq1) - data_start) // N

    hdr_bins, hdr_pmrs, hdr_nibs, result, hdr_fines, hdr_raws = chosen_variant
    if result is None:
        print("  HEADER PARSE FAILED")
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    payload_len, cr, crc_present, hdr_ok = result
    # Flag a header-OK attempt for the carrier rescue ONLY when the payload
    # length is plausible for a real (short) Meshtastic packet.  A garbage
    # decode at a wrong alignment can pass the few-bit HDR_CHK with an absurd
    # PL (e.g. 143/227) — those must NOT trigger the (expensive) rescue.
    if diag_out is not None and hdr_ok and 12 <= payload_len <= 64:
        diag_out['hdr_ok'] = True
    # Broader signal: a real LoRa header decoded with ANY plausible payload
    # length (the same 4..237 sanity range the decoder itself enforces below).
    # Used by process_file's outer-loop bail to reset the consecutive-no-header
    # counter — protects ALL protocols (LoRaWAN/MeshCore/Unknown with PL > 64),
    # unlike the narrower `hdr_ok` flag which is Meshtastic-specific.
    if diag_out is not None and hdr_ok and 4 <= payload_len <= 237:
        diag_out['header_decoded'] = True

    if _HARNESS_OUT:
        _harness_emit('header', chosen_td=int(chosen_td),
                      hdr_bins=[int(b) for b in hdr_bins],
                      payload_len=int(payload_len), cr=int(cr),
                      crc_present=bool(crc_present), hdr_ok=bool(hdr_ok),
                      data_start=int(data_start))

    for i, (b, fb, rawb, pmr) in enumerate(zip(hdr_bins, hdr_fines, hdr_raws, hdr_pmrs)):
        print("    hdr[%d]: bin %3d (fine %7.3f, raw %d) %.1fdB" % (
            i, b, fb, rawb, pmr))

    print("\n  Header: PL=%d CR=4/%d CRC=%s HDR_CHK=%s  nibs=%s" % (
        payload_len, cr + 4, "yes" if crc_present else "no",
        "OK" if hdr_ok else "FAIL", hdr_nibs))

    if chosen_td != 0 and hdr_ok:
        print(f"  HEADER ALIGN: timing={chosen_td:+d} samp pmr_med={float(np.median(hdr_pmrs)):.1f}dB")

    if not hdr_ok:
        pmr_med = float(np.median(hdr_pmrs)) if len(hdr_pmrs) else 0.0
        print("  *** HEADER CHECKSUM FAILED ***")
        if pmr_med < 8.0:
            print(f"  DIAG header: weak header pmr_med={pmr_med:.1f}dB (likely too weak / false lock)")
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    if payload_len < 4 or payload_len > 237:
        print("  *** INVALID PAYLOAD LENGTH ***"); return ('FAIL', preamble_start, 20 * N, preamble_bin)

    # ---- SOFT PAYLOAD DECODE ----
    rdd = cr + 4
    need_bytes = payload_len + (2 if crc_present else 0)
    sym_time_ms = (1 << sf) * 1000.0 / bw
    ppm_pay = (sf - 2) if sym_time_ms > 16.0 else sf
    pay_levels = N if ppm_pay == sf else (N // 4)
    pay_bin_group = 1 if ppm_pay == sf else 4

    print("\n  === SOFT PAYLOAD DECODE ===")
    print("  PL=%d CR=4/%d need=%d bytes, ppm_pay=%d, LDRO=%s" % (
        payload_len, cr + 4, need_bytes, ppm_pay,
        "yes" if sym_time_ms > 16.0 else "no"))

    # ---- Assemble + dewhiten helpers ----
    # Both pack/unpack at byte granularity; vectorised with numpy to remove the
    # pure-python loop overhead that dominated cProfile at ~130k calls/run.
    _WHITEN_NP = np.frombuffer(bytes(WHITENING_SEQ[:256]), dtype=np.uint8)

    def assemble(nibs_list):
        n_pairs = min(need_bytes, len(nibs_list) // 2)
        if n_pairs <= 0:
            return []
        arr = np.asarray(nibs_list[:2 * n_pairs], dtype=np.uint8)
        a = arr[0::2] & 0xF
        b = arr[1::2] & 0xF
        return (a | (b << 4)).tolist()

    def dewhiten(raw):
        if not raw:
            return list(raw)
        # Numpy XOR over the head (payload_len bytes); remaining bytes pass through.
        n_w = min(payload_len, len(raw))
        if n_w <= 0:
            return list(raw)
        head = np.asarray(raw[:n_w], dtype=np.uint8)
        # WHITENING_SEQ is a fixed 256-element pattern cycled by (i & 0xFF).
        head = head ^ _WHITEN_NP[np.arange(n_w) & 0xFF]
        return head.tolist() + list(raw[n_w:])

    # Estimate fractional timing offset from header symbol fine values.
    # A consistent fractional part across all symbols indicates a sub-sample
    # timing error at data_start: a τ-sample offset shifts dechirped peaks by
    # τ bins, splitting energy across adjacent FFT bins and degrading LLR margins.
    # Correction: multiply downchirp by exp(-j2πτt/N) before demodulation.
    _tau_z = np.mean(np.exp(1j * 2.0 * np.pi * np.array([f % 1.0 for f in hdr_fines])))
    if abs(_tau_z) > 0.1:
        _tau_raw = float(np.angle(_tau_z) / (2.0 * np.pi))
        if _tau_raw < 0: _tau_raw += 1.0
        if _tau_raw > 0.5: _tau_raw -= 1.0
        tau_frac = round(_tau_raw * 16) / 16   # round to 1/16-bin resolution
    else:
        tau_frac = 0.0
    if abs(tau_frac) > 0.03:
        print("  Fractional timing offset est: %+.4f bins" % tau_frac)

    # ---- Core soft decode function ----
    def soft_decode_payload(ds_offset, slope=0.0, frac_tau=0.0):
        """Soft decode with given data_start offset, optional per-symbol
        timing slope (samples / symbol), and optional fractional timing
        correction frac_tau (bins) applied as exp(-j2πτt/N) to the downchirp.
        Returns (raw_bytes, payload, soft_info, nibs)."""
        base = data_start + int(ds_offset)

        # Fractional timing correction: exp(-j2πτt/N) applied to the downchirp
        # moves the dechirped peak from bin s+τ back to bin s.
        if abs(frac_tau) > 1e-4:
            _t = np.arange(N, dtype=np.float64)
            dc = (downchirp * np.exp(-1j * 2.0 * np.pi * frac_tau * _t / N)).astype(np.complex64)
        else:
            dc = downchirp

        def sym_seg(sym_idx):
            adj = int(round(float(ds_offset) + float(slope) * sym_idx))
            p = data_start + sym_idx * N + adj
            if p < 0 or p + N > len(iq1):
                return None
            return iq1[p:p + N]

        trial_nibs = list(hdr_nibs[5:])
        trial_soft = []

        # Soft decode header overflow with adjusted params (batched FFT)
        hdr_llrs = []
        _hdr_segs = []
        for i in range(8):
            seg = sym_seg(i)
            if seg is None:
                break
            _hdr_segs.append(seg)
        if len(_hdr_segs) == 8:
            _H = soft_fft_demod_batch(np.stack(_hdr_segs), dc, N, N // 4, ppm, 4,
                                      cfo_shift=cfo_shift)
            hdr_llrs = [_H[i] for i in range(8)]
        if len(hdr_llrs) == 8:
            hdr_soft_cw = soft_deinterleave(hdr_llrs, 8, ppm)
            # Batched header-overflow Hamming decode — only k in [5..ppm)
            # values are needed, but batching all ppm rows is faster than
            # per-row scalar calls (the dominant per-call dispatch overhead
            # disappears in one numpy matmul).
            _b, _, _ = soft_hamming_decode_batch(hdr_soft_cw, 8)
            for k in range(5, ppm):
                if k - 5 < len(trial_nibs):
                    trial_nibs[k - 5] = int(_b[k])

        # Conservative upper bound for available symbols under slope walk.
        if slope == 0.0:
            nd = max(0, (len(iq1) - base) // N)
        else:
            nd = max(0, min(
                (len(iq1) - (data_start + int(round(ds_offset)))) // N,
                (len(iq1) - (data_start + int(round(ds_offset + slope * 64.0)))) // N
            ))

        sp = 8
        while sp + rdd <= nd and len(trial_nibs) // 2 < need_bytes:
            _pay_segs = []
            for i in range(rdd):
                si = sp + i
                seg = sym_seg(si)
                if seg is None:
                    break
                _pay_segs.append(seg)
            if len(_pay_segs) < rdd:
                break
            _P = soft_fft_demod_batch(np.stack(_pay_segs), dc, N, pay_levels,
                                      ppm_pay, pay_bin_group, cfo_shift=cfo_shift)
            pay_llrs = [_P[i] for i in range(rdd)]
            sp += rdd
            pay_cw = soft_deinterleave(pay_llrs, rdd, ppm_pay)
            # Vectorised batched Hamming decode over all ppm_pay codewords —
            # replaces a 16×rdd Python double loop per codeword (the dominant
            # tottime contributor before this change).
            _b, _s, _m = soft_hamming_decode_batch(pay_cw, rdd)
            for k in range(ppm_pay):
                nib_idx = len(trial_nibs)
                trial_nibs.append(int(_b[k]))
                trial_soft.append((nib_idx, int(_b[k]), int(_s[k]), float(_m[k])))

        trial_raw = dewhiten(assemble(trial_nibs))
        return trial_raw, trial_raw[:payload_len], trial_soft, trial_nibs

    # ---- Chase decoder function ----
    # CRC-16 has only 65536 codes.  Each chase trial flips a candidate
    # nibble pattern and checks CRC; with too many trials the probability
    # that one of them passes CRC by chance becomes non-negligible.
    # max_flips=2 × C(16,1)+C(16,2)=136 trials → 0.21% false-positive rate
    # per decode.  Across many sweep offsets, that easily produces a wrong
    # PacketID that reports "CRC OK".  Drop to max_flips=1 (16 trials,
    # 0.024 % FP), and additionally require each flipped nibble to have a
    # genuinely low confidence margin — we should only override the demod
    # for nibbles the demod was uncertain about, never a "confident" one
    # whose flip just happens to satisfy CRC.
    CHASE_K = 12              # top 12 least-confident nibbles
    CHASE_MAX_FLIPS = 2       # C(12,1)+C(12,2)=78 trials.  Combined with
                              # the broadcast-header gate (dst==0xFFFFFFFF)
                              # the false-positive rate is effectively
                              # zero.  Empirically max_flips=3 regressed
                              # hop coverage (the extra trials sometimes
                              # collapse onto an earlier/later
                              # transmission's bits in the same capture).

    def _is_plausible_mesh_header(tr):
        """True iff the first 16 bytes look like a BROADCAST Meshtastic header
        (dst == 0xFFFFFFFF).  Strict on purpose: used to gate the CHASE loop,
        where CRC-16 coincidences are the real false-positive risk."""
        if len(tr) < 16:
            return False
        return tr[0] == 0xFF and tr[1] == 0xFF and tr[2] == 0xFF and tr[3] == 0xFF

    def _is_plausible_mesh_header_uni(tr):
        """Broadcast OR a self-consistent UNICAST header (a real direct message).
        Used ONLY on the PRIMARY (no-chase) CRC path — a clean CRC-16 match on the
        nominal demod is high-confidence (≈1/65536 random), so it's safe to accept
        node-to-node packets here and surface them (encrypted, header-only) even
        though the payload won't decrypt with our channel key.  The chase loop keeps
        the strict broadcast-only gate above so chase coincidences stay suppressed."""
        if len(tr) < 16:
            return False
        dst = tr[0] | tr[1] << 8 | tr[2] << 16 | tr[3] << 24
        if dst == 0xFFFFFFFF:
            return True                                  # broadcast
        src = tr[4] | tr[5] << 8 | tr[6] << 16 | tr[7] << 24
        flags = tr[12]
        hop_limit = flags & 0x07
        hop_start = (flags >> 5) & 0x07
        # real, distinct endpoints + sane hop counts = a genuine unicast frame
        return (src not in (0, 0xFFFFFFFF) and dst not in (0, src)
                and hop_limit <= hop_start)

    def chase_decode(nibs, soft_info, max_flips=CHASE_MAX_FLIPS, k=CHASE_K):
        """Try flipping up to max_flips of the k least confident nibbles.
        Returns (ok, method, corrected_raw_bytes, flipped_list).

        Each CRC pass is gated by `_is_plausible_mesh_header` (dst==
        0xFFFFFFFF for broadcast).  Without that gate, chase finds CRC-16
        coincidence matches on random byte patterns at ~10⁻³ rate and
        emits confidently wrong PacketIDs — the failure mode that made
        non-Meshtastic flag values (hop_start=7 etc.) appear in the log
        from misdecoded captures."""
        if not soft_info:
            return False, '', None, []
        sorted_soft = sorted(soft_info, key=lambda x: x[3])
        K = min(k, len(sorted_soft))
        # Cap chase candidate margin: a nibble with high soft margin (high
        # demod confidence) almost certainly was demoded correctly.  Flipping
        # such nibbles to "fix" a CRC match is a false-positive risk —
        # empirically on the live test set this caused hop0/hop1 misclassifi-
        # cation (flag byte at nibble 24/25 has a few-bin error → chase
        # flips a different nibble that happens to satisfy CRC → wrong
        # hops_taken reported).  Only flip nibbles whose margin is below
        # `MARGIN_CAP`; values above are trusted.
        #
        # Margins from `soft_hamming_decode` are UN-normalized LLR differences
        # (sums of bin-bin LLRs), so the absolute scale depends on per-capture
        # FFT-magnitude levels — a fixed 0-1 cap rejects every trial.  Cap is
        # relative: a nibble is "uncertain" iff its margin is below
        # MARGIN_RATIO × the median margin of the whole payload.  ~50 % of
        # median picks out the genuinely low-confidence nibbles regardless
        # of absolute LLR scale, which is what the rank-K cap (K=12) is
        # already filtering.  The combination of K=12 + relative cap +
        # broadcast-header plausibility gate keeps the false-positive rate
        # comparable to the original 0-1 cap that worked on normalized
        # margins.
        _all_margins = [s[3] for s in soft_info]
        _med_margin = float(np.median(_all_margins)) if _all_margins else 1.0
        MARGIN_RATIO = 0.5
        MARGIN_CAP_ABS = _med_margin * MARGIN_RATIO
        for n_flips in range(1, max_flips + 1):
            for positions in combinations(range(K), n_flips):
                # Bail if any chosen position's margin is above the cap.
                if any(sorted_soft[b][3] > MARGIN_CAP_ABS for b in positions):
                    continue
                trial = list(nibs)
                for b in positions:
                    trial[sorted_soft[b][0]] = sorted_soft[b][2]
                tr = dewhiten(assemble(trial))
                if len(tr) >= payload_len + 2:
                    ok, method = check_crc(tr, payload_len, strict=True)
                    if ok and _is_plausible_mesh_header(tr):
                        flipped = [(sorted_soft[b][0], sorted_soft[b][1],
                                    sorted_soft[b][2], sorted_soft[b][3])
                                   for b in positions]
                        return True, method, tr, flipped
        return False, '', None, []

    # ---- Primary decode at nominal parameters ----
    raw_bytes, payload, pay_soft_info, decoded_nibs = soft_decode_payload(0, 0.0, frac_tau=tau_frac)
    print("  Decoded %d nibbles → %d bytes (need %d)" % (
        len(decoded_nibs), len(raw_bytes), need_bytes))
    print("  Payload hex: %s" % ''.join('%02x' % b for b in payload))

    crc_ok, crc_method = False, ""
    _clean_crc = False   # True only if the whole-packet CRC matched WITHOUT chase
    if crc_present and len(raw_bytes) >= payload_len + 2:
        crc_ok, crc_method = check_crc(raw_bytes, payload_len)
        _clean_crc = crc_ok
        print("  CRC: %s %s" % (crc_method, "OK" if crc_ok else ""))

    if _HARNESS_OUT:
        _margins = [float(s[3]) for s in pay_soft_info] if pay_soft_info else []
        _harness_emit('primary_decode',
                      nibs_count=int(len(decoded_nibs)),
                      first_nibs=[int(n) for n in decoded_nibs[:32]],
                      raw_hex=''.join('%02x' % b for b in raw_bytes[:32]),
                      crc_ok=bool(crc_ok),
                      tau_frac=float(tau_frac),
                      margin_med=float(np.median(_margins)) if _margins else 0.0,
                      margin_min=float(min(_margins)) if _margins else 0.0,
                      margin_n=int(len(_margins)))

    # ---- Parser-first early dispatch ----
    # Try every enabled protocol parser on a clean primary CRC BEFORE the
    # legacy Meshtastic-shaped FP gates run.  Each parser owns its own
    # structural+content validation (Meshtastic: shape+behavioral preconditions
    # + protobuf decrypt validity; MeshCore: payload_type∈valid_set ∧ version=0
    # + path structure; LoRaWAN: MHDR + grid).  A successful parse short-circuits
    # the chase+sweep recovery entirely — those exist to recover MARGINAL
    # Meshtastic, not to second-guess a clean non-Meshtastic decode.  This is
    # both a latency win (chase costs 0.2 s SF7 → seconds at high SF) and a
    # correctness fix (today MeshCore/LoRaWAN frames slip through only when
    # their bytes happen to satisfy a Meshtastic-shaped gate; they get dropped
    # otherwise).
    #
    # FP risk preserved: the Meshtastic shape+behavioral preconditions are the
    # same gates used today — moved from "force chase" semantics to "parser
    # precondition".  Non-Meshtastic parsers each have their own structural
    # validators (already in code; not added here).  Unknown=ON only surfaces
    # bytes that NO parser claimed AND that are not Meshtastic-shaped — so the
    # FP exposure is ~1/65536 × (1 - Meshtastic-shape rate) per CRC-checked
    # packet, opt-in.
    _early_handled = False
    _freq_mhz, _rssi = None, None
    for _p in name.lstrip('.').split('_'):
        if _p.endswith('MHz'):
            try: _freq_mhz = float(_p[:-3])
            except ValueError: pass
        elif _p.startswith('pwr'):
            try: _rssi = int(_p[3:])
            except ValueError: pass
    _rf = {'sf': sf, 'bw': bw, 'freq_mhz': _freq_mhz, 'rssi': _rssi}
    # Attach per-packet hardware fingerprint + CFO/drift (already measured
    # during preamble fit above).  The web layer reads these to cluster
    # packets by transmitter device within a protocol family.  Precise CFO
    # comes from the regression intercept (k_hat_fine × bw / N, in Hz).
    if _hw_fp is not None:
        _rf['hw_fp'] = _hw_fp
    try:
        _rf['cfo_hz'] = float(cfo_hz)
    except (NameError, TypeError):
        pass
    try:
        _rf['cfo_drift'] = float(drift_bins_per_sym)
    except (NameError, TypeError):
        pass

    if crc_ok:
        _early_ctx = {'aes_key': _mesh_aes_key, 'no_key': _mesh_no_key,
                      'clean_crc': True, 'rf': _rf, 'is_mesh': True}
        # Meshtastic: precondition = shape + behavioral (same as the legacy
        # primary gates).  Parser owns decrypt/protobuf validity downstream.
        if ('meshtastic' in _PROTO_ENABLED and len(raw_bytes) >= 16
                and _is_plausible_mesh_header_uni(raw_bytes)
                and _is_chase_acceptable_uni(raw_bytes)):
            _h_meshtastic(payload, _early_ctx)
            _early_handled = True
        # Non-Meshtastic parsers (MeshCore, LoRaWAN, LoRaMesher, ...).  Try-all-
        # rank: run EVERY enabled parser, collect every match, then pick the
        # highest-confidence one to emit.  This surfaces ambiguity instead of
        # hiding it — if two parsers both claim the bytes at the same tier the
        # winner is marked `ambiguous: true` and the runners-up are listed in
        # `alternatives` so the UI can show "looked like X, also matched Y."
        # Latency cost: a few microseconds per packet (parsers fail-fast on the
        # first-byte mismatch), traded against the correctness win of never
        # silently mis-attributing a frame that two protocols' validators both
        # accepted.
        if not _early_handled:
            _matches = []
            for _pname, _handler in _proto_candidates(_sync_word):
                if _pname not in _PROTO_ENABLED:
                    continue
                try:
                    _rec = _handler(payload, _early_ctx)
                except Exception:
                    continue
                if _rec is not None:
                    _matches.append((_pname, _rec))
            if _matches:
                # Rank by confidence tier (higher = stronger).
                _TIER_RANK = {'verified': 3, 'confirmed': 2, 'candidate': 1}
                def _tier(m): return _TIER_RANK.get(m[1].get('confidence', 'candidate'), 0)
                _matches.sort(key=lambda m: -_tier(m))
                _winner_pname, _winner = _matches[0]
                if len(_matches) > 1:
                    _alts = [{'proto': r.get('proto', n),
                              'hint':  r.get('hint'),
                              'confidence': r.get('confidence'),
                              'summary': r.get('summary')} for n, r in _matches[1:]]
                    _winner['alternatives'] = _alts
                    # Same-tier runner-up → genuine ambiguity, flag it.
                    if _tier(_matches[1]) == _tier(_matches[0]):
                        _winner['ambiguous'] = True
                        print("  AMBIGUOUS: %d parsers claimed at tier '%s' — "
                              "emitting %s, listing alternatives" % (
                                  len(_matches), _winner.get('confidence', 'candidate'),
                                  _winner.get('proto', _winner_pname)))
                _emit_proto(_winner, payload, sf, bw, cr, _sync_word, _rf)
                _early_handled = True
        # Unknown=ON: surface non-Meshtastic-shape bytes with no parser match.
        # (Meshtastic-shape unclaimed bytes fall through to chase for recovery,
        # matching today's "force chase, drop on fail" semantics.)
        if (not _early_handled and _PROTO_UNKNOWN
                and not _is_plausible_mesh_header_uni(raw_bytes)):
            print("  UNKNOWN-PROTO: CRC OK, no parser matched, non-Meshtastic "
                  "shape — surfacing as 'unknown'")
            _report_unknown(payload, sf, bw, cr, _sync_word, name)
            _emit_proto({'proto': 'unknown', 'from': None, 'to': None, 'decrypted': False,
                         'summary': '%d bytes · sync %s' % (
                             len(payload), ('0x%02x' % _sync_word) if _sync_word is not None else '?')},
                        payload, sf, bw, cr, _sync_word, _rf)
            _early_handled = True

    # If a parser claimed it (or it was surfaced as unknown), skip the legacy
    # gates+chase+sweep — they have nothing to add to an already-clean decode —
    # and return success with the same packet-length info as the normal path.
    if _early_handled:
        n_pay_syms = int(np.ceil(need_bytes * 8 / sf)) * (cr + 4)
        total_syms = 8 + 4.25 + 8 + n_pay_syms
        pkt_len_samples = int(total_syms * N) + N
        print()
        if _HARNESS_OUT:
            _harness_emit('attempt_end', status='OK', via='early_dispatch',
                          payload_hex=''.join('%02x' % b for b in payload),
                          payload_len=int(len(payload)), crc_ok=bool(crc_ok))
        return ('OK', preamble_start, pkt_len_samples, preamble_bin)

    # ---- Plausibility gate on primary CRC OK ----
    # CRC-16 false-positives happen when multiple symbols are mis-demoded in
    # a self-consistent way (the same demod error propagates through both
    # data and the trailing CRC bytes, so CRC matches even though the
    # underlying bytes are wrong).  Accept a clean primary CRC if the header is
    # a valid BROADCAST or self-consistent UNICAST frame; otherwise force chase.
    # Unicast (direct messages) are now surfaced — their payload won't decrypt
    # with our channel key (PKI/another key), so they're emitted header-only
    # (encrypted) downstream so you can see node↔node traffic exists. The chase
    # loop still uses the strict broadcast-only gate (its coincidences are the
    # real FP risk); only the high-confidence no-chase path admits unicast.
    if crc_ok and not _is_plausible_mesh_header_uni(raw_bytes):
        print("  PLAUSIBILITY: CRC OK but header not a valid broadcast/unicast frame "
              "— likely demod error, retry via chase")
        crc_ok = False  # force chase to try recovery
        _clean_crc = False
    # NO behavioral gate on PRIMARY CRC-OK decodes — accept any structurally
    # valid CRC-OK unicast frame regardless of whether endpoints are previously
    # known.  Trade-off: ~1 false positive per ~150 captures from random CRC
    # coincidences vs being able to gather information from previously-unseen
    # nodes (essential for arbitrary-LoRa intercept where we don't know the
    # network in advance).  The chase loop below still has its own gate to
    # prevent chase-induced false positives (those are the higher-risk path).

    # ---- Chase on primary decode ----
    # The chase loop has its own broadcast-header plausibility gate
    # (dst==0xFFFFFFFF), so we no longer need to gate on sync_word.  This
    # matters because relays sometimes show sync 0x2A (1-bit off from
    # 0x2B) due to demod errors at the sync symbols; if we refuse to chase
    # those captures we lose the relay (hop1) decode for them.
    if crc_present and not crc_ok:
        ok, method, tr, flipped = chase_decode(decoded_nibs, pay_soft_info)
        if _HARNESS_OUT:
            _harness_emit('chase', used=True, ok=bool(ok), method=str(method or ''),
                          n_flipped=int(len(flipped)),
                          flips=[(int(i), int(b), int(s), float(m)) for i, b, s, m in flipped])
        if ok:
            crc_ok, crc_method = True, method
            raw_bytes = tr
            payload = tr[:payload_len]
            print("  CHASE: CRC %s OK! Flipped %d nibble(s):" % (method, len(flipped)))
            for idx, best, second, margin in flipped:
                print("    nib[%d]: 0x%x → 0x%x (margin=%.1f)" % (idx, best, second, margin))

    # ---- Skip sweep on confident demod that failed CRC (perf, esp. low-core SF12) ----
    # The sweep block below burns ~1-2 s exploring timing offsets, which only
    # helps when bit errors were caused by timing (low demod margins).  When
    # the demod was CONFIDENT (high median + no near-zero outliers) AND chase
    # — which already ran above with up to 2 nibble flips — couldn't fix the
    # CRC, the bits are correct as-is and the CRC fail is structural (encrypted
    # payload, wrong PL/CR header byte interpretation, or a different protocol
    # pretending to look Meshtastic-shaped).  Sweep timing tweaks won't help.
    #
    # Live SF12 4-core: 25/25 sweeps found NO CRC match on confident-demod
    # captures, wasting ~25 × 1 s of decode budget.  Skipping recovers that
    # time for actual queue drain without losing any real recoveries.
    _skip_sweep = False
    if crc_present and not crc_ok and pay_soft_info and len(raw_bytes) >= 16:
        _med_m = float(np.median([s[3] for s in pay_soft_info]))
        _min_m = float(min(s[3] for s in pay_soft_info))
        # Confident demod: median margin well above noise, no near-zero outliers.
        # Chase decoder (above) already tried bit-flipping the lowest-margin nibbles —
        # if it didn't find a CRC match, sweep timing tweaks won't either.
        if _med_m > 50000.0 and _min_m > 2000.0:
            print("  SKIP SWEEP: confident demod (med=%.0f min=%.0f) — "
                  "chase failed, sweep timing tweaks won't recover bits" % (_med_m, _min_m))
            _skip_sweep = True
            # Also signal PASS 2 (CFO-recenter sweep) to skip this capture:
            # for a CONFIDENT demod that failed CRC, no CFO shift will help
            # either — the bits are correct, the failure is structural.
            if diag_out is not None:
                diag_out['no_rescue'] = True

    # ---- Combined timing sweep with chase ----
    if crc_present and not crc_ok and not _skip_sweep:
        print("\n  === SWEEP (timing × chase) ===")
        # Limited sweep: ~20 timing offsets × decode-only + chase(1 flip)
        # More offsets × more flips = more false CRC-16 matches
        ds_offsets = ordered_unique_offsets(
            [0] + list(range(-8, 9)) + [-(dec // 2), dec // 2, -dec, dec]
        )

        sweep_start = time.time()
        # Sweep is the last-resort recovery path for the rare LoRa packet
        # whose tau-frac estimate from the preamble doesn't quite align the
        # payload symbols.  2 s is a balance: long enough to recover real
        # marginal LoRa captures (run_4 saw 11 misses at 0.5 s vs 0 at 8 s)
        # but short enough that non-LoRa fail-fast and don't block the queue.
        # Speculative early-recenter passes set a shorter budget (the correct
        # CFO decodes in the first sweep step) so a wrong CFO candidate bails
        # fast; the thorough full-budget PASS 2 remains as the fallback.
        SWEEP_BUDGET = sweep_budget
        sweep_count = 0
        best_candidates = []

        def record_candidate(label, ds_off, slope, frac_tau, tr_raw, tr_soft):
            if len(tr_raw) < payload_len + 2:
                return
            margin_score = 0.0
            if tr_soft:
                vals = sorted(float(m) for (_, _, _, m) in tr_soft)
                take = vals[:min(8, len(vals))]
                margin_score = float(sum(take)) / max(1, len(take))
            best_candidates.append((margin_score, label, ds_off, slope, frac_tau, ''.join('%02x' % b for b in tr_raw[:min(payload_len, 8)])))

        def try_candidate(label, ds_off, slope, frac_tau=0.0):
            nonlocal crc_ok, crc_method, _clean_crc, raw_bytes, payload, sweep_count
            if crc_ok or time.time() - sweep_start > SWEEP_BUDGET:
                return True
            tr_raw, tr_pay, tr_soft, tr_nibs = soft_decode_payload(ds_off, slope, frac_tau=frac_tau)
            sweep_count += 1
            if len(tr_raw) >= payload_len + 2:
                ok, method = check_crc(tr_raw, payload_len, strict=True)
                # Sweep CRC matches without a header gate produce ~1/65536 false
                # positives per timing attempt; over the sweep's ~20 offsets this
                # leaks a spurious "unknown" (or worse, a confidently-wrong
                # Meshtastic packet if the coincidence bytes also happen to look
                # like a plausible header).  Gate with the same uni-aware header
                # check the primary CRC path uses (line ~3671) so the sweep is
                # consistent: a real packet recovered at the right timing has a
                # valid header (CRC validates the header bytes), so this NEVER
                # rejects a real packet; only coincidences on garbage are blocked.
                # Behavioral gate: broadcast OR (structural unicast AND a
                # known node we've already decoded clean traffic from).
                # _is_chase_acceptable_uni handles both: dst=FFFFFFFF always
                # passes; unicast requires src or dst ∈ _KNOWN_NODES.  Without
                # this, sweep CRC-16 coincidence × structural-uni-plausibility
                # leaks ~1 FP per ~150 captures with from/to addresses that
                # are bit-flipped Node addresses.  _KNOWN_NODES auto-populates
                # from clean Meshtastic decrypts.
                if ok and _is_chase_acceptable_uni(tr_raw):
                    print("  %s ds=%+d slope=%+.3f tau=%+.4f: CRC %s OK!" % (label, ds_off, slope, frac_tau, method))
                    crc_ok, crc_method = True, method
                    # Sweep DIRECT CRC + plausibility gate = effectively a "clean"
                    # CRC at a different timing offset (no bit-flipping involved).
                    # Marking _clean_crc=True lets the downstream encrypted-DM
                    # emit (line ~1634) surface PKI-encrypted DMs whose preamble
                    # was off-timing enough that the primary failed but the
                    # sweep recovered.  Confirmed root cause of DM-handshake-gap
                    # misses 2026-05-26: those DMs decode cleanly via sweep but
                    # were suppressed at the encrypted-only emit because
                    # _clean_crc was only set in the primary path.  False-
                    # positive risk: sweep CRC-16 coincidence (1/65536) × plausible-
                    # uni-header structural (~1/2^32) per attempt = ~1/2^48 — negligible.
                    _clean_crc = True
                    raw_bytes, payload = tr_raw, tr_pay
                    return True
                # Chase with max 1 flip in sweep to limit false positives
                ok, method, tr2, flipped = chase_decode(tr_nibs, tr_soft,
                                                         max_flips=1, k=CHASE_K)
                if ok:
                    print("  %s ds=%+d slope=%+.3f tau=%+.4f + chase(%d flips): CRC %s OK!" % (
                        label, ds_off, slope, frac_tau, len(flipped), method))
                    crc_ok, crc_method = True, method
                    raw_bytes = tr2
                    payload = tr2[:payload_len]
                    return True
                record_candidate(label, ds_off, slope, frac_tau, tr_raw, tr_soft)
            return False

        # Two-dimensional timing search: integer ds_off (sample shift) AND
        # fractional tau (sub-sample within a single sample period).  The
        # preamble-derived tau estimate is accurate to ~0.1 bins but can be
        # off by 0.25-0.5 bins when payload timing drifts from preamble
        # timing.  An off-by-0.25 tau splits every symbol's FFT peak between
        # two adjacent bins → ~half the nibbles end up at the wrong bin →
        # CRC fails even though the header (which uses its own header-only
        # tau correction) decoded.  SF/BW-agnostic: tau is bin-units, scales
        # with bw and N automatically.
        TAU_DELTAS = [0.0, -0.125, 0.125, -0.25, 0.25, -0.375, 0.375, -0.5, 0.5]
        for tau_d in TAU_DELTAS:
            if crc_ok: break
            frac_tau_try = tau_frac + tau_d
            # Wrap into ±0.5 range (tau is a circular bin offset)
            while frac_tau_try > 0.5: frac_tau_try -= 1.0
            while frac_tau_try < -0.5: frac_tau_try += 1.0
            for ds_off in ds_offsets:
                if try_candidate('const', ds_off, 0.0, frac_tau=frac_tau_try):
                    break

        # If preamble regression measured clock drift, try applying it.
        # drift_bins_per_sym (from CFO regression) maps to a per-symbol sample
        # offset: slope_samps = drift_bins_per_sym * N.  Try the measured value
        # and scaled versions — clock drift can be underestimated from the short
        # preamble window compared to full packet duration.
        if not crc_ok and abs(drift_bins_per_sym) > 1e-4:
            drift_slope = drift_bins_per_sym * N  # convert bins/sym → samples/sym
            for scale in [1.0, 0.5, 1.5, 2.0, -1.0]:
                slope_try = drift_slope * scale
                if abs(slope_try) < 0.05:
                    continue
                for ds_off in [0, -(dec // 2), dec // 2]:
                    if try_candidate('drift', ds_off, slope_try, frac_tau=tau_frac):
                        break
                if crc_ok:
                    break

        sweep_elapsed = time.time() - sweep_start
        if not crc_ok:
            print("  sweep: %d combos in %.1fs — no CRC match" % (sweep_count, sweep_elapsed))
            if best_candidates:
                best_candidates.sort(key=lambda x: x[0])
                print("  DIAG sweep: lowest-margin candidates:")
                for margin_score, label, ds_off, slope, frac_tau, hx in best_candidates[:5]:
                    print("    %s ds=%+d slope=%+.3f tau=%+.4f worst8avg=%.2f head=%s" % (
                        label, ds_off, slope, frac_tau, margin_score, hx))

    # ---- Final result ----
    print("\n  === RESULT ===")
    payload = raw_bytes[:payload_len]
    print("  Payload (%d bytes): %s" % (payload_len, ''.join('%02x' % b for b in payload)))
    ascii_str = ''.join(chr(b) if 0x20 <= b < 0x7f else '.' for b in payload)
    print("  ASCII: \"%s\"" % ascii_str)
    if crc_present:
        print("  CRC: %s %s" % ("OK" if crc_ok else "FAIL", crc_method))
        if (not crc_ok) and pay_soft_info:
            try:
                ms = np.array([m for (_, _, _, m) in pay_soft_info], dtype=np.float64)
                ms_sorted = np.sort(ms)
                p10 = float(ms_sorted[int(0.1 * (len(ms_sorted) - 1))]) if len(ms_sorted) else 0.0
                med = float(np.median(ms_sorted)) if len(ms_sorted) else 0.0
                mn = float(ms_sorted[0]) if len(ms_sorted) else 0.0
                print("  DIAG crc: margins min/p10/med=%.2f/%.2f/%.2f (higher is better)" % (mn, p10, med))
                worst = sorted(pay_soft_info, key=lambda x: x[3])[:12]
                if worst:
                    s = ", ".join([f"{idx}:{best:x}->{second:x}({margin:.2f})" for (idx, best, second, margin) in worst])
                    print("  DIAG crc: worst nibbles (idx best->second margin): " + s)
            except Exception:
                pass
    else:
        print("  CRC: not present")
    _is_meshtastic = ('meshtastic' in _PROTO_ENABLED) and (_sync_word in (0x2B, 0x0F, None))
    # The 2-symbol sync-word demod is NOT reliable: it mis-reads across spreading
    # factors AND on off-carrier captures (a chirp re-detected ±tens of kHz off the
    # true carrier reads e.g. 0x2A/0x1A/0xBE instead of 0x2B — the cause of DMs
    # leaking out as "unknown").  So do NOT route on the sync byte.  When CRC passed
    # and the header is a plausible Meshtastic frame — BROADCAST or a self-consistent
    # UNICAST (direct-message) header — accept it regardless of sync.  Safe because
    # parse_meshtastic_packet's decrypt-validity / clean-CRC gate rejects anything
    # that doesn't decrypt to a valid protobuf (broadcast) or emits a DM only as
    # encrypted-header-only (corroborated by a known node downstream).  Unicast
    # routing lets off-carrier/mis-sync DM captures decode as Meshtastic (deduped by
    # PacketID) instead of "unknown" — keeping the gate's decode diversity, no loss.
    if (not _is_meshtastic and crc_ok and _is_plausible_mesh_header_uni(payload)):
        _is_meshtastic = True
    if (not crc_ok) and _is_meshtastic and len(payload) >= 8:
        print("  Meshtastic hdr guess: dst=%s src=%s" % (
            ''.join('%02x' % payload[3-i] for i in range(4)),
            ''.join('%02x' % payload[7-i] for i in range(4))))
    if _sync_word is not None:
        print("  Protocol: %s (sync=0x%02X)" % (_brand, _sync_word))
    # CRC gate: if CRC was present but failed after all retries, suppress AES decryption.
    # Decrypting a corrupt payload produces garbage protobuf output and wastes CPU.
    # (Explicit gate ported from meshtastic-sniffer mesh_packet.c:payload_crc_ok check.)
    if crc_present and not crc_ok:
        print("  [CRC GATE] AES decrypt suppressed — payload CRC failed after all retries")
        print()
        return ('FAIL', preamble_start, 20 * N, preamble_bin)

    # CRC passed (or not present) — route through the protocol parser registry.
    # (_rf was hoisted to the early-dispatch block above; reuse it here.)
    _ctx = {'aes_key': _mesh_aes_key, 'no_key': _mesh_no_key,
            'clean_crc': _clean_crc, 'rf': _rf, 'is_mesh': _is_meshtastic}
    _handled = False
    if _ctx['is_mesh'] and len(payload) >= 16:
        _h_meshtastic(payload, _ctx)              # validates + emits its own [PKT]
        _handled = True
    else:
        # Try-all-rank dispatch (same as the pre-chase early-dispatch above).
        # Sync word only orders the candidates for deterministic tiebreaks;
        # the content validators decide and the highest confidence tier wins,
        # with same-tier ties flagged as 'ambiguous'.
        _matches2 = []
        for _pname, _handler in _proto_candidates(_sync_word):
            if _pname not in _PROTO_ENABLED:
                continue
            if len(payload) < 1:
                break
            try:
                _rec = _handler(payload, _ctx)
            except Exception:
                continue
            if _rec is not None:
                _matches2.append((_pname, _rec))
        if _matches2:
            _TIER_RANK = {'verified': 3, 'confirmed': 2, 'candidate': 1}
            def _tier(m): return _TIER_RANK.get(m[1].get('confidence', 'candidate'), 0)
            _matches2.sort(key=lambda m: -_tier(m))
            _winner = _matches2[0][1]
            if len(_matches2) > 1:
                _winner['alternatives'] = [
                    {'proto': r.get('proto', n), 'hint': r.get('hint'),
                     'confidence': r.get('confidence'), 'summary': r.get('summary')}
                    for n, r in _matches2[1:]]
                if _tier(_matches2[1]) == _tier(_matches2[0]):
                    _winner['ambiguous'] = True
            _emit_proto(_winner, payload, sf, bw, cr, _sync_word, _rf)
            _handled = True
    if not _handled and crc_ok and _PROTO_UNKNOWN:
        # Good CRC but no parser recognised it: a real frame of a protocol we
        # don't implement.  Surface it generically in the UI AND log it for the
        # developer report (so unimplemented protocols can be added over time).
        # Only when the user has turned Unknown reporting ON.
        _report_unknown(payload, sf, bw, cr, _sync_word, name)
        _emit_proto({'proto': 'unknown', 'from': None, 'to': None, 'decrypted': False,
                     'summary': '%d bytes · sync %s' % (
                         len(payload), ('0x%02x' % _sync_word) if _sync_word is not None else '?')},
                    payload, sf, bw, cr, _sync_word, _rf)

    print()
    # Compute the packet's total length so the caller can mask it from
    # iq and re-search for additional concurrent packets.
    #  - preamble (8 syms)
    #  - sync words (2 syms) + downchirps (2.25 syms) ≈ 4.25 from preamble end
    #  - header (8 syms at SF7-ppm reduced rate)
    #  - payload syms ≈ ceil(need_bytes * 8 / sf) * (cr + 4)
    n_pay_syms = int(np.ceil(need_bytes * 8 / sf)) * (cr + 4)
    total_syms = 8 + 4.25 + 8 + n_pay_syms
    pkt_len_samples = int(total_syms * N) + N  # +1 sym safety
    if _HARNESS_OUT:
        _harness_emit('attempt_end', status='OK', via='full_path',
                      payload_hex=''.join('%02x' % b for b in payload),
                      payload_len=int(len(payload)), crc_ok=bool(crc_ok))
    return ('OK', preamble_start, pkt_len_samples, preamble_bin)

# Module-level key config (set from command line)
_mesh_aes_key = MESH_AES_KEY
_mesh_no_key = False


if __name__ == '__main__':
    import base64
    args = sys.argv[1:]

    # Parse options: -k KEY, --pki-key HEX32, --pki-peer NODENUM:HEX32PUB
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == '-k' and i + 1 < len(args):
            keyarg = args[i+1]
            if keyarg.upper() in ('NOKEY', 'HAM', 'NONE', '0'):
                _mesh_no_key = True
                _mesh_aes_key = bytes(16)
            elif len(keyarg) == 32 and all(c in '0123456789abcdefABCDEF' for c in keyarg):
                _mesh_aes_key = bytes.fromhex(keyarg)
            else:
                try:
                    kb = base64.b64decode(keyarg)
                    if len(kb) in (16, 32):
                        _mesh_aes_key = kb[:16]
                    else:
                        print("WARNING: Invalid key (got %d bytes, need 16 or 32). Using default." % len(kb))
                except Exception:
                    print("WARNING: Could not decode key. Using default.")
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1

    if filtered_args:
        files = sorted(filtered_args)
    else:
        files = sorted(glob.glob('captures/*.cf32'))
    for f in files:
        try:
            process_file(f)
        except Exception as e:
            import traceback
            print("  ERROR: %s" % e)
            traceback.print_exc()
