"""Learned-channel acquisition needs a persistent spectral peak across symbols."""
import numpy as np
import pytest
import detector


def preamble(sf=11, bw=500000, *, snr=-20, cfo_bins=.5,
             sfo_ppm=0, drift_bins_per_symbol=0, unrelated_bins=False):
    fs = 2 * bw
    n = 2 ** sf
    period = n / bw
    t = np.arange(40 * n) / fs
    within = np.remainder(t * (1 + sfo_ppm * 1e-6), period)
    drift_hz_s = drift_bins_per_symbol * (bw / n) ** 2
    phase = (-bw / 2 * within + bw / (2 * period) * within**2
             + cfo_bins * bw / n * t + .5 * drift_hz_s * t**2)
    if unrelated_bins:
        # Strong per-symbol peaks, but each symbol occupies a different bin:
        # this is not a repeating LoRa preamble.
        phase += (np.floor(t / period).astype(int) * 173 % n) * within / period
    rng = np.random.default_rng(2810)
    noise = (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    iq = np.exp(2j * np.pi * phase) + np.sqrt(10 ** (-snr / 10) / 2) * noise
    return iq.astype(np.complex64), fs


def test_unrelated_per_symbol_maxima_do_not_form_a_preamble():
    iq, fs = preamble(snr=-18, cfo_bins=0, unrelated_bins=True)
    quality, _, _ = detector._dechirp_scan(iq, 11, 500000, fs)
    assert quality < detector.DECHIRP_MF_MIN_DB


@pytest.mark.parametrize('cfo_bins', [-.51, -.50, -.49, .49, .50, .51])
def test_threshold_adjacent_preamble_survives_fractional_and_wrapped_bins(cfo_bins):
    iq, fs = preamble(snr=-21, cfo_bins=cfo_bins)
    quality, position, _ = detector._dechirp_scan(iq, 11, 500000, fs)
    assert quality >= detector.DECHIRP_MF_MIN_DB
    assert 0 <= position < len(iq) / fs - 7 * 2048 / 500000


@pytest.mark.parametrize('cfo_bins', [.49, .50, .51])
def test_below_threshold_preamble_is_not_promoted_by_bin_tolerance(cfo_bins):
    iq, fs = preamble(snr=-22, cfo_bins=cfo_bins)
    assert detector._dechirp_scan(iq, 11, 500000, fs)[0] < detector.DECHIRP_MF_MIN_DB


@pytest.mark.parametrize('sf,bw,sfo,drift,snr', [
    (11, 500000, 30, .08, -20),
    (11, 500000, -30, -.08, -20),
    (12, 62500, 15, .1, -22),
    (12, 31250, 0, .8, -22),
    (12, 41667, 0, -.8, -22),
])
def test_clock_error_and_narrowband_drift_keep_real_preambles(sf, bw, sfo, drift, snr):
    iq, fs = preamble(sf, bw, snr=snr, sfo_ppm=sfo,
                      drift_bins_per_symbol=drift)
    assert detector._dechirp_scan(iq, sf, bw, fs)[0] >= detector.DECHIRP_MF_MIN_DB


def test_zero_and_short_windows_cannot_trigger():
    assert detector._dechirp_scan(np.zeros(32767, np.complex64), 11, 500000, 1000000) is None
    assert detector._dechirp_scan(np.zeros(65536, np.complex64), 11, 500000, 1000000)[0] < 12
