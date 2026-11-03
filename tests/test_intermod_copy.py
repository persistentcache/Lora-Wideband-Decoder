"""A time-coincident, same-slope, much weaker detection is a receiver copy of
the strong frame (intermod onto a spur carrier), not a second transmitter."""
import numpy as np
import pytest

import detector
from detector import _same_slope, INTERMOD_COPY_DB
from test_dechirp_coverage_dsp import chirps

RATE = 10e6
FRES = RATE / 4096.0


def test_slope_families():
    d = lambda sf, bw: {'sf': sf, 'bw': bw}
    assert _same_slope(d(7, 125000), d(9, 250000)) and _same_slope(d(9, 250000), d(11, 500000))
    assert _same_slope(d(12, 62500), d(12, 62500))
    assert not _same_slope(d(7, 125000), d(7, 250000))
    assert not _same_slope(d(12, 62500), d(10, 125000))


def _peak(off_hz, bw_hz, power_db):
    return (2048 + int(round(off_hz / FRES)), max(1, round(bw_hz / FRES)), power_db, -80.0)


def _window(copy_db, copy_sf=9, copy_bw=250000.):
    """A strong SF7/125k burst at +0.85 MHz and its copy copy_db weaker at
    -3.03 MHz (3.88 MHz apart, like the live spur-grid copy), both chirping
    at the same slope and starting together."""
    main = chirps(RATE, 0.85e6, 7, 125000., .12)
    copy = chirps(RATE, -3.03e6, copy_sf, copy_bw, .12) * 10 ** (-copy_db / 20)
    return (main + copy).astype(np.complex64)


@pytest.mark.parametrize('copy_sf,copy_bw', [(9, 250000.), (7, 125000.)])
def test_weak_same_slope_copy_is_dropped(monkeypatch, capsys, copy_sf, copy_bw):
    monkeypatch.setenv('LORA_FFT_WORKERS', '1')
    psd = np.full(4096, -80., np.float32)
    dets = detector.detect_preamble(_window(25.0, copy_sf, copy_bw), RATE, RATE, 915.,
                                    cached_psd=psd, debug=1,
                                    cached_peaks=[_peak(0.85e6, 125e3, -30.), _peak(-3.03e6, copy_bw, -55.)])
    assert [(d['sf'], round(d['freq_mhz'], 2)) for d in dets] == [(7, 915.85)]
    assert '[GATE] intermod copy dropped: 911.96' in capsys.readouterr().err


def test_a_real_second_transmitter_survives(monkeypatch):
    """Comparable power: two nodes, not a copy."""
    monkeypatch.setenv('LORA_FFT_WORKERS', '1')
    psd = np.full(4096, -80., np.float32)
    dets = detector.detect_preamble(_window(6.0), RATE, RATE, 915., cached_psd=psd,
                                    cached_peaks=[_peak(0.85e6, 125e3, -30.), _peak(-3.03e6, 250e3, -36.)])
    assert sorted(round(d['freq_mhz'], 2) for d in dets) == [911.97, 915.85]


def test_a_weak_frame_of_another_slope_survives(monkeypatch):
    monkeypatch.setenv('LORA_FFT_WORKERS', '1')
    psd = np.full(4096, -80., np.float32)
    main = chirps(RATE, 0.85e6, 7, 125000., .12)
    other = chirps(RATE, -3.03e6, 7, 250000., .12) * 10 ** (-25 / 20)   # different slope
    dets = detector.detect_preamble((main + other).astype(np.complex64), RATE, RATE, 915.,
                                    cached_psd=psd,
                                    cached_peaks=[_peak(0.85e6, 125e3, -30.), _peak(-3.03e6, 250e3, -55.)])
    assert sorted(round(d['freq_mhz'], 2) for d in dets) == [911.97, 915.85]
