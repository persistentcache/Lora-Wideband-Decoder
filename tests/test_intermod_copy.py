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


from detector import OnAirFrames, ONAIR_STRONG_DB, _capture_duration_s


def _det(f_mhz, sf, bw, pwr=0.0, **kw):
    d = {'freq_hz': f_mhz * 1e6, 'freq_mhz': f_mhz, 'sf': sf, 'bw': bw, 'peak_power_db': pwr,
         'preamble_t_s': 0.0}
    d.update(kw)
    return d


def test_on_air_registry_recognises_copies_for_the_frame_duration():
    reg = OnAirFrames()
    long_frame = _det(915.0166, 12, 62500)
    assert reg.register(long_frame, 100.0, 52.0, _capture_duration_s(long_frame))
    dur = _capture_duration_s(long_frame)
    assert dur > 25.0
    copy = _det(918.0012, 12, 62500)
    assert reg.copy_of(copy, 100.3, 3.0)                       # preamble window
    assert reg.copy_of(copy, 100.0 + dur - 0.5, -1.0)          # last window of the body
    assert reg.copy_of(copy, 99.9, 2.0)                        # a preamble span of slack before
    assert reg.copy_of(copy, 100.0 + dur + 2.0, 2.0) is None   # frame over
    assert reg.copy_of(_det(911.0, 12, 62500), 110.0, 40.0) is None   # not 20 dB weaker
    assert reg.copy_of(_det(911.0, 10, 125000), 110.0, 2.0) is None   # another slope
    assert reg.copy_of(_det(915.03, 12, 62500), 110.0, 2.0) is None   # the frame's own carrier
    assert reg.copy_of(_det(911.1226, 10, 31250), 110.0, 2.0)         # same-slope alias copies too


def test_weak_frames_are_not_registered_and_old_ones_are_purged():
    reg = OnAirFrames()
    assert not reg.register(_det(911.0, 12, 62500), 10.0, ONAIR_STRONG_DB - 1, 30.0)
    assert len(reg) == 0
    assert reg.register(_det(915.0, 7, 125000), 10.0, 50.0, 2.3)
    reg.register(_det(915.0, 7, 125000), 11.0, 51.0, 2.3)      # the same frame, seen again
    assert len(reg) == 1
    reg.register(_det(915.0, 10, 250000), 60.0, 50.0, 6.0)       # a minute later: the SF7 is gone
    assert len(reg) == 1


def test_main_loop_filters_every_detection_path():
    from pathlib import Path
    src = Path(detector.__file__).read_text()
    for where in ("'pool'", "'inline'", "'slow'"):
        assert f", {where})" in src and '_onair_filter(' in src, where
    assert src.count('_onair_filter(') == 4            # definition + three paths
