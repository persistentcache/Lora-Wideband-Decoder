"""Payload drift tracking must reach the last FEC block of long frames."""
import ast
from pathlib import Path

import numpy as np
import pytest

import decoder


@pytest.fixture(scope='module', params=['_dd_resid_curve', '_measure_resid_curve'])
def tracker(request):
    # Run both exact production trackers, with real FFTs and synthetic chirps.
    # Isolating the closure avoids header search/CRC retries unrelated to its
    # horizon; retained-IQ replays exercise that complete demodulation path.
    tree = ast.parse(Path(decoder.__file__).read_text())
    attempt = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_decode_attempt')
    node = next(n for n in attempt.body if isinstance(n, ast.FunctionDef)
                and n.name == request.param)
    code = compile(ast.Module(body=[node], type_ignores=[]),
                   decoder.__file__, 'exec')

    def run(ppm, ppm_pay, cr, need_bytes, available, *, ds_offset=0,
            partial_symbol=False, drift=0.0):
        # FFT geometry is deliberately small; symbol-count metadata is
        # independent of FFT size in these two tracking functions.
        n = 128
        t = np.arange(n, dtype=np.float64)
        up = np.exp(1j * np.pi * t * t / n)
        symbols = np.arange(available + 1)
        # Unknown data occupies four-bin lattice points, including FFT wrap.
        tones = (4 * (symbols % 32) + drift * symbols)[:, None]
        iq = (up * np.exp(2j * np.pi * tones * t / n)).astype(np.complex64)
        count = available * n + (n // 2 if partial_symbol else 0)
        data_start = 37
        ns = dict(vars(decoder), N=n, data_start=data_start,
                  iq1=np.r_[np.zeros(data_start, np.complex64),
                            iq.ravel()[:count]],
                  downchirp=up.conj().astype(np.complex64), cfo_shift=0,
                  pay_bin_group=4, ppm=ppm, ppm_pay=ppm_pay,
                  cr=cr, need_bytes=need_bytes)
        exec(code, ns)
        return ns[request.param](ds_offset, 0.0, 0.0)

    return run


@pytest.mark.parametrize('ppm,ppm_pay,cr,need_bytes,expected', [
    # Independently counted: five header-overflow nibbles, then whole FEC
    # blocks; eight header symbols are part of the tracked data stream.
    (10, 10, 4, 99, 168),   # PL97+CRC, 20 blocks at CR4/8
    (10, 12, 4, 99, 144),   # accepted LDRO-off, 17 blocks at CR4/8
    (10, 10, 1, 99, 108),   # the same 20 blocks at CR4/5
    (10, 10, 2, 99, 128),
    (10, 10, 3, 99, 148),
    (10, 10, 4, 239, 392),  # maximum supported PL237+CRC, 48 blocks
    (10, 10, 4, 56, 96),   # PL56 without CRC: 11 blocks, legacy horizon
    (10, 10, 4, 58, 104),  # PL56 with CRC: 12 blocks
    (5, 7, 1, 39, 96),     # no header overflow, 68 symbols: keep old96
    (10, 10, 4, 7, 96),    # one payload block: keep old96
])
def test_tracker_covers_actual_frame_without_following_extra_symbols(
        tracker, ppm, ppm_pay, cr, need_bytes, expected):
    curve = tracker(ppm, ppm_pay, cr, need_bytes, expected + 23)
    assert len(curve) == expected


@pytest.mark.parametrize('available', [0, 7, 80, 120])
def test_tracker_clamps_to_complete_available_symbols(tracker, available):
    curve = tracker(10, 10, 4, 99, available, partial_symbol=True)
    assert len(curve) == available


def test_tracker_accounts_for_start_offset_when_clamping(tracker):
    curve = tracker(10, 10, 4, 99, 121, ds_offset=128)
    assert len(curve) == 120


def test_tracking_keeps_following_drift_after_symbol96(tracker):
    curve = tracker(10, 10, 4, 99, 168, drift=0.125)
    assert len(curve) == 168
    # The last72 symbols drift by nine bins. A frozen symbol95 correction
    # leaves several entire LDRO groups wrong, even with perfect input IQ.
    assert curve[-1] - curve[95] == pytest.approx(9.0, abs=0.3)
