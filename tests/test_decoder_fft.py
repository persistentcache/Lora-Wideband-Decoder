"""Whole-capture FFT regressions, independent of payload recovery heuristics."""
import numpy as np
import pytest

import decoder


class _CropsCaptured(Exception):
    pass


def _capture_crops(monkeypatch, iq, filename, *, recenter=False, **kwargs):
    """Run the real FFT/crop path and stop at its demodulation boundary."""
    crops = []
    transforms = []
    fft = decoder._fft

    def record_fft(a, *args, **kw):
        result = fft(a, *args, **kw)
        transforms.append((len(a), len(result)))
        return result

    def capture(iq1, sf, bw, n, ppm, fs, dec, *args, **kw):
        crops.append((iq1.copy(), fs, dec))
        if recenter and len(crops) == 1:
            # A quarter-band CFO triggers the actual early-recenter branch.
            return ('FAIL', 0, 0, n // 4)
        raise _CropsCaptured

    monkeypatch.setattr(decoder, '_fft', record_fft)
    monkeypatch.setattr(decoder, '_decode_attempt', capture)
    with pytest.raises(_CropsCaptured):
        decoder._process_file_label(filename, _iq_override=iq,
                                    _budget_override=0, **kwargs)
    return crops, transforms


def _direct_crop(iq, fft_len, dec, shift_hz=0, fs=500_000):
    """Evaluate the retained DFT bins directly at original BW sample times.

    Sparse impulses make this independent oracle cheap without an FFT,
    fftshift, spectrum slicing, or the production crop implementation.
    """
    width = fft_len // dec
    bins = np.arange(-(width // 2), width - width // 2)
    shift = round(shift_hz * fft_len / fs)
    locations = np.flatnonzero(iq)
    spectrum = np.exp(-2j * np.pi * np.outer(bins + shift, locations) / fft_len) @ iq[locations]
    times = np.arange(0, len(iq), dec)
    return np.exp(2j * np.pi * np.outer(times, bins) / fft_len) @ spectrum / fft_len


def test_pathological_capture_uses_fast_transform_without_dropping_iq(monkeypatch):
    # The exact retained SHORT_TURBO size took 29 CPU seconds on the Pi.
    iq = np.zeros(7_664_160, dtype=np.complex64)
    iq[-1] = 1 + 2j
    crops, transforms = _capture_crops(
        monkeypatch, iq, 'SF7_500k_915MHz_5000ksps_test.cf32')
    assert transforms[0] == (7_664_160, 7_666_560)
    assert len(crops[0][0]) == 766_416
    assert np.any(crops[0][0][-8:] != 0)
    assert iq[-1] == 1 + 2j


@pytest.mark.parametrize('dec, fft_len', [(2, 1024), (3, 1029), (8, 1024), (10, 1050)])
def test_padding_preserves_rate_phase_and_capture_extent(monkeypatch, dec, fft_len):
    iq = np.zeros(1009, dtype=np.complex64)
    iq[[7, 28, 1008]] = [1 + 2j, -0.5j, 2 - 1j]
    original = iq.copy()
    crops, transforms = _capture_crops(
        monkeypatch, iq, f'SF7_500k_915MHz_{500 * dec}ksps_test.cf32', recenter=True)
    assert transforms == [(1009, fft_len)]  # recenter reuses the cached FFT
    for (actual, fs, actual_dec), shift in zip(crops, [0, 125_000]):
        assert fs == 500_000 * dec
        assert actual_dec == dec
        assert len(actual) == (1009 + dec - 1) // dec
        expected = _direct_crop(iq, fft_len, dec, shift, fs)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=3e-6)
    np.testing.assert_array_equal(iq, original)


def test_unity_decimation_keeps_every_original_sample(monkeypatch):
    rng = np.random.default_rng(123)
    iq = (rng.normal(size=1009) + 1j * rng.normal(size=1009)).astype(np.complex64)
    crops, transforms = _capture_crops(
        monkeypatch, iq, 'SF7_500k_915MHz_500ksps_test.cf32')
    assert transforms == [(1009, 1024)]
    np.testing.assert_allclose(crops[0][0], iq, atol=8e-7, rtol=2e-6)


def test_fractional_bandwidth_and_relay_window_keep_original_time_axis(monkeypatch):
    iq = np.zeros(1009, dtype=np.complex64)
    iq[[7, 500, 1008]] = [1, 2j, -1j]
    crops, transforms = _capture_crops(
        monkeypatch, iq, 'SF7_41.67k_915MHz_83ksps_test.cf32',
        relay_after=1, relay_before=3)
    assert transforms == [(1009, 1024)]
    actual, fs, dec = crops[0]
    assert fs == pytest.approx(250_000 / 3)
    assert dec == 2
    expected = _direct_crop(iq, 1024, 2, fs=fs)
    expected[:128] = 0
    expected[384:] = 0
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=3e-6)
