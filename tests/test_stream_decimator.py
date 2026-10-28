"""Keep capture FIR service cheap without changing its samples or stream grid."""
import numpy as np
import pytest
from scipy.signal import firwin, resample_poly, upfirdn

import detector


def iq(n=30001):
    rng = np.random.default_rng(187)
    return (rng.standard_normal(n, dtype=np.float32)
            + 1j * rng.standard_normal(n, dtype=np.float32)).astype(np.complex64)


def reference(x, offset, rate, factor):
    # Independent full-stream zero-phase FIR, including the final padding.
    phase = (np.arange(len(x), dtype=np.float64) * (offset / rate)) % 1.0
    mixed = x * np.exp(-2j * np.pi * phase).astype(np.complex64)
    taps = firwin(20 * factor + 1, 1 / factor,
                  window=('kaiser', 5)).astype(np.float32)
    return resample_poly(mixed, 1, factor, window=taps)


@pytest.mark.parametrize('factor', [2, 5, 20])
def test_real_coefficient_capture_filter_avoids_complex_fir_work(monkeypatch, factor):
    # Regression: real taps sent through scipy's complex FIR made3 streams
    # take0.84s per0.5s input. Observe actual backend work, not wall-clock jitter;
    # the backend still computes every result (no fake outputs).
    dec = detector._StreamDecimator(35000, 10e6, factor)
    backend_dtypes = []

    def observed_fir(h, x, **kwargs):
        backend_dtypes.append(x.dtype)
        return upfirdn(h, x, **kwargs)

    monkeypatch.setattr(detector, '_upfirdn', observed_fir)
    x = iq()
    actual = np.concatenate([dec.feed(x), dec.finish()])
    np.testing.assert_allclose(actual, reference(x, 35000, 10e6, factor),
                               rtol=1e-5, atol=1e-6)
    assert backend_dtypes
    assert all(dtype == np.float32 for dtype in backend_dtypes), (
        'real FIR coefficients must not pay the complex convolution cost')


@pytest.mark.parametrize('factor', [2, 5, 20])
@pytest.mark.parametrize('offset', [0, 35000, -200000])
def test_chunk_seams_and_final_padding_match_full_stream(factor, offset):
    x = iq()
    dec = detector._StreamDecimator(offset, 10e6, factor)
    # Include sub-filter chunks and boundaries off both mixer and FIR grids.
    chunks = np.split(x, [1, 8, 8201, 17729])
    actual = np.concatenate([*(dec.feed(chunk) for chunk in chunks), dec.finish()])
    expected = reference(x, offset, 10e6, factor)
    assert actual.dtype == np.complex64
    assert len(actual) == (len(x) + factor - 1) // factor
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('factor', [2, 5, 20])
def test_single_sample_and_empty_stream_keep_output_grid(factor):
    dec = detector._StreamDecimator(0, 10e6, factor)
    assert len(dec.feed(np.empty(0, np.complex64))) == 0
    assert len(dec.finish()) == 0
    x = np.array([1 + 2j], np.complex64)
    actual = np.concatenate([dec.feed(x), dec.finish()])
    assert len(actual) == 1
    np.testing.assert_allclose(actual, reference(x, 0, 10e6, factor),
                               rtol=1e-6, atol=1e-7)


def test_multiple_detection_streams_keep_independent_phase_and_fir_state():
    x = iq()
    offsets = [35000, -200000, 170000]
    nb = {'decs': {i: detector._StreamDecimator(offset, 10e6, 20)
                   for i, offset in enumerate(offsets)},
          'parts': {i: [] for i in range(3)}}
    for chunk in np.split(x, [8193, 17001]):
        detector.SignalRecorder._crop_feed(nb, chunk)
    for i, dec in nb['decs'].items():
        actual = np.concatenate(nb['parts'][i] + [dec.finish()])
        np.testing.assert_allclose(actual, reference(x, offsets[i], 10e6, 20),
                                   rtol=1e-5, atol=1e-6)
