"""Save corrections must follow the dominant band, not a weak spectral decoy."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import detector


def band_limited_iq(bands, *, tone=None, fs=500000, count=65536):
    """Real IQ with independently specified occupied bands and a noise floor."""
    rng = np.random.default_rng(20261028)
    frequency = np.fft.fftfreq(count, 1 / fs)
    amplitude = np.full(count, .03)
    for center, width, level in bands:
        amplitude[np.abs(frequency - center) < width / 2] = level
    spectrum = (rng.standard_normal(count) + 1j * rng.standard_normal(count)) * amplitude
    iq = np.fft.ifft(spectrum).astype(np.complex64)
    if tone:
        center, level = tone
        iq += level / np.sqrt(count) * np.exp(2j * np.pi * center / fs * np.arange(count))
    return iq


@pytest.fixture(scope='module')
def correct_export():
    # Execute the actual save-worker correction, including its decision to
    # mix the exported samples. The surrounding file queue is unrelated.
    tree = ast.parse(Path(detector.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SignalRecorder')
    worker = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_save_worker')
    block = next(n for n in ast.walk(worker) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == '_CROP_CENTER_FIX and len(nb)')
    code = compile(ast.Module(body=[block], type_ignores=[]), detector.__file__, 'exec')

    def run(iq, *, preamble_sample=0, fs=500000, bw=62500):
        namespace = dict(vars(detector), nb=iq, nb_fs=fs,
                         d={'sf': 12, 'bw': bw, 'freq_hz': 915019500.},
                         _nb_job={}, _crop_corr=0., _CROP_CENTER_FIX=True,
                         _preamble_sample=preamble_sample,
                         self=SimpleNamespace(debug=0))
        exec(code, namespace)
        return namespace['nb'], namespace['_crop_corr']

    return run


def test_stronger_broad_region_prevents_weak_band_from_moving_export(correct_export):
    iq = band_limited_iq([(-20000, 180000, 10), (150000, 62500, 1)])
    saved, correction = correct_export(iq)
    assert correction == 0
    np.testing.assert_array_equal(saved, iq)


@pytest.mark.parametrize('center', [-100000, 50000])
def test_dominant_valid_band_keeps_existing_correction(center, correct_export):
    iq = band_limited_iq([(center, 62500, 10)])
    saved, correction = correct_export(iq)
    assert correction == pytest.approx(center, abs=500)
    # The save-stage mixer moves the actual spectral energy toward DC.
    spectrum = np.abs(np.fft.fft(saved)) ** 2
    frequency = np.fft.fftfreq(len(saved), 1 / 500000)
    assert abs(np.sum(frequency * spectrum) / np.sum(spectrum)) < 1500


def test_narrow_strong_cw_does_not_veto_valid_chirp_band():
    iq = band_limited_iq([(50000, 62500, 10)], tone=(-150000, 200))
    assert detector._nb_plateau_offset(iq, 500000, 62500) == pytest.approx(50000, abs=500)


def test_weaker_broad_region_does_not_veto_dominant_valid_band():
    iq = band_limited_iq([(50000, 62500, 10), (-150000, 180000, 1)])
    assert detector._nb_plateau_offset(iq, 500000, 62500) == pytest.approx(50000, abs=500)


@pytest.mark.parametrize('bw', [62500, 125000, 500000])
def test_actual_chirp_correction_scales_with_bandwidth(bw):
    fs = bw * 4
    count = 65536
    offset = .7 * bw
    t = np.arange(count) / fs
    within = np.remainder(t, 4096 / bw)
    rng = np.random.default_rng(38)
    iq = np.exp(2j * np.pi * (-bw / 2 * within + bw / (2 * (4096 / bw)) * within**2 + offset * t))
    # A usable finite SNR gives the established floor+6dB contour; nearly
    # noiseless chirps instead expose much wider spectral skirts.
    iq += .05 * (rng.standard_normal(count) + 1j * rng.standard_normal(count))
    assert detector._nb_plateau_offset(iq, fs, bw) == pytest.approx(offset, abs=fs / 2048)


def test_insufficient_iq_keeps_abstention():
    assert detector._nb_plateau_offset(np.zeros(8191, np.complex64), 500000, 62500) is None
