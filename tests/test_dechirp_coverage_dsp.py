"""Real detector routing near input edges, without any hardware or decoder."""
import numpy as np
import pytest

import detector


def chirps(rate, offset, sf, bw, duration):
    t=np.arange(int(rate*duration),dtype=np.float64)/rate
    period=(1<<sf)/bw
    within=np.remainder(t,period)
    phase=-bw/2*within+bw/(2*period)*within**2+offset*t
    return np.exp(2j*np.pi*phase).astype(np.complex64)


def scan(iq,rate,offset,bw,channels):
    psd=np.full(4096,-80.,dtype=np.float32)
    peak=(2048+int(round(offset/(rate/4096))),round(bw/(rate/4096)),-80.,-80.)
    return detector.detect_preamble(iq,rate,rate,915.,cached_psd=psd,
        cached_peaks=[peak],dechirp_chans=channels)


@pytest.mark.parametrize('rate',[8e6,10e6])
@pytest.mark.parametrize('sign',[-1,1])
def test_real_covered_trial_keeps_its_narrow_family_at_input_edge(monkeypatch,rate,sign):
    monkeypatch.setenv('LORA_FFT_WORKERS','1')
    offset=sign*(rate/2-50000.)
    iq=chirps(rate,offset,10,62500.,.36)
    detections=scan(iq,rate,offset,125000.,[(offset,10,62500.,True)])
    assert len(detections)==1
    detection=detections[0]
    assert detection['patience_trial'] and detection['forced_dechirp']
    assert (detection['sf'],detection['bw'])==(10,62500)
    assert detection['_patience_covered_family']==((10,62500.),)
    assert detector.patience_cap_params(detection)[0]==62500.
    assert detector._rf_footprint_covered(detection['freq_hz'],62500.,915e6,rate)
    assert not detector._rf_footprint_covered(detection['freq_hz'],125000.,915e6,rate)


@pytest.mark.parametrize('sign',[-1,1])
def test_uncovered_forced_channel_preserves_real_ordinary_sc_result(monkeypatch,sign):
    monkeypatch.setenv('LORA_FFT_WORKERS','1')
    rate=10e6
    offset=sign*4.85e6
    iq=chirps(rate,offset,7,250000.,.12)
    ordinary=scan(iq,rate,offset,250000.,None)
    with_invalid=scan(iq,rate,offset,250000.,[(sign*4.994165349633455e6,7,250000.)])
    assert ordinary
    assert with_invalid==ordinary
    assert all(not detection.get('forced_dechirp') for detection in with_invalid)


def banded_noise(rate, offset, bw, duration, amplitude):
    """Band-limited noise: a strong energy peak with no preamble structure."""
    n = int(rate * duration)
    rng = np.random.default_rng(7)
    white = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    spectrum = np.fft.fft(white)
    freqs = np.fft.fftfreq(n, 1.0 / rate)
    spectrum[np.abs(freqs - offset) > bw / 2] = 0
    shaped = np.fft.ifft(spectrum).astype(np.complex64)
    return (amplitude * shaped / (np.abs(shaped).max() + 1e-12)).astype(np.complex64)


def test_strong_unconfirmed_peak_is_logged_once_per_carrier(capsys, monkeypatch):
    monkeypatch.setattr(detector, '_unconfirmed_last', {})
    rate = 10e6
    iq = banded_noise(rate, 300e3, 125e3, 1.0, 1.0)
    psd = np.full(4096, -80., dtype=np.float32)
    width = round(125e3 / (rate / 4096))
    strong = (2048 + int(round(300e3 / (rate / 4096))), width, -40., -40.)   # +40 dB over floor
    assert detector.detect_preamble(iq, rate, rate, 915., cached_psd=psd,
                                    cached_peaks=[strong], debug=1) == []
    err = capsys.readouterr().err
    assert err.count('[GATE-UNCONFIRMED]') == 1
    assert 'peak 915.30' in err and '+40dB over floor' in err
    assert ('SC best lag' in err) or ('no bandwidth class reached' in err)
    # The same carrier again inside the rate-limit window is not repeated ...
    detector.detect_preamble(iq, rate, rate, 915., cached_psd=psd, cached_peaks=[strong], debug=1)
    assert '[GATE-UNCONFIRMED]' not in capsys.readouterr().err
    # ... a weak unconfirmed peak is never reported ...
    weak = (strong[0], width, -70., -70.)                                        # +10 dB
    monkeypatch.setattr(detector, '_unconfirmed_last', {})
    detector.detect_preamble(iq * 0.03, rate, rate, 915., cached_psd=psd, cached_peaks=[weak], debug=1)
    assert '[GATE-UNCONFIRMED]' not in capsys.readouterr().err
    # ... and nothing is printed at debug 0.
    monkeypatch.setattr(detector, '_unconfirmed_last', {})
    detector.detect_preamble(iq, rate, rate, 915., cached_psd=psd, cached_peaks=[strong], debug=0)
    assert '[GATE-UNCONFIRMED]' not in capsys.readouterr().err


def test_a_real_preamble_is_not_reported_as_unconfirmed(capsys, monkeypatch):
    monkeypatch.setattr(detector, '_unconfirmed_last', {})
    monkeypatch.setenv('LORA_FFT_WORKERS', '1')
    rate, offset = 10e6, 4.85e6
    iq = chirps(rate, offset, 7, 250000., .12)
    psd = np.full(4096, -80., dtype=np.float32)
    peak = (2048 + int(round(offset / (rate / 4096))), round(250000. / (rate / 4096)), -40., -40.)
    dets = detector.detect_preamble(iq, rate, rate, 915., cached_psd=psd, cached_peaks=[peak], debug=1)
    assert dets and all(d['sf'] == 7 for d in dets)
    assert '[GATE-UNCONFIRMED]' not in capsys.readouterr().err
