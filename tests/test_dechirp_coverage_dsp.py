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
