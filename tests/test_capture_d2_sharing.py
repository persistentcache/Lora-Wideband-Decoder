"""D2 views share raw history without changing their individual FIR output."""
import math

import numpy as np
import pytest

import detector
from capture_timeline import CaptureBudget, SharedCapture
from test_capture_timeline import reader


@pytest.mark.parametrize('count', [4, 8])
def test_nearby_d2_captures_fit_aggregate_header_budget(count):
    # The extended live failure had several SF12/500 estimates near one
    # carrier. Four seconds of base plus the full CR4/8 tail is 7.508224s.
    # No IQ allocation is needed to exercise the real admission ledger.
    budget = CaptureBudget(3 * 1024**3, 64)
    header = budget.reserve('header_worker_workspace', 671088640)
    prefixes = budget.reserve('header_prefix_file', 2 * 67108864)
    capture = SharedCapture(
        rate=10_000_000, center=915e6, hop_n=5_000_000, reader=None,
        decimator_factory=detector._StreamDecimator,
        submit=lambda view: False, budget=budget, threaded=False)
    try:
        for index in range(count):
            detection = dict(sf=12, bw=500000, freq_hz=914_873_600 + index * 36_000,
                             peak_power_db=20, detect_conf=.99)
            admitted = capture.register(
                [detection], base_start=0, base_end=40_000_000,
                end=75_082_240, anchor=(40_000_000, 0, 0), target_factor=2,
                export_bw=2_500_000, pre_hop_n=0, wb_n=40_000_000,
                sample_pos=40_000_000)
            assert admitted is not None, capture.failure
        state = capture.stats()
        assert state['active_views'] == count
        assert state['bands'] == 1
        assert state['band_details'][0]['factor'] == 1
        assert state['budget']['used_bytes'] <= 3 * 1024**3
        assert state['budget']['outbox_slots'] == count
        assert all(view.start == 0 and view.end == 75_082_240
                   and view.nb_fs == 5_000_000 for view in capture._views)
    finally:
        capture._fail('test teardown')
        capture.close(wait=False)
        prefixes.release()
        header.release()
    assert budget.stats()['used_bytes'] == budget.stats()['outbox_slots'] == 0


@pytest.mark.parametrize('hop_n', [17, 64, 500])
def test_d2_shared_views_match_direct_fir_across_origins_and_hop_seams(hop_n):
    history = reader(capacity=8000)
    submitted = []
    budget = CaptureBudget(2_000_000, 16)
    capture = SharedCapture(
        rate=1000, center=915e6, hop_n=hop_n, reader=history,
        decimator_factory=detector._StreamDecimator,
        submit=lambda view: submitted.append(view) or True,
        budget=budget, threaded=False)
    # The last center's requested export crosses Nyquist, but its 62.5Hz
    # signal is in-band. Each span starts on its own exact integer sample.
    spans = [(1001, -125.), (1020, -3.7), (1043, 12.9), (1100, 450.)]
    history._rp = history._wp = 5000
    seen = []
    try:
        for start, offset in spans:
            detection = dict(sf=7, bw=62.5, freq_hz=915e6 + offset,
                             peak_power_db=20, detect_conf=.99)
            assert capture.register(
                [detection], base_start=start, base_end=start + 500,
                end=start + 3001, anchor=(start + 500, 0, 0),
                target_factor=2, export_bw=250, pre_hop_n=0,
                wb_n=500, sample_pos=start + 500) is not None
        assert capture.stats()['bands'] == 1
        for _ in range(300):
            capture.step()
            if submitted:
                view = submitted.pop(0)
                start, offset = spans[view.number]
                samples = np.arange(start, start + 3021, dtype=np.float32)
                raw = (samples - 1j * samples).astype(np.complex64)
                direct = detector._StreamDecimator(offset, 1000, 2)
                expected = direct.feed(raw)[:1501]
                actual = view.stream(0)
                assert view.complete and len(actual) == len(expected) == 1501
                assert view.detections[0]['_capture_start_offset_s'] == 0
                # The same float32 mixer/FIR is fed different chunk lengths;
                # the block phasor factorization permits rounding differences.
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=.003)
                seen.append(view.number)
                assert budget.stats()['by_kind']['band_retention'] > 0
                view.release()
            if len(seen) == len(spans):
                break
        assert seen == [0, 1, 2, 3]
        assert capture.stats()['completed_views'] == 4
        assert capture.stats()['incomplete_views'] == 0
        assert capture.stats()['bands'] == 0
        assert budget.stats()['by_kind']['band_retention'] == 0
        capture.close(wait=False)
    finally:
        for view in submitted:
            view.release()
        capture._fail('test teardown')
        capture.close(wait=False)
    assert budget.stats()['used_bytes'] == budget.stats()['outbox_slots'] == 0


def test_d2_shared_view_waits_for_real_right_context():
    history = reader(capacity=8000)
    submitted = []
    capture = SharedCapture(
        rate=1000, center=915e6, hop_n=500, reader=history,
        decimator_factory=detector._StreamDecimator,
        submit=lambda view: submitted.append(view) or True,
        budget=CaptureBudget(2_000_000, 4), threaded=False)
    try:
        history._rp = history._wp = 4021
        capture.register(
            [dict(sf=7, bw=62.5, freq_hz=915e6, peak_power_db=20)],
            base_start=1001, base_end=1501, end=4002,
            anchor=(1501, 0, 0), target_factor=2, export_bw=250,
            pre_hop_n=0, wb_n=500, sample_pos=1501)
        for _ in range(20):
            if not capture.step():
                break
        assert not submitted
        history._rp = history._wp = 4022
        for _ in range(20):
            capture.step()
            if submitted:
                break
        assert len(submitted) == 1
        view = submitted.pop()
        assert view.complete and len(view.stream(0)) == math.ceil(3001 / 2)
        capture.close(wait=False)
        assert capture.budget.stats()['used_bytes'] > 0
        view.release()
        assert capture.budget.stats()['used_bytes'] == 0
    finally:
        for view in submitted:
            view.release()
        capture._fail('test teardown')
        capture.close(wait=False)
