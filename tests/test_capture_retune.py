"""A commanded retune preserves received IQ without crossing radio generations."""
import threading

import numpy as np
import pytest

import detector
from test_capture_timeline import engine, advance, detection
from test_header_capture import fixture as header_fixture, prepare
from test_shared_recorder import rec


def admit(capture, reader, *, start=1, base_end=2001, end=6001, factor=2,
          frequency=None):
    det = dict(detection(), freq_hz=frequency if frequency is not None else capture.center)
    return capture.register([det], base_start=start, base_end=base_end, end=end,
        anchor=reader.history_anchor(), target_factor=factor,
        export_bw=capture.rate / (2 * factor), pre_hop_n=0,
        wb_n=base_end-start, sample_pos=base_end)


def test_idle_commanded_retune_accepts_new_frequency_with_same_budget():
    submitted = []
    capture, history, _ = engine(submit=lambda view: submitted.append(view) or True)
    budget = capture.budget
    advance(capture, history, 2001)
    capture.retune(916e6)
    assert history.history_anchor()[2] == 1
    assert capture.budget is budget
    assert capture.center == 916e6
    assert admit(capture, history, start=2001, base_end=4001, end=5001) is not None
    advance(capture, history, 5100)
    assert len(submitted) == 1
    assert submitted[0].input_center == 916e6
    assert submitted[0].complete
    assert len(submitted[0].stream(0)) == 1500
    submitted[0].release()
    capture.close(wait=False)
    assert not capture.failure
    assert budget.stats()['used_bytes'] == 0


def test_retune_saves_only_received_prefix_with_real_fir_support():
    submitted = []
    capture, history, _ = engine(submit=lambda view: submitted.append(view) or True)
    advance(capture, history, 2001)
    admit(capture, history, frequency=915e6 + 100)
    advance(capture, history, 3001)
    source = next(history.history_parts((1, 0, 0), 3000, 3000))
    expected = detector._StreamDecimator(100, 1000, 2).feed(source)
    assert not submitted  # the requested future tail has not arrived
    capture.retune(916e6)
    capture.step()
    assert len(submitted) == 1
    partial = submitted.pop()
    assert partial.role == 'shared_partial'
    assert not partial.complete
    assert partial.input_center == 915e6
    actual = partial.stream(0)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=.002)
    assert len(actual) < 3000
    partial.release()
    assert capture.stats()['interrupted_views'] == 1
    assert capture.stats()['completed_views'] == 0
    assert admit(capture, history, start=3001, base_end=4001, end=5001) is not None
    advance(capture, history, 5100)
    fresh = submitted.pop()
    assert fresh.complete and fresh.input_center == 916e6
    fresh.release()
    capture.close(wait=False)
    assert not capture.failure
    assert capture.budget.stats()['used_bytes'] == 0


@pytest.mark.parametrize('factor', [2, 5, 8])
def test_writer_owned_capture_remains_bit_identical_across_retunes(factor):
    submitted = []
    capture, history, _ = engine(submit=lambda view: submitted.append(view) or True)
    advance(capture, history, 2001)
    admit(capture, history, end=4001, factor=factor, frequency=915e6 + 10)
    advance(capture, history, 4300)
    view = submitted.pop()
    before = view.stream(0)
    budget = capture.budget.stats()['used_bytes']
    for center in (916e6, 914e6, 917e6):
        capture.retune(center)
        np.testing.assert_array_equal(view.stream(0), before)
        assert view.input_center == 915e6 and view.complete
        assert capture.budget.stats()['used_bytes'] == budget
    view.release()
    capture.close(wait=False)
    assert not capture.failure
    assert capture.stats()['completed_views'] == 1
    assert capture.budget.stats()['used_bytes'] == 0


@pytest.mark.parametrize('copy_succeeds', [False, True])
def test_inflight_old_crop_cannot_publish_into_or_fail_new_generation(copy_succeeds):
    capture, history, _ = engine()
    advance(capture, history, 2001)
    admit(capture, history)
    original = history.received_history_parts
    entered, release = threading.Event(), threading.Event()

    def paused(*args):
        data = next(original(*args)) if copy_succeeds else None
        entered.set()
        assert release.wait(2)
        if data is None:
            raise detector.StreamHistoryUnavailable('stream skipped or retuned')
        yield data

    history.received_history_parts = paused
    thread = threading.Thread(target=capture.step)
    thread.start()
    try:
        assert entered.wait(2)
        capture.retune(916e6)
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not capture.failure
    history.received_history_parts = original
    assert admit(capture, history, start=2001, base_end=3001, end=4001) is not None
    advance(capture, history, 4100)
    assert capture.stats()['ready_views'] == 1
    capture._fail('test teardown')
    capture.close(wait=False)
    capture.step()
    assert capture.budget.stats()['used_bytes'] == 0


def test_pre_retune_published_snapshot_cannot_rewind_generation():
    capture, history, _ = engine()
    advance(capture, history, 2001)
    original = history.received_history_anchor
    entered, release = threading.Event(), threading.Event()

    def paused_snapshot():
        snapshot = original()
        if not entered.is_set():
            entered.set()
            assert release.wait(2)
        return snapshot

    history.received_history_anchor = paused_snapshot
    thread = threading.Thread(target=capture.step)
    thread.start()
    try:
        assert entered.wait(2)
        capture.retune(916e6)
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not capture.failure
    assert capture._watermark[2] == 1
    capture.close(wait=False)


def test_new_generation_rejects_pre_retune_lookback_without_reading_it():
    capture, history, _ = engine()
    advance(capture, history, 2001)
    capture.retune(916e6)
    assert admit(capture, history, start=1) is None
    assert not capture.failure
    assert not capture._views
    assert capture.stats()['non_admitted'] == 1
    capture.close(wait=False)


@pytest.mark.parametrize('requested', [False, True])
def test_retune_releases_old_header_entries_without_closing_worker(header_fixture, requested):
    f = header_fixture
    from test_header_capture import advance as header_advance
    if requested:
        prepare(f)
    else:
        header_advance(f, 71000)
    assert f.controller._entries
    service = f.controller.service
    f.capture.retune(916e6)
    assert not f.controller._entries
    assert f.controller.service is service and not f.controller._closed
    assert f.capture.budget.stats()['by_kind']['header_hold'] == 0
    assert f.capture.budget.stats()['by_kind']['header_prefix_file'] == 0
    for view in f.submitted:
        f.controller.write_prefix(view)  # already-transferred stale prefix is harmless
        view.release()
    f.submitted.clear()
    f.capture.step()
    assert f.controller.counts['early'] == 0
    assert not f.capture.failure


def test_delayed_recorder_result_from_commanded_old_generation_is_discarded(rec):
    from test_capture_timeline import reader
    history = reader()
    history._rp = history._wp = 4000
    rec.set_history_reader(history)
    old = history.history_anchor()
    rec.retune(916e6)
    rec.update_deferred([detection()], np.zeros(4000, np.complex64), sample_pos=4000,
        need_tail_n=2000, history_anchor=old)
    assert not rec._shared.failure
    assert not rec._shared._views
    assert rec.center_hz == rec._shared.center == 916e6


def test_partial_batch_keeps_each_bands_actual_endpoint():
    submitted = []
    capture, history, _ = engine(submit=lambda view: submitted.append(view) or True)
    advance(capture, history, 2001)
    detections = [dict(detection(), freq_hz=915e6+offset) for offset in (0, 100)]
    capture.register(detections, base_start=1, base_end=2001, end=6001,
        anchor=history.history_anchor(), target_factor=5, export_bw=100,
        pre_hop_n=0, wb_n=2000, sample_pos=2001)
    capture.step()  # first band has emitted real samples; second has none
    band = capture._bands[0]
    before = np.concatenate([a for _, a in band.chunks]).copy()
    assert len(before) > 0 and capture._bands[1].produced == 0
    capture.retune(916e6)
    capture.step()
    partial = submitted.pop()
    np.testing.assert_array_equal(partial.stream(0), before)
    assert partial.stream(1).size == 0
    assert partial.metadata['requested_end'] == 6001
    assert partial.metadata['received_ends'][0] > partial.metadata['received_ends'][1]
    partial.release()
    capture.close(wait=False)
    assert not capture.failure and capture.budget.stats()['used_bytes'] == 0


def test_retune_during_prefix_write_leaves_cleanup_with_writer(header_fixture):
    f = header_fixture
    from test_header_capture import advance as header_advance
    header_advance(f, 71000)
    prefix = f.submitted.pop()
    original = prefix.stream
    entered, release = threading.Event(), threading.Event()

    def paused_stream(index):
        samples = original(index)
        entered.set()
        assert release.wait(2)
        return samples

    prefix.stream = paused_stream
    thread = threading.Thread(target=f.controller.write_prefix, args=(prefix,))
    thread.start()
    try:
        assert entered.wait(2)
        held = f.capture.budget.stats()['by_kind']['header_prefix_file']
        assert held > 0
        f.capture.retune(916e6)
        assert f.capture.budget.stats()['by_kind']['header_prefix_file'] == held
    finally:
        release.set()
        thread.join(2)
        prefix.release()
    assert not thread.is_alive()
    assert not f.headers.requests and not f.controller._entries
    assert not list(f.path.glob('.header-*'))
    assert f.capture.budget.stats()['by_kind']['header_prefix_file'] == 0
    assert not f.capture.failure


def test_partial_recorder_save_keeps_received_iq_and_explicit_role(rec):
    from test_header_recorder import worker
    from pathlib import Path
    submitted = []
    capture, history, _ = engine(submit=lambda view: submitted.append(view) or True)
    advance(capture, history, 2001)
    admit(capture, history)
    advance(capture, history, 3001)
    capture.retune(916e6)
    capture.step()
    partial = submitted.pop()
    original = partial.stream(0)
    rec.center_hz = 916e6
    assert rec._queue_shared_save(partial)
    worker(rec)
    saved = list(Path(rec.export_dir).glob('*_shared_partial_*.cf32'))
    assert len(saved) == 1
    np.testing.assert_array_equal(np.fromfile(saved[0], np.complex64), original)
    assert '915.0000MHz' in saved[0].name
    assert not capture.failure and partial._released
    capture.close(wait=False)
    assert capture.budget.stats()['used_bytes'] == 0
