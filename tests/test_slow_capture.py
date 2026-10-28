"""Delayed slow-scan registration must preserve contiguous packet tails."""
import queue
import threading
from collections import deque

import numpy as np
import pytest
import detector


def samples(start, count):
    values = np.arange(start, start + count, dtype=np.float32)
    return (values - 1j * values).astype(np.complex64)


def ring_reader(consumed, *, received=None, capacity=16000):
    # Real StreamBuffer methods over a populated ring; no input/reader thread.
    reader = detector.StreamBuffer.__new__(detector.StreamBuffer)
    reader._lock = threading.Lock()
    reader._ring_n = capacity
    reader._ring_elems = capacity * 2
    reader._ring = np.zeros(capacity * 2, dtype=np.int16)
    reader._scale = np.float32(1)
    reader._rp = consumed
    reader._wp = consumed if received is None else received
    reader._total_drops = 0
    reader._history_generation = 0
    start = max(0, reader._wp - capacity)
    for index in range(start, reader._wp):
        pos = index % capacity
        reader._ring[2 * pos] = index
        reader._ring[2 * pos + 1] = -index
    return reader


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    # Preserve this seam regression's explicit legacy-length fixture.
    monkeypatch.setenv('LORA_PAYLOAD_SYM_BUDGET', '120')
    monkeypatch.setenv('LORA_NB_PENDING', '0')
    monkeypatch.setattr(detector, '_IQ_INVERT', False)
    # Exercise real registration/feed/finalize; only the disk/DSP save sink is off.
    monkeypatch.setattr(detector.SignalRecorder, '_save_worker', lambda self: None)
    rec = detector.SignalRecorder(str(tmp_path), 1000, 915e6, 500)
    yield rec
    rec.flush_pending()
    for worker in rec._save_threads:
        worker.join(timeout=1)


def detection(sf=12, bw=62500):
    return {'sf': sf, 'bw': bw, 'freq_hz': 915035000.0,
            'peak_power_db': 20.0, 'detect_conf': 0.9, 'preamble_t_s': 0.0}


def saved_samples(recorder):
    item = recorder._save_queue.get_nowait()
    payload = item[0]
    assert payload[0] == 'PARTS'
    _, base, tail, needed = payload
    tail_iq = np.concatenate(tail)[:needed] if tail else np.empty(0, np.complex64)
    return np.concatenate(base + [tail_iq])


@pytest.mark.parametrize('delay_hops', [1, 14, 22])
def test_real_recorder_retains_full_frame_after_async_and_optional_hold_delay(recorder, delay_hops):
    base_end, hop = 4000, 500
    current = base_end + delay_hops * hop
    reader = ring_reader(current)
    anchor = (base_end, 0, 0)
    recent = deque((samples(end - hop, hop)
                    for end in range(max(base_end, current - 7 * hop), current + 1, hop)), maxlen=8)
    counters = reader.progress()
    detector._record_slow_capture(
        recorder, [detection()], samples(0, base_end), base_end,
        sample_pos=current, hop_n=hop, win_n=1000, recent_hops=recent,
        history_reader=reader, history_anchor=anchor)
    while recorder._pending:
        recorder.feed_tail(samples(current, hop))
        current += hop
    captured = saved_samples(recorder)
    expected_size = base_end + int(148.25 * 4096 / 62500 * 1000)
    assert len(captured) == expected_size
    np.testing.assert_array_equal(captured, samples(0, expected_size))
    assert len(captured) >= 8150  # complete ~8.15-second RF frame, not 4s base
    assert reader.progress() == counters  # historical recovery never consumes


def test_short_gap_still_uses_existing_hops_without_ring_retrieval(recorder, monkeypatch):
    reader = ring_reader(4500)
    monkeypatch.setattr(reader, 'history_parts', lambda *args: pytest.fail('Unnecessary ring copy'))
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=4500, hop_n=500, win_n=1000, recent_hops=[samples(4000, 500)],
        history_reader=reader, history_anchor=(4000, 0, 0))
    assert recorder._pending[0]['wb_n'] == 4500


@pytest.mark.parametrize('failure', ['overwritten', 'skipped'])
def test_unavailable_history_finalizes_only_contiguous_prefix(recorder, failure, capsys):
    reader = ring_reader(11000, received=25000 if failure == 'overwritten' else 11000)
    if failure == 'skipped':
        reader._total_drops = 500
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=11000, hop_n=500, win_n=1000,
        recent_hops=[samples(7000 + i * 500, 500) for i in range(8)],
        history_reader=reader, history_anchor=(4000, 0, 0))
    assert not recorder._pending
    recorder.feed_tail(samples(11000, 500))
    np.testing.assert_array_equal(saved_samples(recorder), samples(0, 4000))
    assert 'SLOW-GAP' in capsys.readouterr().err


def test_history_anchor_rejects_wrong_consumed_position():
    reader = ring_reader(11000)
    assert reader.history_anchor(11000) == (11000, 0, 0)
    assert reader.history_anchor(10500) is None


def test_history_rejects_retune_even_without_dropped_samples(recorder, capsys):
    reader = ring_reader(11000)
    reader.invalidate_history()
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=11000, hop_n=500, win_n=1000, recent_hops=[],
        history_reader=reader, history_anchor=(4000, 0, 0))
    assert not recorder._pending
    assert recorder._save_queue.empty()
    assert 'SLOW-GAP' in capsys.readouterr().err


def test_history_validates_again_after_copy(monkeypatch):
    reader = ring_reader(11000)
    convert = reader._convert_span
    def overwritten_during_copy(pos, count):
        result = convert(pos, count)
        reader._wp += 16000
        return result
    monkeypatch.setattr(reader, '_convert_span', overwritten_during_copy)
    with pytest.raises(detector.StreamHistoryUnavailable):
        list(reader.history_parts((4000, 0, 0), 7000, 500))
    assert reader._rp == 11000
    assert reader._total_drops == 0


@pytest.mark.parametrize('sf,bw,window_s,immediate', [
    (12, 62500, 1.0, True), (12, 62500, 2.0, False),
    (12, 125000, 1.0, False), (12, 125000, 0.25, True),
    (11, 62500, 1.0, False), (7, 500000, 1.0, False),
    (7, 31250, 4.0, True), (7, 41667, 4.0, True),
])
def test_immediate_recording_preserves_legacy_and_main_window_ineligible_modes(sf,bw,window_s,immediate):
    assert detector._slow_record_immediately(detection(sf,bw), window_s) == immediate


@pytest.fixture(params=[False, True], ids=['wideband', 'narrowband'])
def seeded_recorder(tmp_path, monkeypatch, request):
    monkeypatch.setenv('LORA_PAYLOAD_SYM_BUDGET', '120')
    monkeypatch.setenv('LORA_NB_PENDING', str(int(request.param)))
    monkeypatch.setattr(detector, '_IQ_INVERT', False)
    monkeypatch.setattr(detector.SignalRecorder, '_save_worker', lambda self: None)
    rec = detector.SignalRecorder(str(tmp_path), 1000, 915e6, 500)
    yield rec
    rec.flush_pending(wait=True)
    if rec._crop_queue is not None:
        rec._crop_queue.put(None)
        rec._crop_thread.join(timeout=2)
    for worker in rec._save_threads:
        worker.join(timeout=1)


def scaled_detection():
    # Scale BW and SF together to retain ~64ms symbols in a small real-DSP
    # fixture. The production NB setup/streaming filter remain unmodified.
    det = detection(sf=2, bw=62.5)
    det['freq_hz'] = 915e6
    return det


def assert_saved_prefix(rec, size):
    if not rec._nb_mode:
        np.testing.assert_array_equal(saved_samples(rec), samples(0, size))
        return
    rec._crop_queue.join()
    payload = rec._save_queue.get(timeout=2)[0]
    assert payload[0] == 'NB'
    reference = detector._StreamDecimator(0, 1000, 8)
    expected = np.concatenate([reference.feed(samples(0, size)), reference.finish()])
    assert payload[1]['nb_fs'] == 125
    np.testing.assert_allclose(payload[1]['streams'][0], expected, rtol=1e-6, atol=0.01)


def test_one_pass_history_seed_then_future_tail_through_real_cropper(seeded_recorder):
    rec = seeded_recorder
    reader = ring_reader(11000)
    detector._record_slow_capture(
        rec, [scaled_detection()], samples(0, 4000), 4000,
        sample_pos=11000, hop_n=500, win_n=1000, recent_hops=[],
        history_reader=reader, history_anchor=(4000, 0, 0))
    assert rec._pending[0]['got'] == 7000
    pos = 11000
    while rec._pending:
        rec.feed_tail(samples(pos, 500))
        pos += 500
    assert_saved_prefix(rec, 4000 + int(148.25 * 4 / 62.5 * 1000))


def test_mid_recovery_loss_keeps_only_contiguous_prefix(seeded_recorder, monkeypatch, capsys):
    rec = seeded_recorder
    reader = ring_reader(11000)
    convert = reader._convert_span
    def lose_second_hop(pos, count):
        if pos == 4500:
            reader._total_drops += 500
        return convert(pos, count)
    monkeypatch.setattr(reader, '_convert_span', lose_second_hop)
    detector._record_slow_capture(
        rec, [scaled_detection()], samples(0, 4000), 4000,
        sample_pos=11000, hop_n=500, win_n=1000, recent_hops=[],
        history_reader=reader, history_anchor=(4000, 0, 0))
    assert not rec._pending
    rec.feed_tail(samples(11000, 500))
    assert_saved_prefix(rec, 4500)
    assert '500/9488 tail samples' in capsys.readouterr().err


@pytest.mark.parametrize('seeded_recorder', [True], indirect=True)
def test_crop_queue_full_during_seed_never_appends_future(seeded_recorder, monkeypatch, capsys):
    rec = seeded_recorder
    enqueue = rec._crop_queue.put_nowait
    calls = 0
    def full_on_second_hop(item):
        nonlocal calls
        if item[0] == 'HOP':
            calls += 1
            if calls == 2:
                raise queue.Full
        enqueue(item)
    monkeypatch.setattr(rec._crop_queue, 'put_nowait', full_on_second_hop)
    rec.update_deferred([scaled_detection()], samples(0, 4000), 4000,
                        need_tail_n=9488, owned_buf=True,
                        tail_parts=(samples(i, 500) for i in range(4000, 11000, 500)))
    assert not rec._pending
    rec.feed_tail(samples(11000, 500))
    assert_saved_prefix(rec, 4500)
    assert '500/9488 tail samples' in capsys.readouterr().err


@pytest.mark.parametrize('seeded_recorder', [True], indirect=True)
def test_base_queue_full_falls_back_to_contiguous_wideband(seeded_recorder, monkeypatch):
    rec = seeded_recorder
    def full(item):
        raise queue.Full
    monkeypatch.setattr(rec._crop_queue, 'put_nowait', full)
    rec.update_deferred([scaled_detection()], samples(0, 4000), 4000,
                        need_tail_n=1000, owned_buf=True,
                        tail_parts=(samples(i, 500) for i in (4000, 4500)))
    assert not rec._pending
    np.testing.assert_array_equal(saved_samples(rec), samples(0, 5000))


@pytest.mark.parametrize('drops', [0, 500])
def test_ring_history_wrap_and_prior_drops_preserve_absolute_coordinates(drops):
    reader = ring_reader(20000)
    reader._total_drops = drops
    anchor = (15000, drops, 0)
    before = reader.progress()
    parts = list(reader.history_parts(anchor, 5000, 500))
    assert all(len(part) <= 500 for part in parts)
    np.testing.assert_array_equal(np.concatenate(parts), samples(15000, 5000))
    assert reader.progress() == before


def test_skip_invalidates_even_short_gap_owned_history(recorder, capsys):
    reader = ring_reader(5000)
    reader._total_drops = 500
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=4500, hop_n=500, win_n=1000,
        recent_hops=[samples(4500, 500)], history_reader=reader,
        history_anchor=(4000, 0, 0))
    assert not recorder._pending
    np.testing.assert_array_equal(saved_samples(recorder), samples(0, 4000))
    assert 'SLOW-GAP' in capsys.readouterr().err


def test_retune_with_clock_mismatch_discards_old_center_base(recorder, capsys):
    reader = ring_reader(11500)
    reader.invalidate_history()
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=11000, hop_n=500, win_n=1000, recent_hops=[],
        history_reader=reader, history_anchor=(4000, 0, 0))
    assert not recorder._pending
    assert recorder._save_queue.empty()
    assert 'retune' in capsys.readouterr().err


@pytest.mark.parametrize('seeded_recorder', [True], indirect=True)
def test_future_crop_queue_full_keeps_truthful_partial_count(seeded_recorder, monkeypatch, capsys):
    rec = seeded_recorder
    rec.update_deferred([scaled_detection()], samples(0, 4000), 4000,
                        need_tail_n=1000, owned_buf=True)
    job = rec._pending[0]
    def full(item):
        raise queue.Full
    monkeypatch.setattr(rec._crop_queue, 'put_nowait', full)
    rec.feed_tail(samples(4000, 500))
    assert not rec._pending
    assert job['got'] == 0
    assert_saved_prefix(rec, 4000)
    assert 'queue full' in capsys.readouterr().err


def test_misaligned_snapshot_truncates_even_if_later_clock_aligns(recorder, capsys):
    reader = ring_reader(11000)
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=11000, hop_n=500, win_n=1000, recent_hops=[],
        history_reader=reader, history_anchor=(4500, 0, 0))
    assert not recorder._pending
    np.testing.assert_array_equal(saved_samples(recorder), samples(0, 4000))
    assert 'SLOW-GAP' in capsys.readouterr().err


def test_manual_skip_invalidates_nonconsuming_history():
    reader = ring_reader(4500, received=5500)
    anchor = reader.history_anchor(4500)
    assert reader.skip_to_latest(500) == 500
    before = reader.progress()
    with pytest.raises(detector.StreamHistoryUnavailable, match='skipped'):
        list(reader.history_parts(anchor, 500, 500))
    assert reader.progress() == before
