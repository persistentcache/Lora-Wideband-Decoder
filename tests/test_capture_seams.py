"""Production registration order must not append the current hop twice."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import detector


def samples(start, count):
    values = np.arange(start, start + count, dtype=np.float32)
    return (values - 1j * values).astype(np.complex64)


def detection():
    # Time-scaled symbols keep real NB filtering small (D8,64ms symbols).
    return {'sf': 2, 'bw': 62.5, 'freq_hz': 915e6,
            'peak_power_db': 20, 'detect_conf': .9}


@pytest.fixture(params=[False, True], ids=['WB', 'NB'])
def recorder(tmp_path, monkeypatch, request):
    monkeypatch.setenv('LORA_NB_PENDING', str(int(request.param)))
    monkeypatch.setenv('LORA_EXPORT_DEC', '2')
    monkeypatch.setenv('LORA_PAYLOAD_SYM_BUDGET', '120')
    monkeypatch.setattr(detector.SignalRecorder, '_save_worker', lambda self: None)
    rec = detector.SignalRecorder(str(tmp_path), 1000, 915e6, 500)
    yield rec
    rec.flush_pending(wait=True)
    if rec._crop_queue is not None:
        rec._crop_queue.put(None)
        rec._crop_thread.join(timeout=2)
    for worker in rec._save_threads:
        worker.join(timeout=1)


@pytest.fixture(scope='module')
def production():
    tree = ast.parse(Path(detector.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    inline = next(n for n in ast.walk(main) if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call)
                  and ast.unparse(n.value.func) == 'recorder.update_deferred'
                  and any(k.arg == 'need_tail_n' and ast.unparse(k.value) == '_max_tail_n'
                          for k in n.value.keywords))
    feed = next(n for n in ast.walk(main) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == 'recorder.feed_tail')
    commit = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef)
                  and n.name == '_commit_oldest')
    pool = next(n for n in commit.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'recorder')
    def compile_block(node):
        return compile(ast.Module(body=[node], type_ignores=[]), detector.__file__, 'exec')
    return {name: compile_block(node) for name, node in [('inline', inline), ('feed', feed), ('pool', pool)]}


def feed_loop(code, env, start, count=500):
    env.update(_hop_own=samples(start, count), tot_s=start + count)
    exec(code, env)
    rec = env['recorder']
    if rec._crop_queue is not None:
        rec._crop_queue.join()


def assert_saved_contiguous(rec, count):
    if rec._crop_queue is not None:
        rec._crop_queue.join()
    payload = rec._save_queue.get(timeout=2)[0]
    expected = samples(0, count)
    if payload[0] == 'PARTS':
        _, base, tail, need = payload
        actual = np.concatenate(base + ([np.concatenate(tail)[:need]] if tail else []))
        np.testing.assert_array_equal(actual, expected)
    else:
        assert payload[0] == 'NB'
        ref = detector._StreamDecimator(0, 1000, 8)
        expected_nb = np.concatenate([ref.feed(expected), ref.finish()])
        np.testing.assert_allclose(payload[1]['streams'][0], expected_nb, rtol=1e-6, atol=.01)


def test_actual_inline_registration_then_current_loop_feed_has_no_duplicate(recorder, production):
    env = dict(vars(detector), recorder=recorder, dets=[detection()],
               buf=samples(0, 1000), tot_s=1000, pre_hop=None, _max_tail_n=1000)
    exec(production['inline'], env)
    # Main registers from a window ending1000 BEFORE feeding its current hop.
    feed_loop(production['feed'], env, 500)
    feed_loop(production['feed'], env, 1000)
    feed_loop(production['feed'], env, 1500)
    assert not recorder._pending
    assert_saved_contiguous(recorder, 2000)


@pytest.mark.parametrize('hop', [250, 500, 750], ids=['75pct_overlap', '50pct_overlap', '25pct_overlap'])
@pytest.mark.parametrize('last_seed_seq', [1, 2], ids=['current_hop_needed', 'current_hop_already_seeded'])
def test_actual_pool_commit_and_loop_feed_keep_contiguous_seed_seam(recorder, production, last_seed_seq, hop):
    windows = {i: samples(i * hop, 1000) for i in range(3)}
    recorder.hop_n = hop
    det = detection()
    needed = 9488
    if not recorder._nb_mode:
        # Exercise the real pool's NB request with an actual D<2 fallback,
        # rather than the separate eager-WB pool path (which never pends).
        recorder._nb_mode = True
        det.update(sf=12, bw=62500)
        needed = 9715
    env = dict(vars(detector), recorder=recorder, dets0=[det], win_n=1000,
               hop_n=hop, _inflight=deque({'seq': i, 'slot': i}
                                        for i in range(1, last_seed_seq + 1)),
               _pool=SimpleNamespace(slot_array=lambda slot: windows[slot]),
               it={'seq': 0, 'slot': 0, 'tot_s': 1000, 'pre_hop': None})
    exec(production['pool'], env)
    assert recorder._pending[0]['got'] == last_seed_seq * hop
    # A slot-full commit seeded only through the prior window needs this hop;
    # an ordinary commit seeded through the current window already has it.
    feed_loop(production['feed'], env, 1000 + hop, hop)
    start = 1000 + 2 * hop
    while recorder._pending:
        feed_loop(production['feed'], env, start, hop)
        start += hop
    assert_saved_contiguous(recorder, 1000 + needed)


def test_partial_overlap_only_appends_new_samples(recorder):
    recorder.update_deferred([detection()], samples(0, 1000), 1000,
                             need_tail_n=1000, owned_buf=True)
    recorder.feed_tail(samples(750, 500), sample_pos=1250)
    assert recorder._pending[0]['got'] == 250
    recorder.feed_tail(samples(1250, 500), sample_pos=1750)
    recorder.feed_tail(samples(1750, 500), sample_pos=2250)
    assert not recorder._pending
    assert_saved_contiguous(recorder, 2000)


def test_missing_future_hop_finalizes_prefix_without_splicing(recorder, capsys):
    recorder.update_deferred([detection()], samples(0, 1000), 1000,
                             need_tail_n=2000, owned_buf=True)
    job = recorder._pending[0]
    recorder.feed_tail(samples(1000, 500), sample_pos=1500)
    recorder.feed_tail(samples(2000, 500), sample_pos=2500)
    assert not recorder._pending
    assert job['got'] == 500
    recorder.feed_tail(samples(2500, 500), sample_pos=3000)
    assert_saved_contiguous(recorder, 1500)
    assert 'missing 500 samples' in capsys.readouterr().err


def test_pre_hop_changes_base_start_but_not_next_tail_position(recorder, production):
    env = dict(vars(detector), recorder=recorder, dets=[detection()],
               buf=samples(500, 1000), tot_s=1500,
               pre_hop=samples(0, 500), _max_tail_n=500)
    exec(production['inline'], env)
    feed_loop(production['feed'], env, 1000)
    assert recorder._pending[0]['got'] == 0
    feed_loop(production['feed'], env, 1500)
    assert_saved_contiguous(recorder, 2000)


def test_slow_registration_after_loop_feed_keeps_next_future_hop(recorder, production):
    env = dict(vars(detector), recorder=recorder)
    feed_loop(production['feed'], env, 4500)  # no job exists yet
    detector._record_slow_capture(
        recorder, [detection()], samples(0, 4000), 4000,
        sample_pos=5000, hop_n=500, win_n=1000,
        recent_hops=[samples(4000, 500), samples(4500, 500)])
    start = 5000
    while recorder._pending:
        feed_loop(production['feed'], env, start)
        start += 500
    assert_saved_contiguous(recorder, 4000 + 9488)


def test_explicit_stream_break_still_prevents_splice_when_consumed_clock_hides_drop(recorder, capsys):
    recorder.update_deferred([detection()], samples(0, 1000), 1000,
                             need_tail_n=2000, owned_buf=True)
    recorder.feed_tail(samples(1000, 500), sample_pos=1500)
    recorder.break_pending('controlled physical sample drop')
    # The consumed clock excludes drops, so only break_pending can identify
    # this missing physical interval. Main-loop skip calls must remain.
    recorder.feed_tail(samples(2000, 500), sample_pos=2000)
    assert_saved_contiguous(recorder, 1500)
    assert 'physical sample drop' in capsys.readouterr().err


@pytest.mark.parametrize('hop', [250, 500, 750])
def test_pool_tail_window_count_matches_new_samples_per_window(recorder, hop):
    tree = ast.parse(Path(detector.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assignment = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == '_tail_seg_n')
    function = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef)
                    and n.name == '_tail_windows_needed')
    env = dict(vars(detector), recorder=recorder, win_n=1000, hop_n=hop)
    exec(compile(ast.Module(body=[assignment, function], type_ignores=[]),
                 detector.__file__, 'exec'), env)
    needed = detector._capture_tail_n(recorder, [detection()], 1000, warn=False)
    assert env['_tail_windows_needed']([detection()]) == -(-needed // hop)
