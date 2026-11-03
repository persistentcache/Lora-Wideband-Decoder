"""Supported long frames must survive both recording and save-worker trimming."""
import ast
import os
from pathlib import Path

import numpy as np
import pytest
import detector


@pytest.fixture
def nb_recorder(tmp_path, monkeypatch):
    monkeypatch.setenv('LORA_NB_PENDING', '1')
    monkeypatch.delenv('LORA_PAYLOAD_SYM_BUDGET', raising=False)
    monkeypatch.setenv('LORA_EXPORT_DEC', '2')
    monkeypatch.setattr(detector.SignalRecorder, '_save_worker', lambda self: None)
    rec = detector.SignalRecorder(str(tmp_path), 250000, 915e6, 125000)
    yield rec
    rec.flush_pending(wait=True)
    rec._crop_queue.put(None)
    rec._crop_thread.join(timeout=2)
    for worker in rec._save_threads:
        worker.join(timeout=1)


def detection(sf=12, bw=62500):
    return {'sf': sf, 'bw': bw, 'freq_hz': 915e6, 'peak_power_db': 20,
            'detect_conf': .9, 'preamble_t_s': 2.686976}


@pytest.fixture(scope='module')
def save_trim():
    # Run the actual save-worker sizing and NB slice, without its unrelated
    # carrier/dedup/file machinery. This catches a second cutoff after capture.
    tree = ast.parse(Path(detector.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SignalRecorder')
    worker = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_save_worker')
    for node in ast.walk(worker):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        starts = [i for i, child in enumerate(body)
                  if isinstance(child, ast.Assign)
                  and '_cap_bw, sym_time' in ast.unparse(child.targets[0])]
        if starts:
            start = starts[0]
            end = next(i for i in range(start, len(body))
                       if isinstance(body[i], ast.Assign)
                       and ast.unparse(body[i].targets[0]) == 'extended')
            sizing = body[start:end + 1]
            break
    trim = next(n for n in ast.walk(worker) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == 'nb'
                and 'buf_dur_s + max_pkt_s' in ast.unparse(n.value))
    code = compile(ast.Module(body=sizing + [trim], type_ignores=[]),
                   detector.__file__, 'exec')
    def apply(rec, nb):
        env = dict(vars(detector), self=rec, anchor=detection(),
                   detections=[detection()], pre_hop_s=0,
                   wb_n=4 * rec.wb_fs, _nb_job={}, extended_full=None,
                   nb_fs=125000, nb=nb)
        exec(code, env)
        return env['nb']
    return apply


def test_pl97_frame_survives_real_nb_tail_and_save_trim(nb_recorder, save_trim):
    rec = nb_recorder
    base = np.ones(4 * rec.wb_fs, np.complex64)
    detector._record_slow_capture(
        rec, [detection()], base, len(base), sample_pos=len(base),
        hop_n=rec.hop_n, win_n=rec.wb_fs, recent_hops=[])
    hop = np.ones(rec.hop_n, np.complex64)
    while rec._pending:
        rec.feed_tail(hop)
        rec._crop_queue.join()
    rec._crop_queue.join()
    payload = rec._save_queue.get(timeout=2)[0]
    assert payload[0] == 'NB'
    retained = payload[1]['streams'][0]
    # Actual audited PL97/CR4/8 starts2.686976s in and ends15.024128s in.
    required_end = int(15.024128 * payload[1]['nb_fs'])
    assert len(retained) >= required_end
    # The supported PL237/CR4/8/32-preamble bound exceeds the old24s cap.
    expected_max = round((4 + 28.065792) * payload[1]['nb_fs'])
    assert len(retained) == expected_max
    saved = save_trim(rec, retained)
    assert len(saved) == expected_max
    assert np.all(np.abs(saved[required_end - 100:required_end - 50]) > .9)


def test_save_worker_does_not_retruncate_sufficient_pl97_iq(nb_recorder, save_trim):
    available = np.ones(16 * 125000, np.complex64)
    saved = save_trim(nb_recorder, available)
    assert len(saved) >= int(15.024128 * 125000)


@pytest.mark.parametrize('sf,bw,seconds', [
    (12, 62500, 28.065792), (12, 31250, 56.131584),
    (11, 125000, 7.671808), (10, 125000, 4.294656),
    (7, 500000, .207936),
])
def test_supported_max_includes_32_symbol_preamble_cr8_and_forced_ldro(monkeypatch, sf, bw, seconds):
    monkeypatch.delenv('LORA_PAYLOAD_SYM_BUDGET', raising=False)
    assert detector._capture_duration_s(detection(sf, bw)) == pytest.approx(seconds)


@pytest.mark.parametrize('budget,seconds', [(100, 8.404992), (120, 9.715712), (160, 12.337152)])
def test_explicit_symbol_override_controls_tail_and_save(nb_recorder, save_trim, monkeypatch, budget, seconds):
    rec = nb_recorder
    monkeypatch.setenv('LORA_PAYLOAD_SYM_BUDGET', str(budget))
    detector._record_slow_capture(
        rec, [detection()], np.ones(rec.wb_fs * 4, np.complex64), rec.wb_fs * 4,
        sample_pos=rec.wb_fs * 4, hop_n=rec.hop_n, win_n=rec.wb_fs,
        recent_hops=[])
    assert rec._pending[0]['need'] == round(seconds * rec.wb_fs)
    saved = save_trim(rec, np.ones(32 * 125000, np.complex64))
    assert len(saved) == round((4 + seconds) * 125000)


def test_patience_alias_budget_covers_each_actual_family_member(monkeypatch):
    monkeypatch.delenv('LORA_PAYLOAD_SYM_BUDGET', raising=False)
    trial = dict(detection(10, 62500), patience_trial=True)
    duration = detector._capture_duration_s(trial)
    family = detector.patience_trial_family(10, 62500)
    assert len(family) > 1
    assert duration == max(detector._capture_duration_s(detection(sf, bw))
                           for sf, bw in family)


@pytest.mark.parametrize('failure', ['queue_full', 'degenerate', 'disabled'])
def test_actual_wideband_fallback_keeps_old_bound(nb_recorder, monkeypatch, capsys, failure):
    import queue
    rec = nb_recorder
    if failure == 'queue_full':
        def full(item):
            raise queue.Full
        monkeypatch.setattr(rec._crop_queue, 'put_nowait', full)
    elif failure == 'degenerate':
        rec.wb_fs = 1000  # D<2: actual _nb_setup returns None
    else:
        rec._nb_mode = False
    requested = round(28.065792 * rec.wb_fs)
    rec.update_deferred([detection()], np.ones(1, np.complex64), 1,
                        need_tail_n=requested, owned_buf=True)
    job = rec._pending[0]
    assert 'nb' not in job
    assert job['need'] == int(9.715712 * rec.wb_fs)
    assert 'CAPTURE-LIMIT' in capsys.readouterr().err


def test_missing_scipy_fallback_retains_bound(nb_recorder, tmp_path, monkeypatch, capsys):
    import builtins
    original_import = builtins.__import__
    def import_without_scipy(name, *args, **kwargs):
        if name == 'scipy.signal':
            raise ImportError('controlled unavailable NB dependency')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', import_without_scipy)
    rec = detector.SignalRecorder(str(tmp_path / 'no_scipy'), 1000, 915e6, 500)
    assert not rec._nb_mode
    rec.update_deferred([detection()], np.ones(1, np.complex64), 1,
                        need_tail_n=28066, owned_buf=True)
    assert rec._pending[0]['need'] == 9715
    assert 'CAPTURE-LIMIT' in capsys.readouterr().err
    rec.flush_pending()
    for worker in rec._save_threads:
        worker.join(timeout=1)


def test_wideband_window_ceiling_and_gap_accounting(nb_recorder, capsys):
    rec = nb_recorder
    rec._nb_mode = False
    rec.wb_fs = 1000
    rec.window_n = 250  # 24 windows=6 seconds, below9.715s legacy airtime
    rec.update_deferred([detection()], np.ones(4000, np.complex64), 5000,
                        need_tail_n=27066, owned_buf=True,
                        gap_parts=[np.ones(1000, np.complex64)])
    assert rec._pending[0]['need'] == 5000
    assert rec._pending[0]['wb_n'] == 5000
    assert 'CAPTURE-LIMIT' in capsys.readouterr().err


def production_main():
    tree = ast.parse(Path(detector.__file__).read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')


@pytest.mark.parametrize('blocking', [False, True])
def test_real_inline_request_preserves_blocking_read_bound(nb_recorder, monkeypatch, capsys, blocking):
    if blocking:
        monkeypatch.setenv('LORA_NO_DEFER', '1')
    else:
        monkeypatch.delenv('LORA_NO_DEFER', raising=False)
    branch = next(n for n in ast.walk(production_main()) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'recorder and is_live and dets')
    request = next(n for n in branch.body if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == '_max_tail_n')
    env = dict(vars(detector), recorder=nb_recorder, dets=[detection()], win_n=250000)
    exec(compile(ast.Module(body=[request], type_ignores=[]), detector.__file__, 'exec'), env)
    expected = 9.715712 if blocking else 28.065792
    assert env['_max_tail_n'] == round(expected * nb_recorder.wb_fs)
    assert ('CAPTURE-LIMIT' in capsys.readouterr().err) == blocking


def test_pool_commit_without_trailing_window_keeps_future_nb_tail(nb_recorder):
    from collections import deque
    from types import SimpleNamespace
    main = production_main()
    commit = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef)
                  and n.name == '_commit_oldest')
    record = next(n for n in commit.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'recorder')
    rec = nb_recorder
    base = np.ones(rec.wb_fs, np.complex64)
    env = dict(vars(detector), recorder=rec, dets0=[detection()], win_n=rec.wb_fs,
               _inflight=deque(), _pool=SimpleNamespace(slot_array=lambda slot: base),
               it={'slot': 0, 'tot_s': len(base), 'pre_hop': None})
    exec(compile(ast.Module(body=[record], type_ignores=[]), detector.__file__, 'exec'), env)
    assert len(rec._pending) == 1
    assert rec._pending[0]['need'] == round(28.065792 * rec.wb_fs)
    assert rec._pending[0]['got'] == 0


def test_pool_tail_window_policy_handles_no_recorder():
    from types import SimpleNamespace
    main = production_main()
    function = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef)
                    and n.name == '_tail_windows_needed')
    env = dict(vars(detector), recorder=None, a=SimpleNamespace(rate=250000),
               win_n=250000, _tail_seg_n=125000)
    exec(compile(ast.Module(body=[function], type_ignores=[]), detector.__file__, 'exec'), env)
    assert env['_tail_windows_needed']([detection()]) == 0
