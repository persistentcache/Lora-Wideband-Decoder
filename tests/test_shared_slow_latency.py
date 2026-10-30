"""Exercise main's slow-result routing without RX, FFTs, or large IQ storage."""
import ast
from collections import deque
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import detector
from capture_timeline import CaptureBudget, SharedCapture


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def compile_node(node):
    return compile(ast.fix_missing_locations(ast.Module([node], type_ignores=[])),
                   detector.__file__, 'exec')


@pytest.fixture(scope='module')
def production():
    main = next(node for node in ast.parse(Path(detector.__file__).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    routes = [node for node in ast.walk(main) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'recorder'
              and any(assigns(child, '_slow_now') for child in ast.walk(node))]
    assert len(routes) == 1
    held = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                and ast.unparse(node.test) == '_slow_hold and recorder')
    # Execute production statements, including the real capture helper for
    # shared routing; do not duplicate the proposed branch in this fixture.
    return dict(route=compile_node(routes[0]), held=compile_node(held))


def routing_state(shared, sf, bw):
    env, admitted = {}, []
    recorder = SimpleNamespace(
        _shared=object() if shared else None, _nb_mode=True, wb_fs=10_000_000,
        _saved_abs_t=[], _saved_abs_t_lock=threading.Lock(),
        update_deferred=lambda *args, **kwargs:
            admitted.append((env['_slow_tick'], args, kwargs)))
    detection = dict(sf=sf, bw=bw, freq_hz=915_003_600., preamble_t_s=.25,
                     detect_conf=.95, peak_power_db=30.)
    # Represent a real four-second base with an eight-byte backing allocation.
    # Shared routing must pass this object through without enumerating a tail.
    base = np.broadcast_to(np.array([1 + 2j], np.complex64), (40_000_000,))
    anchor = (125_000_000, 5_000_000, 7)
    history = SimpleNamespace(history_parts=lambda *args, **kwargs:
                              pytest.fail('shared routing copied raw history'))
    env.update(vars(detector))
    env.update(recorder=recorder, dets_slow=[detection], _iq_long=base,
               _reg_tot_s=120_000_000, _history_done=anchor,
               tot_s=145_000_000, hop_n=5_000_000, win_n=10_000_000,
               _slow_halves=deque(maxlen=8), _slow_hold=[], _slow_tick=50,
               _LAG=4, reader=history, is_live=True,
               a=SimpleNamespace(rate=10_000_000),
               _source_before_retune=lambda source: False)
    return env, admitted


@pytest.fixture
def mixed_shared(monkeypatch):
    """Real recorder admission/ledger; omit only reader threads and crop work."""
    monkeypatch.delenv('LORA_PAYLOAD_SYM_BUDGET', raising=False)
    monkeypatch.setenv('LORA_EXPORT_DEC', '2')
    env, _ = routing_state(True, 12, 62_500)
    immediate = dict(env['dets_slow'][0], peak_power_db=20.)
    delayed = dict(immediate, sf=11, bw=125_000, peak_power_db=30.)
    assert detector._slow_record_immediately(immediate, 1.)
    assert not detector._slow_record_immediately(delayed, 1.)
    history = SimpleNamespace(history_anchor=lambda sample_pos=None: env['_history_done'])
    capture = SharedCapture(
        rate=10_000_000, center=915e6, hop_n=5_000_000, reader=None,
        decimator_factory=detector._StreamDecimator,
        submit=lambda view: False, budget=CaptureBudget(3 * 1024**3, 8),
        threaded=False)
    recorder = detector.SignalRecorder.__new__(detector.SignalRecorder)
    recorder._shared, recorder._history_reader = capture, history
    recorder._nb_mode, recorder.wb_fs = True, 10_000_000
    env.update(recorder=recorder, reader=history, dets_slow=[immediate, delayed])
    try:
        yield env, capture
    finally:
        capture._fail('test teardown')
        capture.close(wait=False)
        assert capture.budget.stats()['used_bytes'] == 0
        assert capture.budget.stats()['outbox_slots'] == 0


@pytest.mark.parametrize('separation', [0., 5_000.], ids=['same-carrier', 'nearby-carrier'])
def test_mixed_slow_partitions_keep_both_labels_through_real_admission(
        production, mixed_shared, separation):
    env, capture = mixed_shared
    env['dets_slow'][1]['freq_hz'] += separation

    exec(production['route'], env)

    labels = [(d['sf'], d['bw']) for view in capture._views for d in view.detections]
    assert labels == [(12, 62_500), (11, 125_000)], (
        'merging formerly separate groups lets the stronger label evict the weaker one')
    assert len(capture._views) == 2
    assert env['_slow_tick'] == 50 and not env['_slow_hold']
    assert not capture.failure


def test_separate_carriers_keep_each_partition_tail_and_export_geometry(
        production, mixed_shared):
    env, capture = mixed_shared
    env['dets_slow'][1]['freq_hz'] += 20_000.
    tails = [detector._capture_tail_n(env['recorder'], [d], env['win_n'])
             for d in env['dets_slow']]
    assert tails[0] > tails[1]

    exec(production['route'], env)

    assert len(capture._views) == 2, 'distinct groups inherited one strongest export geometry'
    for view, detection, tail, factor in zip(
            capture._views, env['dets_slow'], tails, [80, 40]):
        assert len(view.detections) == 1
        admitted = view.detections[0]
        assert (admitted['sf'], admitted['bw'], admitted['freq_hz']) == (
            detection['sf'], detection['bw'], detection['freq_hz'])
        assert view.start == 85_000_000
        assert view.metadata['base_end'] == 125_000_000
        assert view.metadata['sample_pos'] == env['_reg_tot_s']
        assert view.metadata['identity'] == env['_history_done'][1:]
        assert view.end == 125_000_000 + tail
        assert view.target_factor == factor
        assert view.export_bw == detection['bw']
        assert view.nb_fs == 2 * detection['bw']
        assert view.role == 'shared_full'
    assert not env['_slow_hold'] and not capture.failure


@pytest.mark.parametrize('sf,bw', [(11, 500_000), (7, 250_000)],
                         ids=['CUSTOM_SF11_500', 'SHORT_FAST'])
def test_shared_slow_result_admits_same_turn_with_original_span_and_full_tail(
        production, monkeypatch, sf, bw):
    monkeypatch.delenv('LORA_PAYLOAD_SYM_BUDGET', raising=False)
    env, admitted = routing_state(True, sf, bw)
    # The existing airtime builder remains authoritative; routing may not
    # replace its full fallback tail with the early-header endpoint or gap.
    expected_tail = detector._capture_tail_n(
        env['recorder'], env['dets_slow'], env['win_n'])
    assert expected_tail > 0

    exec(production['route'], env)

    assert len(admitted) == 1, 'shared slow result waited in the legacy hold'
    tick, args, kwargs = admitted[0]
    assert tick == env['_slow_tick'] == 50
    assert args[0] == env['dets_slow']
    assert args[0][0] is env['dets_slow'][0]
    assert args[1] is env['_iq_long']
    assert args[2] == env['_reg_tot_s'] == 120_000_000
    assert args[2] - len(args[1]) == 80_000_000
    assert kwargs == dict(pre_hop=None, need_tail_n=expected_tail, owned_buf=True,
                          history_anchor=env['_history_done'])
    assert kwargs['history_anchor'] is env['_history_done']
    assert kwargs['history_anchor'][0] - len(args[1]) == 85_000_000
    assert not env['_slow_hold']


@pytest.mark.parametrize('sf,bw', [(11, 500_000), (7, 250_000)],
                         ids=['CUSTOM_SF11_500', 'SHORT_FAST'])
@pytest.mark.parametrize('lag,due', [(0, 56), (4, 58), (12, 58)])
def test_legacy_slow_result_keeps_hold_until_existing_due_tick(
        production, sf, bw, lag, due):
    env, admitted = routing_state(False, sf, bw)
    calls = []
    # This test observes the legacy route boundary. Its history reconstruction
    # is covered by test_slow_capture.py with actual populated StreamBuffers.
    env.update(_LAG=lag, _record_slow_capture=lambda *args, **kwargs:
               calls.append((env['_slow_tick'], args, kwargs)))

    exec(production['route'], env)

    assert not calls and not admitted
    assert len(env['_slow_hold']) == 1
    held = env['_slow_hold'][0]
    assert held[0] == due
    assert held[1] == env['dets_slow']
    assert held[2] is env['_iq_long']
    assert held[3] == env['_reg_tot_s']
    assert held[4] is env['_history_done']
    env['_slow_tick'] = due - 1
    exec(production['held'], env)
    assert not calls and len(env['_slow_hold']) == 1

    env['_slow_tick'] = due
    exec(production['held'], env)

    assert not env['_slow_hold'] and len(calls) == 1
    tick, args, kwargs = calls[0]
    assert tick == due
    assert args[0] is env['recorder']
    assert args[1] == env['dets_slow']
    assert args[2] is env['_iq_long']
    assert args[3] == env['_reg_tot_s']
    assert kwargs == dict(sample_pos=env['tot_s'], hop_n=env['hop_n'],
                          win_n=env['win_n'], recent_hops=env['_slow_halves'],
                          history_reader=env['reader'],
                          history_anchor=env['_history_done'])
