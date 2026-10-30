"""Execute real detector-loop retune boundaries without starting RX or DSP."""
import ast
from collections import deque
import copy
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import detector
from capture_timeline import CaptureBudget, SharedCapture
from test_capture_timeline import reader


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def compile_nodes(nodes):
    return compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])),
                   detector.__file__, 'exec')


@pytest.fixture(scope='module')
def production():
    main = next(node for node in ast.parse(Path(detector.__file__).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While)
                and any(assigned(child, 'dets') for child in node.body))
    control = next(node for node in loop.body if isinstance(node, ast.If)
                   and '_ctl_path' in ast.unparse(node.test))
    slide_start = next(i for i, node in enumerate(loop.body)
                       if isinstance(node, ast.If) and ast.unparse(node.test) == '_n_iq >= win_n')
    slide_end = next(i for i, node in enumerate(loop.body)
                     if isinstance(node, ast.If) and 'buf_pos < win_n' in ast.unparse(node.test))
    # One original main-loop turn: a real warmup continue exits this singleton
    # loop, while reaching the detection boundary sets an observable flag.
    slide = ast.For(target=ast.Name('_one_hop', ast.Store()),
                    iter=ast.Tuple([ast.Constant(None)], ast.Load()),
                    body=copy.deepcopy(loop.body[slide_start:slide_end + 1])
                         + ast.parse('reached_detection = True').body, orelse=[])
    commit = copy.deepcopy(next(node for node in ast.walk(main)
                                if isinstance(node, ast.FunctionDef)
                                and node.name == '_commit_oldest'))
    commit.body = [node for node in commit.body if not isinstance(node, ast.Nonlocal)]
    # The extracted function still mutates the original loop's total counter.
    commit.body.insert(1, ast.Global(['tot_d']))
    helpers = [node for node in main.body if isinstance(node, ast.FunctionDef)
               and node.name == '_source_before_retune']
    slow = next(node for node in loop.body if isinstance(node, ast.While)
                and '_slow_results.empty()' in ast.unparse(node.test))
    held = next(node for node in loop.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == '_slow_hold and recorder')
    dispatch_anchor = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                           and isinstance(node.value, ast.Call)
                           and ast.unparse(node.value.func) == '_inflight.append')
    return dict(control=compile_nodes([control]), slide=compile_nodes([slide]),
                functions=compile_nodes(helpers + [commit]), slow=compile_nodes([slow]),
                held=compile_nodes([held]), dispatch_anchor=compile_nodes([dispatch_anchor]))


@pytest.fixture
def state(tmp_path):
    history = reader(capacity=12000)
    history._rp, history._wp, history._total_drops = 4000, 5500, 500
    capture = SharedCapture(
        rate=1000, center=915e6, hop_n=500, reader=history,
        decimator_factory=detector._StreamDecimator, submit=lambda view: False,
        budget=CaptureBudget(2_000_000, 8), threaded=False)
    recorder = detector.SignalRecorder.__new__(detector.SignalRecorder)
    recorder._shared, recorder._history_reader = capture, history
    recorder.center_hz = 915e6
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'center_hz': 915250000}))
    env = dict(vars(detector), reader=history, recorder=recorder, is_live=True,
               _ctl_path=str(control), _ctl_mtime=None, _last_ctl_poll=0.,
               _t_step=10., _agc=None, _live_center_mhz=915., _history_floor=0,
               _lz=None, buf=np.full(1000, 99 + 99j, np.complex64), buf_pos=1000,
               win_n=1000, hop_n=500, pre_hop=np.ones(500, np.complex64),
               _carry_tail=np.ones(50, np.complex64), _prehop_hist=deque([(1, 'old')]),
               _slice_cache={1: 'old'}, _slow_halves=deque([np.ones(500)], maxlen=8),
               _slow_hold=[('old',)], _slow_pending={1: 'old'}, _slow_seeds={1: 5},
               _slow_exact={1: 8}, _slow_backoff={1: 4}, _slow_fruitless={1: 3},
               _slow_last_fire=40, _slow_tick=42, _slow_busy=[True],
               _seq=7, tot_s=3500, tot_d=2, _inflight=deque([{'seq': 6, 'slot': 0}]),
               _pat_hits=np.ones(4), _pat_seen=np.ones(4), _pat_promoted={1: 1},
               _pat_futile={1: 1}, _pat_confirmed={1: 1}, _pat_cooldown={1: 1},
               PRESCREEN_ON=True, _jk_hits=np.ones(4), _jk_elev=np.ones(4),
               _jk_wid=np.ones(4), _jk_seen=1., _jk_junk={1: 1},
               _jk_det_bins={1}, _jk_verify_recent={1: 1},
               _max_pre_hop_n=500, _prehop_ref_on=True, _prehop_verify=False,
               _prehop_ref_hits=0, _prehop_copies=0)
    yield env
    capture._fail('test teardown')
    capture.close(wait=False)


@pytest.mark.parametrize('lazy', [False, True])
def test_control_retune_resets_history_without_releasing_running_workers(production, state, lazy):
    if lazy:
        state['_lz'] = detector._LazyWindow(1000, True)
        state['_lz'].set_full(np.full(2000, 99, np.int16))
        state['buf'] = None
    original_inflight = state['_inflight'][0]
    exec(production['control'], state)
    assert state['recorder']._shared.center == 915250000
    assert state['reader']._history_generation == 1
    assert state['_history_floor'] == 5000  # received raw 5500 minus 500 drops
    assert state['buf_pos'] == 0
    assert state['pre_hop'] is state['_carry_tail'] is None
    actual = state['_lz'].materialize() if lazy else state['buf']
    np.testing.assert_array_equal(actual, np.zeros(1000, np.complex64))
    for name in ('_prehop_hist', '_slice_cache', '_slow_halves', '_slow_hold',
                 '_slow_pending', '_slow_seeds', '_slow_exact', '_slow_backoff',
                 '_slow_fruitless'):
        assert not state[name], name
    assert state['_slow_last_fire'] < state['_slow_tick']
    assert state['_slow_busy'] == [True]
    assert list(state['_inflight']) == [original_inflight]
    assert (state['tot_s'], state['_seq'], state['_slow_tick']) == (3500, 7, 42)


@pytest.mark.parametrize('lazy', [False, True])
def test_new_window_and_lookback_exclude_pre_retune_backlog(production, state, lazy):
    if lazy:
        state['_lz'] = detector._LazyWindow(1000, True)
        state['buf'] = None
    state.update(buf_pos=0, _history_floor=5000, _prehop_hist=deque(),
                 _carry_tail=None, pre_hop=None)
    # There are 1500 consumed-clock samples of old receive backlog at reset.
    # Even a full-sized slide stays in warmup until it contains only fresh IQ.
    for end in range(4000, 6501, 500):
        value = 3 if end <= 5000 else 7
        state.update(tot_s=end, _n_iq=500, reached_detection=False,
                     iq=np.full(1000 if lazy else 500, value,
                                np.int16 if lazy else np.complex64))
        exec(production['slide'], state)
        assert state['reached_detection'] == (end >= 6000)
        if end == 6000:
            assert state['pre_hop'] is None
        if end == 6500:
            assert len(state['pre_hop']) == 500
            # The production CS16 receiver uses the B200's 12-bit scale.
            expected = (7 + 7j) / 2048 if lazy else 7
            np.testing.assert_array_equal(state['pre_hop'],
                                          np.full(500, expected, np.complex64))


def detection():
    return dict(freq_hz=915e6, freq_mhz=915., sf=7, bw=125000,
                detect_conf=.99, peak_power_db=20, preamble_t_s=.1)


@pytest.mark.parametrize('generation, accepted', [(0, False), (1, True)])
def test_pool_result_keeps_slot_until_worker_finishes_then_checks_generation(
        production, state, generation, accepted, capsys):
    state['reader']._history_generation = 1
    entered, finish = threading.Event(), threading.Event()
    releases, observed, saved, errors = [], [], [], []
    def result(seq):
        assert seq == 6
        entered.set()
        assert finish.wait(2)
        return [detection()]
    def release(slot):
        assert finish.is_set()
        releases.append(slot)
    state.update(_pool=SimpleNamespace(result=result, release_slot=release,
                                      slot_array=lambda slot: np.zeros(1000, np.complex64)),
                 _inflight=deque([dict(seq=6, slot=0, tot_s=1000, pre_hop=None,
                                      history_anchor=(1500, 500, generation))]),
                 a=SimpleNamespace(rate=1000, debug=0), t_start=0,
                 _pat_note_detection=lambda frequency: observed.append(('pat', frequency)),
                 _jk_note_detection=lambda frequency: observed.append(('junk', frequency)),
                 _capture_tail_n=lambda *args: 100,
                 recorder=SimpleNamespace(_nb_mode=True, _shared=object(),
                     update_deferred=lambda *args, **kwargs: saved.append((args, kwargs))))
    exec(production['functions'], state)
    def commit():
        try:
            state['_commit_oldest']()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=commit)
    thread.start()
    try:
        assert entered.wait(1)
        assert not releases
    finally:
        finish.set()
        thread.join(2)
    assert not thread.is_alive() and not errors
    assert releases == [0]
    assert bool(observed) == bool(saved) == accepted
    assert state['tot_d'] == 2 + int(accepted)
    assert ('DETECTED' in capsys.readouterr().out) == accepted


@pytest.mark.parametrize('generation, accepted', [(0, False), (1, True)])
def test_slow_result_checks_generation_before_trigger_state_or_publication(
        production, state, generation, accepted, capsys):
    state['reader']._history_generation = 1
    results = queue.Queue()
    results.put(([detection()], np.zeros(4000, np.complex64), 4000,
                 [2048], (4500, 500, generation)))
    observed, admitted = [], []
    state.update(_slow_results=results, _slow_hold=[], _slow_seeds={},
                 _slow_backoff={}, _slow_fruitless={}, _LAG=4,
                 a=SimpleNamespace(rate=1000), t_start=0,
                 _record_slow_capture=lambda *args, **kwargs: admitted.append((args, kwargs)),
                 _pat_note_detection=lambda frequency: observed.append(('pat', frequency)),
                 _jk_note_detection=lambda frequency: observed.append(('junk', frequency)))
    exec(production['functions'], state)
    exec(production['slow'], state)
    assert results.empty()
    assert bool(state['_slow_seeds']) == bool(admitted) == accepted
    assert not state['_slow_hold']
    assert bool(observed) == accepted
    assert state['tot_d'] == 2 + int(accepted)
    assert ('DETECTED' in capsys.readouterr().out) == accepted


@pytest.mark.parametrize('generation, accepted', [(0, False), (1, True)])
def test_delayed_slow_hold_checks_generation_before_recording(production, state, generation, accepted):
    state['reader']._history_generation = 1
    saved = []
    state.update(_slow_hold=[(41, [detection()], np.zeros(4000, np.complex64),
                             4000, (4500, 500, generation))],
                 a=SimpleNamespace(rate=1000),
                 recorder=SimpleNamespace(_saved_abs_t_lock=threading.Lock(), _saved_abs_t=[]),
                 _record_slow_capture=lambda *args, **kwargs: saved.append((args, kwargs)))
    exec(production['functions'], state)
    exec(production['held'], state)
    assert not state['_slow_hold']
    assert bool(saved) == accepted


def test_live_pool_window_keeps_generation_without_a_shared_recorder(production, state):
    state.update(recorder=None, _inflight=deque(), _slot=0, tot_s=3500)
    exec(production['dispatch_anchor'], state)
    assert state['_inflight'][0]['history_anchor'] == (4000, 500, 0)
