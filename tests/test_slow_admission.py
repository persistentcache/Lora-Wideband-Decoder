"""Slow scans use current budget and actual spare detection capacity."""
import ast
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import detector


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


@pytest.fixture(scope='module')
def production_blocks():
    # Execute the real main-loop budget and admission blocks in isolation.
    # No detector/radio process, or duplicate policy implementation, is used.
    tree = ast.parse(Path(detector.__file__).read_text())
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    body = next(node.body for node in ast.walk(main)
                if hasattr(node, 'body') and isinstance(node.body, list)
                and any(assigned(child, '_cap_eff') for child in node.body))
    end = next(i for i, node in enumerate(body) if assigned(node, '_cap_eff'))
    start = next(i for i, node in enumerate(body[:end]) if assigned(node, '_cap_budget'))
    rate_start = next(i for i, node in enumerate(body)
                      if assigned(node, '_slow_rate_ok'))
    launch_index = next(i for i, node in enumerate(body) if isinstance(node, ast.If)
                        and '_slow_busy[0]' in ast.unparse(node.test)
                        and '_slow_rate_ok' in ast.unparse(node.test))
    launch = body[launch_index]
    # Preserve the exact launch condition; replace only its hardware/DSP body.
    launch_probe = ast.If(test=copy.deepcopy(launch.test),
                          body=[copy.deepcopy(launch.body[0])]
                               + ast.parse('admitted = True').body, orelse=[])
    def code(nodes):
        return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                       detector.__file__, 'exec')
    return code(body[start:end + 1]), code(body[rate_start:launch_index] + [launch_probe])


class DetectPool:
    n_slots = 8
    def __init__(self, unfinished, occupied):
        self.unfinished = unfinished
        self.occupied = occupied
    def ready(self, seq):
        return seq >= self.unfinished
    def drain_costs(self):
        return []
    def n_free(self):
        return self.n_slots - self.occupied


def state(*, backlog_s=0.24, inflight=0, unfinished=0, historical_cap=2,
          last_fire=100, tick=400, busy=False, live=True):
    return {
        'sys': sys, 'os': os, 'a': SimpleNamespace(detect_workers=3, rate=10_000_000),
        'is_live': live,
        'reader': SimpleNamespace(available=lambda: int(backlog_s * 10_000_000)),
        '_PEAK_CAP_HI': 24, '_PEAK_CAP_LO': 3, '_peak_cap': 24, '_cap_calm': 0,
        '_ring_n_total': 160_000_000, 'hop_n': 5_000_000,
        '_pool': DetectPool(unfinished, inflight),
        '_inflight': [{'seq': i} for i in range(inflight)], '_LAG': 4,
        '_bgt': {'margin': .85, 'pp': .417, 'ovh': .090, 'cap_last': historical_cap},
        '_hop_budget_s': .5, '_BUDGET_CAP_FLOOR': 2,
        '_budget_track_pp': lambda *args: None, '_fire': [2050],
        '_SLOW_MIN_TICKS': 4, '_slow_halves': list(range(8)),
        '_slow_tick': tick, '_slow_last_fire': last_fire,
        '_slow_rate_ok': True, '_slow_busy': [busy], 'admitted': False,
    }


def run(blocks, **kwargs):
    env = state(**kwargs)
    for block in blocks:
        exec(block, env)
    return env


@pytest.mark.parametrize('historical_cap', [0, 1, 2])
def test_recovered_current_budget_releases_old_floor_latch(production_blocks, historical_cap):
    env = run(production_blocks, historical_cap=historical_cap)
    assert env['_bgt']['cap_last'] == historical_cap  # history remains untouched
    assert env['_cap_eff'] == 24  # actual production budget has recovered
    assert env['admitted']


@pytest.mark.parametrize('kwargs, expected_cap', [
    ({'inflight': 5, 'unfinished': 3}, 2),
    ({'backlog_s': 1.0}, 2),
    ({'backlog_s': 5.0}, 0),
])
def test_actual_floor_and_emergency_still_block(production_blocks, kwargs, expected_cap):
    env = run(production_blocks, **kwargs)
    assert env['_cap_eff'] == expected_cap
    assert not env['admitted']


def test_hard_ring_pressure_blocks_even_if_current_budget_is_high(production_blocks):
    env = state(backlog_s=5.0)
    env['_cap_eff'] = 24
    exec(production_blocks[1], env)
    assert not env['admitted']


@pytest.mark.parametrize('kwargs', [{'last_fire': 398}, {'busy': True}])
def test_minimum_spacing_and_single_scan_guard_remain(production_blocks, kwargs):
    env = run(production_blocks, **kwargs)
    assert env['_cap_eff'] == 24
    assert not env['admitted']


def test_file_mode_ignores_live_budget_floor(production_blocks):
    assert run(production_blocks, live=False)['admitted']


def test_empty_trigger_or_incomplete_assembly_never_launch(production_blocks):
    for key, value in [('_fire', []), ('_slow_halves', list(range(7)))]:
        env = state()
        env[key] = value
        for block in production_blocks:
            exec(block, env)
        assert not env['admitted']


@pytest.mark.parametrize('current_cap', [0, 1, 2])
def test_current_floor_veto_does_not_depend_on_cached_high_cap(production_blocks, current_cap):
    env = state(historical_cap=24, inflight=5, unfinished=3)
    env['_cap_eff'] = current_cap
    exec(production_blocks[1], env)
    assert not env['admitted']


def test_debug_deferral_identifies_actual_cap_and_reason(production_blocks, monkeypatch, capsys):
    monkeypatch.setenv('LORA_SLOW_DEBUG', '1')
    run(production_blocks, backlog_s=1.0)
    message = capsys.readouterr().err
    assert 'cap_eff=2' in message
    assert 'cap_last=2' in message
    assert 'reason=budget' in message


@pytest.mark.parametrize('unfinished', [0, 1, 2])
def test_ready_tail_slots_allow_one_spare_worker_without_raising_peak_cap(
        production_blocks, unfinished):
    env = run(production_blocks, inflight=5, unfinished=unfinished)
    assert env['_cap_eff'] == 2
    assert env['_bgt']['cap_last'] == 2
    assert env['admitted']


@pytest.mark.parametrize('unfinished', [3, 4, 5])
def test_occupied_workers_keep_floor_veto(production_blocks, unfinished):
    env = run(production_blocks, inflight=5, unfinished=unfinished)
    assert env['_cap_eff'] == 2
    assert not env['admitted']


def test_completed_slots_still_need_one_free_slot(production_blocks):
    env = run(production_blocks, inflight=8, unfinished=0)
    assert env['_cap_eff'] == 2
    assert not env['admitted']


@pytest.mark.parametrize('backlog_s, admitted', [(.4999999, True), (.5, False), (1., False)])
def test_floor_exception_requires_less_than_one_hop(production_blocks, backlog_s, admitted):
    env = run(production_blocks, inflight=5, backlog_s=backlog_s)
    assert env['_cap_eff'] == 2
    assert env['admitted'] is admitted


@pytest.mark.parametrize('emergency', [1, 2])
def test_floor_exception_never_overrides_emergency_state(production_blocks, emergency):
    env = state(inflight=5)
    env['_cap_eff'] = 2
    env['_bgt']['emergency'] = emergency
    exec(production_blocks[1], env)
    assert not env['admitted']


@pytest.mark.parametrize('cap', [0, 1])
def test_idle_pool_never_overrides_emergency_cap(production_blocks, cap):
    env = state(inflight=5)
    env['_cap_eff'] = cap
    # Also cover a configured floor of1; this is not spare-capacity permission.
    env['_BUDGET_CAP_FLOOR'] = max(1, cap)
    exec(production_blocks[1], env)
    assert not env['admitted']


def test_serial_detection_cannot_claim_an_idle_pool_worker(production_blocks):
    env = state()
    env['_pool'] = None
    env['_cap_eff'] = 2
    exec(production_blocks[1], env)
    assert not env['admitted']


@pytest.mark.parametrize('kwargs', [{'last_fire': 398}, {'busy': True}])
def test_spare_floor_capacity_keeps_spacing_and_single_scan_guards(production_blocks, kwargs):
    env = run(production_blocks, inflight=5, **kwargs)
    assert env['_cap_eff'] == 2
    assert not env['admitted']


@pytest.mark.parametrize('before, after, admitted', [(0, 3, False), (3, 0, True)])
def test_admission_refreshes_readiness_after_dispatch(production_blocks, before, after, admitted):
    env = state(inflight=5, unfinished=before)
    exec(production_blocks[0], env)
    assert env['_unfinished'] == before
    env['_pool'].unfinished = after
    exec(production_blocks[1], env)
    assert env['admitted'] is admitted


@pytest.mark.parametrize('unfinished, decision', [(0, 'floor-spare'), (3, 'budget')])
def test_debug_reports_fresh_capacity_on_admit_and_defer(
        production_blocks, monkeypatch, capsys, unfinished, decision):
    monkeypatch.setenv('LORA_SLOW_DEBUG', '1')
    run(production_blocks, inflight=5, unfinished=unfinished)
    message = capsys.readouterr().err
    assert f'reason={decision}' in message
    assert f'unfinished={unfinished}' in message
    assert 'free_slots=3' in message
    assert 'backlog_s=0.240' in message
    assert 'emergency=0' in message
