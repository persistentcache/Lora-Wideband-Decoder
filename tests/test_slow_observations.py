"""Fast-work shedding must not erase measured slow-preamble observations."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import detector


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


def production_blocks():
    tree = ast.parse(Path(detector.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    body = next(n.body for n in ast.walk(main) if hasattr(n, 'body')
                and isinstance(n.body, list) and any(assigned(c, '_cap_eff') for c in n.body))
    snapshots = [n for n in body if assigned(n, '_slow_observed_peaks')]
    cap = next(n for n in body if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'len(_gate_peaks) > _cap_eff')
    start = next(i for i,n in enumerate(body) if assigned(n, '_det_bins'))
    end = next(i for i,n in enumerate(body) if assigned(n, '_fire'))
    rate = next(i for i,n in enumerate(body) if assigned(n, '_slow_rate_ok'))
    launch = next(i for i,n in enumerate(body) if isinstance(n, ast.If)
                  and '_slow_busy[0]' in ast.unparse(n.test)
                  and '_slow_rate_ok' in ast.unparse(n.test))
    # Only replace the launched DSP/thread body; use the actual admission test.
    launched = ast.If(test=copy.deepcopy(body[launch].test),
                      body=ast.parse('admitted = True; _slow_busy[0] = True').body,
                      orelse=[])
    prescreen = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == '_jk_sup_bins')
    def code(nodes):
        return compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])),
                       detector.__file__, 'exec')
    # Keep actual production ordering: moving the snapshot below the cap must
    # fail the behavioral tests instead of being repaired by this fixture.
    fast = [n for n in body if n in snapshots or n is cap]
    return {'fast': code(fast),
            'seed': code(body[start:end+1]),
            'launch': code(body[rate:launch] + [launched]),
            'prescreen': code([prescreen]),
            'prescreen_end': prescreen.end_lineno,
            'snapshot_line': snapshots[0].lineno if snapshots else None}


@pytest.fixture(scope='module')
def blocks():
    return production_blocks()


def state(*, busy=False, low_core=False):
    return dict(vars(detector), a=SimpleNamespace(rate=10_000_000, detect_workers=5),
        _live_center_mhz=915., dets=[], _slow_seeds={}, _slow_exact={},
        _slow_backoff={}, _slow_fruitless={}, _slow_pending={}, _slow_tick=0,
        _slow_halves=list(range(8)), _slow_busy=[busy], _slow_last_fire=-1000,
        _SLOW_MIN_TICKS=4 if low_core else 0, _BUDGET_CAP_FLOOR=2,
        _bgt={'emergency':0,'cap_last':24}, _pool=None, _inflight=[],
        reader=SimpleNamespace(available=lambda:0), is_live=True,
        _ring_n_total=160_000_000, hop_n=5_000_000)


def observe(blocks, env, peaks, psd, cap, *, forced=()):
    env.update(_gate_peaks=list(peaks), _psd_gate=psd, _cap_eff=cap, admitted=False)
    env['_slow_tick'] += 1
    exec(blocks['fast'], env)
    fast = list(env['_gate_peaks'])
    # Existing channelizer placeholders are added only AFTER fast-budget cap.
    env['_gate_peaks'].extend(forced)
    exec(blocks['seed'], env)
    exec(blocks['launch'], env)
    return fast, env['admitted']


def signal(width=27, power=30., center=2048):
    psd=np.zeros(4096, np.float32)
    psd[center-width//2:center+(width+1)//2]=power
    return [(float(center),width,power,0.)], psd


@pytest.mark.parametrize('caps',[(2,0,0),(0,0,0),(24,24,24)])
def test_three_measured_windows_launch_even_when_fast_worker_work_is_shed(blocks,caps):
    env=state();peaks,psd=signal()
    outcomes=[observe(blocks,env,peaks,psd,cap) for cap in caps]
    assert [admit for _,admit in outcomes]==[False,False,True]
    assert env['_slow_seeds'][256]==3
    assert all(len(fast)==int(cap>0) for (fast,_),cap in zip(outcomes,caps))


def test_fast_sort_and_cap_remain_identical_while_all_observations_are_retained(blocks):
    env=state();psd=np.zeros(4096,np.float32)
    peaks=[]
    for center,power in [(1000,35),(1500,25),(2048,30),(3000,20)]:
        psd[center-10:center+11]=power;peaks.append((float(center),21,power,0.))
    fast,_=observe(blocks,env,peaks,psd,2)
    assert fast==sorted(peaks,key=lambda p:-p[2])[:2]
    assert set(env['_slow_seeds'])=={int(p[0])//8 for p in peaks}


def test_prescreen_suppression_is_respected_before_snapshot(blocks):
    assert blocks['snapshot_line'] > blocks['prescreen_end']
    env=state();peaks,psd=signal()
    for _ in range(3):
        env.update(_gate_peaks=peaks,_jk_sup_bins=[(2048,14)],_jk_sup_peaks=0,
                   _jk_kept=[],_jk_sup_windows=0,PRESCREEN_VERIFY=False)
        exec(blocks['prescreen'],env)
        _,admitted=observe(blocks,env,env['_gate_peaks'],psd,0)
        assert not admitted
    assert not env['_slow_seeds']


def test_artificial_channel_placeholder_does_not_create_a_measured_streak(blocks):
    env=state();peaks,psd=signal()
    for _ in range(4):
        _,admitted=observe(blocks,env,[],psd,0,forced=peaks)
        assert not admitted
    assert not env['_slow_seeds']


@pytest.mark.parametrize('width,power',[(27,9.9),(80,35.),(1,0.)])
def test_existing_strength_and_narrowness_criteria_still_reject_noise(blocks,width,power):
    env=state();peaks,psd=signal(width,power)
    for _ in range(100):
        _,admitted=observe(blocks,env,peaks,psd,0)
        assert not admitted
    assert not env['_slow_seeds']


def test_busy_scan_cannot_be_duplicated_by_persistent_uncapped_observations(blocks):
    env=state();peaks,psd=signal()
    launches=0
    for _ in range(100):
        _,admitted=observe(blocks,env,peaks,psd,0)
        launches+=admitted
    assert launches==1
    assert env['_slow_busy']==[True]


def test_low_core_emergency_service_veto_is_unchanged(blocks):
    env=state(low_core=True);peaks,psd=signal()
    for _ in range(3):
        _,admitted=observe(blocks,env,peaks,psd,0)
    assert env['_slow_seeds'][256]==3
    assert not admitted
    assert env['_slow_defer_reason']=='budget'


def test_existing_completed_detection_still_suppresses_duplicate_seed(blocks):
    env=state();env['dets']=[{'freq_hz':915e6}];peaks,psd=signal()
    for _ in range(3):
        _,admitted=observe(blocks,env,peaks,psd,0)
        assert not admitted
    assert not env['_slow_seeds']
