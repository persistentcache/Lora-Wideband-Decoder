"""A failed coarse ranking must not hide the refined capture-origin grid."""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import decoder


@pytest.fixture(scope='module')
def production_search():
    # Keep the real candidate collection, deduplication and both fallbacks;
    # supply only expensive RF search boundaries with controlled outcomes.
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'coarse_positions'
                         for t in n.targets))
    end = next(i for i, n in enumerate(fn.body[start:], start)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'run_phases'
                       for t in n.targets)) + 2
    return compile(ast.Module(body=fn.body[start:end], type_ignores=[]),
                   decoder.__file__, 'exec')


def search(code, *, coarse=(128, 256, 384), coarse_run=False,
           origin_run=True, origin_phase=13, unrefined_run=True,
           global_run=False, deadline=None, clocks=(0.,), scan_results=None,
           run_results=None):
    calls = []
    clock = iter(clocks)
    last_clock = [clocks[-1]]

    def now():
        last_clock[0] = next(clock, last_clock[0])
        return last_clock[0]

    def xcorr(iq, position):
        calls.append(('xcorr', position))
        return origin_phase if position == 0 else position + 7

    def energy(iq, position, n_score):
        calls.append(('energy', position))
        return position + 9

    def scan(iq, phase):
        calls.append(('scan', phase))
        if run_results is not None and phase in run_results:
            runs = run_results[phase]
            return runs, [symbol for run in runs for symbol in run]
        if scan_results is not None and phase in scan_results:
            syms = scan_results[phase]
            return [syms], syms
        # Distinct bins retain each candidate through production's dedup.
        bin_ = (70 if origin_run and phase == origin_phase and phase != 0
                else 20 if unrefined_run and phase == 0
                else 44 if coarse_run and phase == 135
                else 100 if global_run and phase == 1 else None)
        syms = [(i, bin_ or 0, 12.) for i in range(4)]
        return ([syms] if bin_ is not None else []), syms

    n = 128
    ns = dict(vars(decoder), N=n, iq1=np.zeros(n * (16 if global_run else 8),
              dtype=np.complex64), upchirp=np.ones(n, dtype=np.complex64),
              _find_coarse_positions=lambda iq, n_top: list(coarse),
              _refine_start_xcorr=xcorr, _refine_start=energy, _scan_from=scan,
              deadline=deadline, time=SimpleNamespace(process_time=now))
    exec(code, ns)
    return SimpleNamespace(candidates=[(r[0][1], phase)
                           for r, phase in zip(ns['all_runs'], ns['run_phases'])],
                           runs=ns['all_runs'],
                           calls=calls, best_phase=ns['best_phase'], ns=ns)


def test_failed_coarse_search_retains_refined_and_unrefined_origin_grids(production_search):
    result = search(production_search)
    assert result.candidates == [(20, 0), (70, 13)]
    assert result.best_phase == 0


def test_successful_coarse_search_keeps_existing_candidates_and_work(production_search):
    result = search(production_search, coarse_run=True)
    assert result.candidates == [(44, 135), (20, 0)]
    assert result.best_phase == 135
    assert ('xcorr', 0) not in result.calls


@pytest.mark.parametrize('refined_bin', [20, 23])
def test_refined_origin_keeps_later_unrefined_run_in_same_bin_bucket(
        production_search, refined_bin):
    result = search(production_search, scan_results={
        13: [(i, refined_bin, 12.) for i in range(4)],
        0: [(i, 20, 12.) for i in range(16, 24)],
    })
    # The original longer, later run must survive even when the optional
    # refined-origin grid has the same bin or the same20-bin bucket.
    assert result.candidates == [(20, 0), (refined_bin, 13)]
    assert [r[0][0] for r in result.runs] == [16, 0]
    assert [len(r) for r in result.runs] == [8, 4]
    assert result.best_phase == 0


def test_origin_already_ranked_is_not_refined_twice(production_search):
    result = search(production_search, coarse=(0, 128, 256), origin_run=False)
    assert result.candidates == [(20, 0)]
    assert result.calls.count(('xcorr', 0)) == 1


def test_empty_origin_refinement_preserves_unrefined_fallback(production_search):
    result = search(production_search, origin_run=False)
    assert result.candidates == [(20, 0)]


def test_zero_refined_offset_is_scanned_once_by_existing_fallback(production_search):
    result = search(production_search, origin_phase=0)
    assert result.candidates == [(20, 0)]
    assert result.calls.count(('scan', 0)) == 1


def test_no_local_run_preserves_global_xcorr_fallback(production_search):
    result = search(production_search, origin_run=False, unrefined_run=False,
                    global_run=True)
    assert result.candidates == [(100, 1)]


def test_optional_origin_does_not_suppress_existing_global_search(production_search):
    result = search(production_search, unrefined_run=False, global_run=True)
    assert set(result.candidates) == {(100, 1), (70, 13)}
    baseline = search(production_search, origin_run=False,
                      unrefined_run=False, global_run=True)
    assert result.best_phase == baseline.best_phase


@pytest.fixture(scope='module')
def production_candidate_selection():
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'cand_infos'
                         for t in n.targets))
    end = next(i for i, n in enumerate(fn.body[start:], start)
               if isinstance(n, ast.Assign) and any(
                   isinstance(t, ast.Tuple) and any(
                       isinstance(v, ast.Name) and v.id == 'best_score'
                       for v in t.elts) for t in n.targets)) + 1
    body = fn.body[start:end] + ast.parse('return locals()').body
    wrapper = ast.FunctionDef(name='select_candidate',
                              args=ast.arguments(posonlyargs=[], args=[],
                                                 kwonlyargs=[], kw_defaults=[],
                                                 defaults=[]),
                              body=body, decorator_list=[])
    return compile(ast.fix_missing_locations(ast.Module(
        body=[wrapper], type_ignores=[])), decoder.__file__, 'exec')


@pytest.mark.parametrize('expire_before_optional', [False, True])
def test_optional_origin_retains_sixth_baseline_sfd_candidate(
        production_search, production_candidate_selection, expire_before_optional):
    # F is the sixth baseline run by length and the only real SFD. A new,
    # longer origin-grid run G must not evict F before production scoring.
    baseline = [[(first + i, bin_, 20.) for i in range(length)]
                for first, length, bin_ in zip((64, 128, 192, 256, 320, 0),
                                               (10, 9, 8, 7, 6, 5),
                                               (0, 21, 42, 63, 84, 105))]
    optional = [(384 + i, 0, 20.) for i in range(11)]
    result = search(production_search, run_results={0: baseline, 13: [optional]})
    n = 128
    upchirp = np.exp(1j * np.pi * np.arange(n) ** 2 / n).astype(np.complex64)
    iq = np.zeros(448 * n, np.complex64)
    for run, phase in [(run, 0) for run in baseline] + [(optional, 13)]:
        first = run[0][0] * n + phase
        tone = np.exp(2j * np.pi * run[0][1] * np.arange(n) / n)
        iq[first:first + len(run) * n] = np.tile(upchirp * tone, len(run))
    tone_f = np.exp(2j * np.pi * 105 * np.arange(n) / n)
    iq[7 * n:9 * n] = np.tile(np.conj(upchirp) * tone_f, 2)
    refined = []

    def refine(pre_loc, run_bin, n_syms):
        refined.append(pre_loc)
        return pre_loc

    result.ns.update(iq1=iq, sf=7, bw=125000, upchirp=upchirp,
                     downchirp=np.conj(upchirp), skip_bins=[], Counter=Counter,
                     refine_preamble_start=refine,
                     deadline=1. if expire_before_optional else None,
                     time=SimpleNamespace(process_time=lambda: 2.),
                     print=lambda *a, **kw: None)
    exec(production_candidate_selection, result.ns)
    result.ns.update(result.ns['select_candidate']())
    assert result.ns['preamble_start'] == 0
    assert result.ns['best_sfd_ok'] == 2
    assert set(refined).issuperset(run[0][0] * n for run in baseline)
    assert (384 * n + 13 in refined) is not expire_before_optional


def test_successful_coarse_path_keeps_its_six_run_evaluation_limit(
        production_search, production_candidate_selection):
    runs = [[(64 * j + i, 0, 20.) for i in range(10 - j)]
            for j in range(7)]
    result = search(production_search, run_results={135: runs, 0: []})
    n = 128
    upchirp = np.exp(1j * np.pi * np.arange(n) ** 2 / n).astype(np.complex64)
    refined = []

    def refine(pre_loc, run_bin, n_syms):
        refined.append(pre_loc)
        return pre_loc

    result.ns.update(iq1=np.zeros(448 * n, np.complex64), sf=7, bw=125000,
                     upchirp=upchirp, downchirp=np.conj(upchirp),
                     skip_bins=[], Counter=Counter,
                     refine_preamble_start=refine, print=lambda *a, **kw: None)
    exec(production_candidate_selection, result.ns)
    result.ns['select_candidate']()
    assert refined == [135 + run[0][0] * n for run in runs[:6]]
    assert ('xcorr', 0) not in result.calls


@pytest.mark.parametrize('clock', [1., 2.])
def test_expired_deadline_does_not_add_origin_work(production_search, clock):
    result = search(production_search, deadline=1., clocks=(clock,))
    assert result.candidates == [(20, 0)]
    assert ('xcorr', 0) not in result.calls


def test_deadline_after_refinement_prevents_added_scan(production_search):
    result = search(production_search, deadline=1., clocks=(0., 2.))
    assert result.candidates == [(20, 0)]
    assert ('scan', 13) not in result.calls
