"""A later verified-header frame may retry one mean-CFO/no-slope demodulation."""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import decoder


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


@pytest.fixture(scope='module')
def production_fit():
    tree = ast.parse(Path(decoder.__file__).read_text())
    attempt = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_decode_attempt')
    start = next(i for i, n in enumerate(attempt.body) if assigned(n, 'k_hat_fine'))
    block = attempt.body[start:start + 3]
    assert isinstance(block[-1], ast.If) and 'cfo_vals' in ast.unparse(block[-1].test)
    shell = ast.parse('def fit(cfo_vals, cfo_indices, preamble_fit=True, diag_out=None):\n'
                      '    return k_hat_fine, drift_bins_per_sym')
    shell.body[0].body = block + shell.body[0].body
    ast.fix_missing_locations(shell)
    namespace = dict(np=np)
    exec(compile(shell, '<production preamble regression>', 'exec'), namespace)
    return namespace['fit']


def test_default_fit_keeps_regression_intercept_and_slope(production_fit):
    assert production_fit([10., 12., 14., 16.], [0, 1, 2, 3]) == (10., 2.)


def test_disabled_fit_keeps_mean_cfo_and_zero_slope(production_fit):
    assert production_fit([10., 12., 14., 16.], [0, 1, 2, 3], False) == (13., 0.)


@pytest.mark.parametrize('enabled', [True, False])
def test_short_preamble_keeps_existing_mean_cfo(production_fit, enabled):
    assert production_fit([10., 12., 14.], [0, 1, 2], enabled) == (12., 0.)


@pytest.fixture(scope='module')
def production_loop():
    tree = ast.parse(Path(decoder.__file__).read_text())
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_process_file_label')
    maximum = next(n for n in process.body if assigned(n, 'MAX_ATTEMPTS'))
    start = next(i for i, n in enumerate(process.body) if assigned(n, '_need_rescue'))
    end = next(i for i, n in enumerate(process.body) if isinstance(n, ast.For)
               and isinstance(n.target, ast.Name) and n.target.id == 'attempt')
    shell = ast.parse('def run(iq1_work, sf, N, ppm): return iq1_work')
    shell.body[0].body = [maximum] + process.body[start:end + 1] + shell.body[0].body
    ast.fix_missing_locations(shell)
    return compile(shell, '<production first-pass loop>', 'exec')


def run_loop(code, *, failure_diag=None, bw=62500, prior_verified=True,
             first_frame_fails=False, retry_fails=False, budget_expires=False,
             fail_frame=2, frame_count=2, respect_skip_bins=False,
             initial_skip_bins=()):
    # Only the expensive demodulation boundary is supplied. Eligibility,
    # masking, retry order, attempt accounting and CPU checks are production.
    namespace = dict(vars(decoder))
    calls, recovered = [], []
    cpu = [0.0]
    last_verified = [False]
    base_diag = dict(header_decoded=True, crc_ok=False, preamble_fit_applied=True)
    base_diag.update(failure_diag or {})

    def demod(iq, *args, preamble_fit=True, grid_sweep=False, diag_out=None, **kw):
        skipped = list(kw.get('skip_bins', []))
        calls.append(dict(fit=preamble_fit, grid=grid_sweep, skip_bins=skipped))
        last_verified[0] = False
        present = np.flatnonzero(iq)
        if not len(present):
            return None
        start = int(present[0])
        frame = int(iq[start].real)
        if frame >= fail_frame and (preamble_fit or retry_fails
                                    or (respect_skip_bins and 0 in skipped)):
            diag_out.update(base_diag)
            if budget_expires:
                cpu[0] = 2.0
            return ('FAIL', start, 20, 0)
        recovered.append(frame)
        verified = prior_verified or frame != 1
        diag_out.update(verified=verified, crc_ok=verified)
        last_verified[0] = verified
        namespace['_EMIT_COUNT'] += 1
        return ('OK', start, 20, 0)

    namespace.update(_decode_attempt=demod, _recenter_and_decode=lambda cfo: None,
                     _EMIT_COUNT=0, _LAST_EMIT_VERIFIED=last_verified,
                     _BUDGET_S=1.0 if budget_expires else 0, _process_start=0,
                     time=SimpleNamespace(process_time=lambda: cpu[0]),
                     skip_bins=list(initial_skip_bins), found_cfos_hz=[], _ddl=None,
                     bw=bw, fs=bw * 8, dec=8, name='synthetic.cf32', Counter=Counter)
    exec(code, namespace)
    iq = np.zeros(frame_count * 70 + 20, dtype=np.complex64)
    for frame in range(1, frame_count + 1):
        if frame == 1 and first_frame_fails:
            continue
        iq[20 + (frame - 1) * 70:40 + (frame - 1) * 70] = frame
    namespace['run'](iq, 12, 4096, 10)
    return calls, recovered, iq


def test_verified_first_frame_allows_one_regression_off_retry(production_loop):
    calls, recovered, iq = run_loop(production_loop)
    assert recovered == [1, 2]
    assert [c['fit'] for c in calls] == [True, True, False, True]
    assert not np.any(iq)


def test_retry_preserves_same_preamble_eligibility_and_earlier_skips(production_loop):
    calls, recovered, iq = run_loop(production_loop, respect_skip_bins=True,
                                    initial_skip_bins=(777,))
    assert recovered == [1, 2]
    fallback = next(c for c in calls if not c['fit'])
    assert fallback['skip_bins'] == [777]


@pytest.mark.parametrize('options', [
    {'prior_verified': False},
    {'first_frame_fails': True},
    {'bw': 125000},
    {'failure_diag': {'header_decoded': False}},
    {'failure_diag': {'crc_ok': None}},
    {'failure_diag': {'crc_ok': True}},
    {'failure_diag': {'preamble_fit_applied': False}},
    {'failure_diag': {'fluke_gated': True}},
    {'failure_diag': {'no_rescue': True}},
])
def test_retry_does_not_expand_to_unverified_noise_or_other_failure_classes(
        production_loop, options):
    calls, recovered, iq = run_loop(production_loop, **options)
    assert all(c['fit'] for c in calls)


def test_failed_fallback_is_not_repeated(production_loop):
    calls, recovered, iq = run_loop(production_loop, retry_fails=True)
    assert recovered == [1]
    assert sum(not c['fit'] for c in calls) == 1


def test_later_success_does_not_grant_a_second_fallback(production_loop):
    calls, recovered, iq = run_loop(production_loop, frame_count=3)
    assert recovered == [1, 2]
    assert sum(not c['fit'] for c in calls) == 1


def test_fallback_does_not_extend_global_attempt_limit(production_loop):
    calls, recovered, iq = run_loop(production_loop, fail_frame=5, frame_count=6)
    assert recovered == [1, 2, 3, 4]
    assert len(calls) == 5
    assert all(c['fit'] for c in calls)
    assert np.all(iq[300:320] == 5)


def test_fallback_does_not_reset_cpu_deadline(production_loop):
    calls, recovered, iq = run_loop(production_loop, budget_expires=True)
    assert recovered == [1]
    assert len(calls) == 2
    assert all(c['fit'] for c in calls)
    assert np.all(iq[90:110] == 2)
