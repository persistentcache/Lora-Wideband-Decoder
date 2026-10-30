"""Payload recovery cannot create missing FEC symbols or received CRC bytes."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import decoder


@pytest.mark.parametrize('ppm,ppm_pay,cr,need_bytes,symbols', [
    (10, 10, 1, 58, 68),
    (10, 12, 1, 58, 58),
    (10, 10, 1, 56, 63),  # Same payload without two CRC bytes.
    (10, 10, 4, 99, 168),
    (10, 12, 4, 99, 144),
    (10, 10, 4, 239, 392),
    (5, 7, 1, 39, 68),
])
def test_capacity_requires_the_last_complete_fec_block(
        ppm, ppm_pay, cr, need_bytes, symbols):
    # Counts are hand-derived from header overflow and complete FEC blocks.
    start, n = 1000, 128
    end = start + symbols * n
    assert decoder._payload_fits_samples(end, start, n, ppm, ppm_pay, cr,
                                         need_bytes)
    assert not decoder._payload_fits_samples(end - 1, start, n, ppm, ppm_pay,
                                             cr, need_bytes)


@pytest.mark.parametrize('offset', [-8, -16, -64])
def test_capacity_keeps_a_frame_reachable_at_an_earlier_timing_offset(offset):
    start, n = 1000, 128
    end = start + 68 * n + offset
    assert decoder._payload_fits_samples(end, start, n, 10, 10, 1, 58, offset)
    assert not decoder._payload_fits_samples(end - 1, start, n, 10, 10, 1,
                                             58, offset)


def test_short_ldro_on_does_not_imply_short_ldro_off():
    start, n = 1000, 128
    end = start + 67 * n
    assert not decoder._payload_fits_samples(end, start, n, 10, 10, 1, 58)
    assert decoder._payload_fits_samples(end, start, n, 10, 12, 1, 58)


@pytest.fixture(scope='module')
def production_blocks():
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    primary_start = next(i for i, n in enumerate(fn.body)
                         if isinstance(n, ast.Assign)
                         and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name)
                         and n.value.func.id == 'soft_decode_payload')
    ldro_end = next(i for i, n in enumerate(fn.body)
                    if isinstance(n, ast.If)
                    and 'LORA_LDRO_FALLBACK' in ast.unparse(n.test)) + 1
    recovery = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in
                {'ds_offsets', '_payload_recovery_fits'} for t in node.targets):
            recovery.append(copy.deepcopy(node))
        if not isinstance(node, ast.If):
            continue
        body_text = ast.unparse(node)
        stage = ('primary_dd' if '_primary_dd_curve' in body_text
                 else 'timing_sweep' if '=== SWEEP (timing' in body_text
                 else 'curved_dd' if '_retry_bases' in body_text else None)
        if stage:
            # Preserve production eligibility; replace only expensive work.
            selected = copy.deepcopy(node)
            selected.body = ast.parse(f"calls.append('{stage}')").body
            selected.orelse = []
            recovery.append(selected)
    assert len([n for n in recovery if isinstance(n, ast.If)]) == 3

    def compile_body(body):
        return compile(ast.fix_missing_locations(ast.Module(
            body=body, type_ignores=[])), decoder.__file__, 'exec')

    return (compile_body(fn.body[primary_start:ldro_end]),
            compile_body(recovery))


def run_recovery(blocks, *, available=67, ppm_pay=10, dec=8,
                 tail_adjust=0, primary=False, alternate_crc=False, **flags):
    n, start = 128, 1000
    ns = dict(vars(decoder), N=n, sf=12, ppm=10, ppm_pay=ppm_pay, cr=1,
              need_bytes=58, payload_len=56, data_start=start, dec=dec,
              iq1=np.zeros(start + available * n + tail_adjust, np.complex64),
              pay_levels=n // 4, pay_bin_group=4, sym_time_ms=32.768,
              crc_present=True, crc_ok=False, hdr_ok=True, _fluke_junk=False,
              _skip_sweep=False, _c2_skip=False, _payload_recovery_fits=True,
              deadline=None, time=SimpleNamespace(process_time=lambda: 0.),
              _drift_applied=False, tau_frac=0., diag_out={}, calls=[],
              print=lambda *args, **kwargs: None)
    modes = []

    def demod(*args, **kwargs):
        mode = ns['ppm_pay']
        modes.append(mode)
        size = 58 if mode == 12 else 57
        raw = [42] * size
        return raw, raw[:56], [], [1] * (size * 2)

    ns.update(soft_decode_payload=demod,
              check_crc=lambda raw, length, **kw: (alternate_crc, 'fixture'))
    ns.update(flags)
    if primary:
        exec(blocks[0], ns)
    exec(blocks[1], ns)
    return SimpleNamespace(ns=ns, modes=modes)


def test_incomplete_mode_skips_only_its_impossible_payload_recovery(production_blocks):
    trial = run_recovery(production_blocks)
    assert trial.ns['calls'] == []
    assert trial.ns['hdr_ok'] and trial.ns['diag_out'] == {}
    assert trial.ns['crc_present'] and not trial.ns['crc_ok']


@pytest.mark.parametrize('options', [
    {'available': 68},
    {'available': 58, 'ppm_pay': 12},
    {'available': 68, 'tail_adjust': -8},
    {'available': 68, 'tail_adjust': -64, 'dec': 64},
])
def test_complete_or_timing_reachable_mode_retains_recovery(production_blocks, options):
    trial = run_recovery(production_blocks, **options)
    assert trial.ns['calls'] == ['primary_dd', 'timing_sweep', 'curved_dd']


def test_one_sample_beyond_earliest_offset_cannot_be_recovered(production_blocks):
    trial = run_recovery(production_blocks, available=68, tail_adjust=-65, dec=64)
    assert trial.ns['calls'] == []


def test_alternate_ldro_success_is_kept_before_capacity_pruning(
        production_blocks, monkeypatch):
    monkeypatch.setenv('LORA_LDRO_FALLBACK', '1')
    trial = run_recovery(production_blocks, primary=True, alternate_crc=True)
    assert trial.modes == [10, 12]
    assert trial.ns['ppm_pay'] == 12
    assert trial.ns['crc_ok'] and trial.ns['_clean_crc']
    assert trial.ns['diag_out']['crc_ok']
    assert len(trial.ns['raw_bytes']) == 58
    assert trial.ns['calls'] == []


def test_failed_alternate_keeps_primary_partial_and_outer_search_evidence(
        production_blocks, monkeypatch):
    monkeypatch.setenv('LORA_LDRO_FALLBACK', '1')
    trial = run_recovery(production_blocks, primary=True, alternate_crc=False)
    assert trial.modes == [10, 12]
    assert trial.ns['ppm_pay'] == 10
    assert len(trial.ns['raw_bytes']) == 57
    assert trial.ns['hdr_ok'] and 'no_rescue' not in trial.ns['diag_out']
    assert trial.ns['calls'] == []


def test_crc_absent_path_does_not_gain_or_lose_payload_recovery(production_blocks):
    assert run_recovery(production_blocks, crc_present=False).ns['calls'] == []
