"""Gate decisions as data: every window leaves a record that says what the
gate did with each strong peak and what the detect stage concluded."""
import json
import os
import time

import numpy as np
import pytest

import detector
from gate_ledger import GateLedger, ledger_from_environ
from test_dechirp_coverage_dsp import chirps, banded_noise

RATE = 10e6
FRES = RATE / 4096.0


def test_record_carries_every_stage_for_strong_peaks_only(tmp_path):
    led = GateLedger(str(tmp_path / 'l.jsonl'))
    rec = led.begin(7, 123.456, 1.0, -80.0, FRES, 915e6)
    tooth = (2048 + 100, 4, -35.0, -80.0)          # +45 dB
    real = (2048, 51, -42.0, -80.0)                 # +38 dB
    weak = (500, 20, -70.0, -80.0)                  # +10 dB: counted, not listed
    led.scanned(rec, [tooth, real, weak])
    led.removed(rec, 'doomed', [tooth])
    led.removed(rec, 'shed', [weak])
    led.forced(rec, 1)
    led.dispatched(rec, [real], 'pool', 12)
    led.outcomes(rec, {'peaks': [{'f': 915.0, 'w': 51, 'e': 38, 'r': 'detected',
                                  'lag': 1024, 'sc': 0.99, 'thr': 0.4, 'det': [(7, 125000)]}],
                       'cap_override': 0})
    led.write(rec); led.close()
    row = json.loads((tmp_path / 'l.jsonl').read_text())
    assert row['seq'] == 7 and (row['t0'], row['t1']) == (122.456, 123.456)
    assert row['route'] == 'pool' and row['cap'] == 12
    assert (row['scan'], row['doomed'], row['shed'], row['forced']) == (3, 1, 1, 1)
    assert not any(k.startswith('_') for k in row)
    fates = {p['f']: p for p in row['peaks']}
    assert fates[915.2441]['fate'] == 'doomed' and fates[915.2441]['e'] == 45
    assert fates[915.0] == {'f': 915.0, 'w': 51, 'e': 38, 'fate': 'kept', 'r': 'detected',
                            'lag': 1024, 'sc': 0.99, 'thr': 0.4, 'det': [[7, 125000]]}
    assert 'cap_override' not in row                 # zero is not recorded


def test_weak_peak_verdicts_are_kept_only_when_they_detected(tmp_path):
    led = GateLedger(str(tmp_path / 'l.jsonl'))
    rec = led.begin(1, 10.0, 1.0, -80.0, FRES, 915e6)
    led.dispatched(rec, [(500, 20, -70.0, -80.0), (600, 20, -70.0, -80.0)], 'inline', 24)
    led.outcomes(rec, [{'f': round((915e6 + (500 - 2048) * FRES) / 1e6, 4), 'w': 20, 'e': 10, 'r': 'sc-below', 'sc': 0.1},
                       {'f': round((915e6 + (600 - 2048) * FRES) / 1e6, 4), 'w': 20, 'e': 10, 'r': 'detected', 'det': [(9, 125000)]}])
    assert [p['r'] for p in rec['peaks']] == ['detected']


def test_rotation_keeps_one_previous_file(tmp_path):
    led = GateLedger(str(tmp_path / 'l.jsonl'), max_bytes=300)
    for i in range(10):
        led.write(led.begin(i, i, 1.0, -80.0, FRES, 915e6))
    led.close()
    assert (tmp_path / 'l.jsonl.1').exists()
    rows = [json.loads(l) for l in (tmp_path / 'l.jsonl').read_text().splitlines()]
    assert rows and rows[-1]['seq'] == 9


def test_default_on_at_debug_one_off_at_zero_env_overrides(tmp_path):
    assert ledger_from_environ({}, 0, str(tmp_path)) is None
    led = ledger_from_environ({}, 1, str(tmp_path))
    assert led.path == os.path.join(str(tmp_path), 'lora_gate_ledger.jsonl')
    assert ledger_from_environ({'LORA_GATE_LEDGER': ''}, 1, str(tmp_path)) is None
    assert ledger_from_environ({'LORA_GATE_LEDGER': '/x/y.jsonl'}, 0, str(tmp_path)).path == '/x/y.jsonl'


def _peak(offset_hz, width_hz, power_db, floor_db=-80.0):
    return (2048 + int(round(offset_hz / FRES)), max(1, round(width_hz / FRES)), power_db, floor_db)


def test_detect_stage_reports_a_verdict_per_peak(monkeypatch):
    monkeypatch.setenv('LORA_FFT_WORKERS', '1')
    psd = np.full(4096, -80., np.float32)
    iq = chirps(RATE, 4.85e6, 7, 250000., .12)
    out = []
    dets = detector.detect_preamble(iq, RATE, RATE, 915., cached_psd=psd,
                                    cached_peaks=[_peak(4.85e6, 250e3, -40.)], outcomes=out)
    assert dets and len(out) == 1
    o = out[0]
    assert o['r'] == 'detected' and o['det'] == [(7, 250000)] and o['e'] == 40
    assert o['f'] == round(915 + round(4.85e6 / FRES) * FRES / 1e6, 4)   # the gate bin's carrier
    assert o['sc'] >= o['thr'] and o['lag'] > 0

    out = []
    noise = banded_noise(RATE, 300e3, 125e3, 1.0, 1.0)
    assert detector.detect_preamble(noise, RATE, RATE, 915., cached_psd=psd,
                                    cached_peaks=[_peak(300e3, 125e3, -40.)], outcomes=out) == []
    assert out[0]['r'] == 'sc-below' and out[0]['sc'] < out[0]['thr'] and 'det' not in out[0]

    out = []
    tooth = (2048 + 123, 4, -35.0, -80.0)           # strong and narrower than any class
    assert detector.detect_preamble(noise, RATE, RATE, 915., cached_psd=psd,
                                    cached_peaks=[tooth], outcomes=out) == []
    assert out[0] == {'f': round(915 + 123 * FRES / 1e6, 4), 'w': 4, 'e': 45, 'r': 'no-class'}

    out = []
    detector.detect_preamble(noise, RATE, RATE, 915., cached_psd=psd,
                             cached_peaks=[_peak(300e3, 125e3, -40.)], outcomes=out,
                             only_bws={31250, 41667})
    assert out[0]['r'] == 'slow-wide'

    # Without the argument nothing changes and no thread-local state leaks.
    detector.detect_preamble(noise, RATE, RATE, 915., cached_psd=psd, cached_peaks=[tooth])
    assert getattr(detector._DPQ_MEMO, 'why', None) is None


def test_pool_returns_the_verdicts_beside_the_result(monkeypatch):
    monkeypatch.setenv('LORA_FFT_WORKERS', '1')
    import detect_pool
    sig = chirps(RATE, 4.85e6, 7, 250000., .12)      # a preamble burst, not a continuous train
    n = len(sig)
    params = dict(wb_fs=RATE, wb_bw=RATE, center=915., sc_threshold=.55, ethresh=6.,
                  dc_notch=0., spur_notch=None)
    pool = detect_pool.DetectPool(1, 2, int(RATE), params)
    try:
        slot = pool.acquire_slot()
        pool.slot_array(slot)[:n] = sig
        psd = np.full(4096, -80., np.float32)
        pool.dispatch(slot, 0, n, psd, [_peak(4.85e6, 250e3, -40.), (2048 + 400, 4, -35.0, -80.0)])
        dets = pool.result(0)
        out = pool.outcomes(0)
        assert dets and all(d['sf'] == 7 for d in dets)
        assert out['cap_override'] == 0
        assert [o['r'] for o in out['peaks']] == ['detected', 'no-class']
        assert pool.outcomes(0) is None
        pool.release_slot(slot)
        slot = pool.acquire_slot()
        pool.slot_array(slot)[:n] = sig
        pool.set_cap_override(1)
        pool.dispatch(slot, 1, n, psd, [_peak(4.85e6, 250e3, -40.), (2048 + 400, 4, -35.0, -80.0)])
        pool.result(1)
        assert pool.outcomes(1)['cap_override'] == 1 and len(pool.outcomes(1) or {'peaks': []}['peaks']) == 0
    finally:
        pool.close()


def test_main_loop_wiring_is_complete():
    """Every route a window can take ends in exactly one write."""
    import ast
    from pathlib import Path
    src = Path(detector.__file__).read_text()
    for route in ("'skipped'", "'pool'", "'quiet'", "'inline'"):
        assert f"_ledger.dispatched(_lrec, _gate_peaks, {route}" in src, route
    assert src.count('_ledger.write(') == 4           # skipped, quiet, inline, pooled commit
    assert "_ledger.removed(_lrec, 'prescreen'" in src
    assert "_ledger.removed(_lrec, 'doomed'" in src
    assert "_ledger.removed(_lrec, 'shed'" in src
    assert "_pool.outcomes(it['seq'])" in src and "'ledger': _lrec" in src
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    begin = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == '_ledger.begin')
    prescreen = src.index('[PRESCREEN] outcome-learned persistent-junk suppression')
    assert src[:prescreen].count('_ledger.begin(') == 1      # record starts before the prescreen


def test_file_mode_run_writes_one_record_per_window(tmp_path):
    """The whole program, not an exec'd block: a 3 s file at 2 Msps with one
    SF7/125k preamble burst leaves a ledger whose records cover the run and
    whose burst window reports the detection."""
    import subprocess, sys
    rate = 2_000_000
    sig = chirps(rate, 300e3, 7, 125000., .05)
    iq = np.zeros(3 * rate, np.complex64)
    iq[rate: rate + len(sig)] = sig
    rng = np.random.default_rng(3)
    iq += 0.003 * (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)).astype(np.complex64)
    raw = np.empty(2 * iq.size, np.int16)
    raw[0::2] = np.clip(iq.real * 8000, -32767, 32767); raw[1::2] = np.clip(iq.imag * 8000, -32767, 32767)
    (tmp_path / 'in.sc16').write_bytes(raw.tobytes())
    ledger = tmp_path / 'ledger.jsonl'
    env = {k: v for k, v in os.environ.items() if not k.startswith('LORA_')}
    env.update(LORA_GATE_LEDGER=str(ledger), LORA_FFT_WORKERS='1', LORA_SHARED_CAPTURE='0')
    src = os.path.join(os.path.dirname(detector.__file__), 'detector.py')
    proc = subprocess.run([sys.executable, src, '-f', str(tmp_path / 'in.sc16'), '-t', 'sc16',
                           '-r', str(rate), '-b', str(rate), '-c', '915.0', '--overlap', '.5',
                           '--energy-threshold', '6', '--threshold', '.55', '--detect-workers', '1',
                           '-d', '0'], env=env, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert len(rows) >= 4 and [r['seq'] for r in rows] == sorted(r['seq'] for r in rows)
    assert all(r['route'] in ('inline', 'pool', 'quiet', 'skipped') for r in rows)
    hits = [p for r in rows for p in r['peaks'] if p.get('r') == 'detected']
    assert hits and any(abs(p['f'] - 915.3) < 0.01 and [7, 125000] in p['det'] for p in hits)
    assert 'DETECTED' in proc.stdout
