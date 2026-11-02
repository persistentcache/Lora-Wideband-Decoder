"""A gate peak the SC stage is certain to refuse never takes a cap slot from
one that can become a detection; the slow pass still sees every peak."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import detector
from detector import _sc_width_admits, _gate_peak_doomed, MIN_LORA_BW_HZ

RATE = 10e6
FRES = RATE / 4096.0          # 2441 Hz per bin


def test_width_rule_is_shared_between_gate_and_sc_stage():
    # +45 dB tooth, 4 bins = 9.8 kHz: no class at or above 31.25 kHz admits it.
    assert not _sc_width_admits(MIN_LORA_BW_HZ, 4 * FRES, 45.0)
    assert _gate_peak_doomed((100, 4, -35.0, -80.0), FRES, -70.0)
    # 6 bins = 14.6 kHz * 2.5 = 36.6 kHz admits the 31.25 kHz class.
    assert _sc_width_admits(MIN_LORA_BW_HZ, 6 * FRES, 45.0)
    assert not _gate_peak_doomed((100, 6, -35.0, -80.0), FRES, -70.0)
    # A weak narrow peak keeps every class: its width is not trusted.
    assert _sc_width_admits(MIN_LORA_BW_HZ, 4 * FRES, 10.0)
    assert not _gate_peak_doomed((100, 4, -70.0, -80.0), FRES, -70.0)
    # Elevation is measured against the peak's own pass floor when carried,
    # else the window floor; exactly the SC stage's reference.
    assert not _gate_peak_doomed((100, 4, -35.0, -45.0), FRES, -80.0)
    assert _gate_peak_doomed((100, 4, -35.0), FRES, -80.0)
    assert _gate_peak_doomed((100, 4, -35.0, None), FRES, -80.0)


def _main_body():
    tree = ast.parse(Path(detector.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets)
    body = next(n.body for n in ast.walk(main) if hasattr(n, 'body')
                and isinstance(n.body, list) and any(assigns(c, '_cap_eff') for c in n.body))
    start = next(i for i, n in enumerate(body) if assigns(n, '_n_doomed'))
    cap = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'len(_gate_peaks) > _cap_eff')
    snapshot = next(n for n in body if assigns(n, '_slow_observed_peaks'))
    return body, start, cap, snapshot


def test_slow_snapshot_precedes_the_doomed_drop():
    body, start, cap, snapshot = _main_body()
    assert snapshot.lineno < body[start].lineno < body[cap].lineno


def _run_cap(peaks, cap, *, debug=1, reported=-1e9):
    body, start, cap_i, _ = _main_body()
    code = compile(ast.fix_missing_locations(ast.Module(body[start:cap_i + 1], type_ignores=[])),
                   detector.__file__, 'exec')
    env = dict(vars(detector), a=SimpleNamespace(rate=RATE, debug=debug),
               _gate_peaks=list(peaks), _psd_gate=np.full(4096, -80., np.float32),
               _cap_eff=cap, _gate_doomed={'peaks': 0, 'windows': 0, 'reported': reported})
    exec(code, env)
    return env


def comb(n, power=-35.0, width=4, floor=-80.0):
    """n spur-comb teeth on a 50 kHz grid, like ladder_live8 (2026-10-31)."""
    step = int(round(50e3 / FRES))
    return [(600 + k * step, width, power + (k % 3), floor) for k in range(n)]


def test_comb_teeth_do_not_outrank_a_real_preamble_at_the_cap(capsys):
    real = (2048, 51, -42.0, -80.0)                    # SF7/125k at +38 dB, 125 kHz wide
    peaks = comb(30) + [real]
    env = _run_cap(peaks, cap=12)
    kept = env['_gate_peaks']
    assert real in kept and len(kept) == 1               # every tooth is gone, nothing to shed
    assert env['_gate_doomed'] == {'peaks': 0, 'windows': 0, 'reported': env['_gate_doomed']['reported']}
    err = capsys.readouterr().err
    assert '[GATE] dropped 30 spur-width peak(s) over 1 window(s)' in err
    assert '[GATE-THROTTLE]' not in err


def test_without_the_drop_the_same_window_would_have_shed_the_preamble():
    real = (2048, 51, -42.0, -80.0)
    peaks = comb(30) + [real]
    ranked = sorted(peaks, key=lambda p: -p[2])[:12]
    assert real not in ranked                            # the pre-fix outcome


def test_wide_peaks_still_shed_weakest_first_and_count_nothing(capsys):
    peaks = [(500 + 100 * k, 51, -60.0 + k, -80.0) for k in range(6)]   # +20..+25 dB, LoRa-shaped
    env = _run_cap(peaks, cap=4)
    kept = env['_gate_peaks']
    assert [p[2] for p in kept] == [-55.0, -56.0, -57.0, -58.0]
    assert env['_gate_doomed']['peaks'] == 0
    err = capsys.readouterr().err
    assert '[GATE-THROTTLE] shed 2 weakest' in err and '[GATE] dropped' not in err


def test_weak_narrow_peaks_are_left_for_the_sc_stage():
    peaks = [(500 + 100 * k, 4, -70.0, -80.0) for k in range(5)]        # +10 dB, width untrusted
    env = _run_cap(peaks, cap=24)
    assert env['_gate_peaks'] == peaks


def test_drop_report_is_rate_limited_and_debug_gated(capsys):
    env = _run_cap(comb(3), cap=24, debug=0)
    assert env['_gate_doomed']['peaks'] == 3 and capsys.readouterr().err == ''
    import time
    env = _run_cap(comb(3), cap=24, debug=1, reported=time.monotonic())
    assert env['_gate_doomed']['peaks'] == 3                             # accumulated, not printed
    assert '[GATE]' not in capsys.readouterr().err


def test_sc_stage_refuses_exactly_the_peaks_the_gate_drops(monkeypatch):
    """The two sites share one rule: a tooth the gate drops returns nothing
    from the SC stage, and a peak the gate keeps reaches it."""
    monkeypatch.setattr(detector, '_unconfirmed_last', {})
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal(int(RATE)) + 1j * rng.standard_normal(int(RATE))).astype(np.complex64)
    psd = np.full(4096, -80., np.float32)
    tooth = (2048 + 123, 4, -35.0, -80.0)
    assert _gate_peak_doomed(tooth, FRES, -80.0)
    assert detector.detect_preamble(iq, RATE, RATE, 915., cached_psd=psd,
                                    cached_peaks=[tooth], debug=0) == []
    seen = []
    orig = detector._schmidl_cox_curve_multilag
    monkeypatch.setattr(detector, '_schmidl_cox_curve_multilag',
                        lambda *a, **k: (seen.append(1), orig(*a, **k))[1])
    detector.detect_preamble(iq, RATE, RATE, 915., cached_psd=psd, cached_peaks=[tooth], debug=0)
    assert not seen                                      # no class evaluated
    wide = (2048 + 123, 6, -35.0, -80.0)
    assert not _gate_peak_doomed(wide, FRES, -80.0)
    detector.detect_preamble(iq, RATE, RATE, 915., cached_psd=psd, cached_peaks=[wide], debug=0)
    assert seen                                          # at least one class evaluated
