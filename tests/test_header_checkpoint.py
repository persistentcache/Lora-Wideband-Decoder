"""Header readiness must be conservative metadata, never payload verification."""
import ast
from pathlib import Path

import numpy as np
import pytest

import decoder


@pytest.fixture(scope='module')
def header_boundary():
    # Execute actual header validation and the transition into payload work.
    # Existing code reaches payload even when a header-only request is desired.
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'not hdr_ok')
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'rdd' for t in n.targets))
    body = fn.body[start:end] + ast.parse("return {'payload_reached': True}").body
    wrapper = ast.parse('def boundary():\n    pass').body[0]
    wrapper.body = body
    return compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                   decoder.__file__, 'exec')


def boundary(code, **values):
    env = dict(vars(decoder))
    env.update(header_only=True, hdr_ok=True, hdr_pmrs=[20.0] * 8,
               payload_len=97, cr=4, crc_present=True, sf=12, bw=62500,
               N=4096, ppm=10, data_start=226048, preamble_start=139008,
               preamble_bin=10, sfd_ok=2, _hdr_minm=100.0,
               valid_variants=[], chosen_td=0)
    env.update(values)
    exec(code, env)
    return env['boundary']()


def independent_symbols(sf, payload_len, cr, crc_present):
    # Count uncoded nibbles explicitly, then fill complete FEC blocks for each
    # possible LDRO mode. Header overflow is already carried by its8 symbols.
    nibbles = 2 * (payload_len + 2 * crc_present)
    remaining = max(0, nibbles - (sf - 2 - 5))
    counts = []
    for width in (sf, sf - 2):
        blocks = 0
        while blocks * width < remaining:
            blocks += 1
        counts.append(8 + blocks * (cr + 4))
    return max(counts)


@pytest.mark.parametrize('sf,pl,cr,crc', [
    (sf, pl, cr, crc) for sf in (7, 12) for pl in (4, 97, 237)
    for cr in (1, 2, 3, 4) for crc in (False, True)
])
def test_header_checkpoint_covers_header_overflow_crc_and_both_ldro_modes(
        header_boundary, sf, pl, cr, crc):
    n = 2 ** sf
    result = boundary(header_boundary, sf=sf, N=n, ppm=sf - 2,
                      payload_len=pl, cr=cr, crc_present=crc)
    assert 'payload_reached' not in result
    assert result['payload_len'] == pl and result['cr'] == cr
    assert result['crc_present'] == crc
    expected = independent_symbols(sf, pl, cr, crc)
    assert result['data_symbols'] == expected
    assert result['ready_end_bw'] == 226048 + (expected + 1) * n
    assert result['header_end_bw'] == 226048 + 8 * n
    assert 'verified' not in result and 'crc_ok' not in result


@pytest.mark.parametrize('preamble,data', [(100, 100 + 20 * 4096), (9000, 9000 + 36 * 4096)])
def test_actual_data_start_controls_end_not_assumed_preamble_length(header_boundary, preamble, data):
    result = boundary(header_boundary, preamble_start=preamble, data_start=data)
    assert result['preamble_start_bw'] == preamble
    assert result['ready_end_bw'] == data + 169 * 4096


@pytest.mark.parametrize('changes', [
    {'hdr_ok': False}, {'payload_len': 3}, {'payload_len': 238},
    {'cr': 0}, {'cr': 5}, {'sfd_ok': 0}, {'sfd_ok': 1},
    {'_hdr_minm': float('nan')}, {'_hdr_minm': 0},
    {'preamble_start': -1}, {'data_start': 139008},
    {'valid_variants': [(1, 0, (None, None, None, (40, 4, True, True)))]},
])
def test_unusable_or_conflicting_header_cannot_create_readiness(header_boundary, changes):
    result = boundary(header_boundary, **changes)
    assert not isinstance(result, dict)


def test_normal_payload_transition_remains_unchanged(header_boundary):
    assert boundary(header_boundary, header_only=False) == {'payload_reached': True}


def test_wrapper_uses_one_attempt_actual_rate_and_leaves_prefix_unchanged(monkeypatch):
    rng = np.random.default_rng(17)
    prefix = (rng.normal(size=4097) + 1j * rng.normal(size=4097)).astype(np.complex64)
    prefix.setflags(write=False)
    before = prefix.tobytes()
    calls = []
    expected = {'ready_end_bw': 90000, 'payload_len': 97}

    def attempt(iq, sf, bw, n, ppm, fs, dec, name, counter, **kwargs):
        calls.append((iq, sf, bw, n, ppm, fs, dec, name, kwargs))
        return expected

    monkeypatch.setattr(decoder, '_decode_attempt', attempt)
    monkeypatch.setattr(decoder.time, 'process_time', lambda: 10)
    bw = 125000 / 3
    token = ('stream', 3, 4, 1234)
    result = decoder.probe_frame_header(prefix, 12, bw, bw * 8, 'diagnostic',
                                       budget_s=2, identity_token=token)
    assert result == dict(expected, identity_token=token) and len(calls) == 1
    iq, sf, actual_bw, n, ppm, fs, dec, name, kwargs = calls[0]
    assert (sf, actual_bw, n, ppm, fs, dec) == (12, bw, 4096, 10, bw * 8, 8)
    assert len(iq) == (len(prefix) + 7) // 8 and iq.dtype == np.complex64
    assert kwargs['header_only'] is True and kwargs['deadline'] == 12
    assert prefix.tobytes() == before


@pytest.mark.parametrize('label_bw', [41667, 41670])
def test_d30_realized_rate_uses_canonical_bandwidth_and_exact_time_grid(monkeypatch, label_bw):
    rate, physical_bw = 10e6 / 30, 125000 / 3
    # A bin-centered physical tone makes the FFT extraction's sample-clock
    # contract independently observable, without mocking its numeric work.
    prefix = np.exp(2j * np.pi * 32 * np.arange(8192) / 8192).astype(np.complex64)
    seen = []

    def attempt(iq, sf, bw, n, ppm, fs, dec, name, counter, **kwargs):
        seen.append((iq, bw, fs, dec))
        return {'bw': bw, 'ready_end_bw': 50000}

    monkeypatch.setattr(decoder, '_decode_attempt', attempt)
    result = decoder.probe_frame_header(prefix, 12, label_bw, rate, 'd30', budget_s=2)
    assert result and len(seen) == 1
    iq, bw, actual_rate, dec = seen[0]
    assert (bw, actual_rate, dec) == (physical_bw, rate, 8)
    expected = np.exp(2j * np.pi * 32 * np.arange(1024) / 1024)
    np.testing.assert_allclose(iq, expected, atol=2e-6, rtol=2e-6)
    assert result['bw'] == physical_bw
    # Geometry maps through the returned physical BW, not its rounded label.
    assert result['ready_end_bw'] * rate / result['bw'] == pytest.approx(400000)
    assert result['ready_end_bw'] / result['bw'] == pytest.approx(1.2)


def test_true_fractional_clock_uses_full_capture_fallback_without_fabricating_endpoint(monkeypatch):
    rate, bw = 510000.0, 62500.0  # ratio8.16: actual clock, not label rounding
    t = np.arange(8192) / rate
    prefix = np.exp(2j * np.pi * 12000 * t).astype(np.complex64)
    before = prefix.tobytes()
    monkeypatch.setattr(decoder, '_fft', lambda *a, **k: pytest.fail('unsupported clock must abstain'))
    monkeypatch.setattr(decoder, '_decode_attempt', lambda *a, **k: pytest.fail('no fabricated geometry'))
    assert decoder.probe_frame_header(prefix, 12, bw, rate, 'fractional', budget_s=2) is None
    assert prefix.tobytes() == before


@pytest.mark.parametrize('kind', ['expired', 'nonfinite', 'bad_rate', 'empty', 'bad_sf'])
def test_invalid_or_expired_prefix_does_no_frontend_work(monkeypatch, kind):
    monkeypatch.setattr(decoder.time, 'process_time', lambda: 10)
    monkeypatch.setattr(decoder, '_fft', lambda *a, **k: pytest.fail('unexpected FFT'))
    prefix = np.ones(128, np.complex64)
    sf, rate, budget = 12, 500000, 2
    if kind == 'expired': budget = 0
    if kind == 'nonfinite': prefix[1] = np.nan
    if kind == 'bad_rate': rate = 490000
    if kind == 'empty': prefix = prefix[:0]
    if kind == 'bad_sf': sf = 13
    assert decoder.probe_frame_header(prefix, sf, 62500, rate, 'x', budget_s=budget) is None


def test_failed_attempt_is_not_header_metadata(monkeypatch):
    monkeypatch.setattr(decoder.time, 'process_time', lambda: 10)
    monkeypatch.setattr(decoder, '_decode_attempt', lambda *a, **k: ('FAIL', 0, 100, 4))
    assert decoder.probe_frame_header(np.ones(100, np.complex64), 12, 62500,
                                      500000, 'x', budget_s=2) is None


def test_header_only_skips_every_frontend_capture_state_side_effect():
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'rdd' for t in n.targets))
    blocks = [node for node in fn.body[:end] if isinstance(node, ast.If)
              and any(name in ast.unparse(node.test) for name in (
                  '_HARNESS_OUT', '_CURRENT_HW_FP', '_sweep_gate_enabled'))]
    blocks += [node for node in fn.body[:end] if isinstance(node, ast.If)
               and '_LAST_EMIT_VERIFIED[0] = False' in ast.unparse(node)]
    assert len(blocks) == 6
    env = dict(vars(decoder), header_only=True, _HARNESS_OUT=True,
               _HARNESS_ATTEMPT=[17], _LAST_EMIT_VERIFIED=[True],
               _CURRENT_HW_FP=[{'untouched': 1}],
               _GATE_HDR_TUPLES={(97, 4, True)}, _GATE_MAX_MNORM=[8.0],
               _harness_emit=lambda *a, **k: pytest.fail('harness side effect'),
               _extract_tx_fingerprint=lambda *a, **k: pytest.fail('fingerprint side effect'),
               _sweep_gate_enabled=lambda: pytest.fail('fluke-state side effect'))
    exec(compile(ast.Module(body=blocks, type_ignores=[]), decoder.__file__, 'exec'), env)
    assert env['_LAST_EMIT_VERIFIED'] == [True]
    assert env['_HARNESS_ATTEMPT'] == [17]
    assert env['_CURRENT_HW_FP'] == [{'untouched': 1}]
    assert env['_GATE_HDR_TUPLES'] == {(97, 4, True)}
    assert env['_GATE_MAX_MNORM'] == [8.0]


@pytest.mark.parametrize('failure', ['fft_exception', 'fft_deadline', 'attempt_exception'])
def test_failure_paths_keep_state_and_cannot_emit(monkeypatch, failure):
    clock = [10.0]
    monkeypatch.setattr(decoder.time, 'process_time', lambda: clock[0])
    monkeypatch.setattr(decoder, '_LAST_EMIT_VERIFIED', [True])
    monkeypatch.setattr(decoder, '_CURRENT_HW_FP', [{'untouched': 1}])
    monkeypatch.setattr(decoder, '_GATE_HDR_TUPLES', {(97, 4, True)})
    monkeypatch.setattr(decoder, '_GATE_MAX_MNORM', [8.0])
    monkeypatch.setattr(decoder, '_emit_pkt', lambda *a: pytest.fail('packet emission'))
    original_fft, original_attempt = decoder._fft, decoder._decode_attempt

    def fft(*args, **kwargs):
        if failure == 'fft_exception':
            raise RuntimeError('backend failed')
        result = original_fft(*args, **kwargs)
        if failure == 'fft_deadline':
            clock[0] = 13
        return result

    def attempt(*args, **kwargs):
        if failure != 'attempt_exception':
            pytest.fail('deadline/exception must stop before attempt')
        # Real frontend entry must leave verification state untouched, even
        # if a later frontend backend raises and the checkpoint falls back.
        original_attempt(*args, **kwargs)
        raise RuntimeError('frontend failed')

    monkeypatch.setattr(decoder, '_fft', fft)
    monkeypatch.setattr(decoder, '_decode_attempt', attempt)
    assert decoder.probe_frame_header(np.zeros(4096, np.complex64), 7, 125000,
                                      125000, 'x', budget_s=2) is None
    assert decoder._LAST_EMIT_VERIFIED == [True]
    assert decoder._CURRENT_HW_FP == [{'untouched': 1}]
    assert decoder._GATE_HDR_TUPLES == {(97, 4, True)}
    assert decoder._GATE_MAX_MNORM == [8.0]
