"""Truncated frames, header grid phase and same-preamble aliases in one attempt."""
import ast
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import decoder

H = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
     [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
     [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]]


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


def header_nibbles(payload_len, cr, crc_present, ppm):
    """Explicit-header nibbles with a valid checksum plus filler payload nibbles."""
    nibs = [payload_len >> 4, payload_len & 0xF, (cr << 1) | int(crc_present)]
    bits = [(nibs[i] >> (3 - j)) & 1 for i in range(3) for j in range(4)]
    calc = [sum(h & b for h, b in zip(row, bits)) & 1 for row in H]
    nibs += [calc[0], (calc[1] << 3) | (calc[2] << 2) | (calc[3] << 1) | calc[4]]
    nibs += list(np.random.default_rng(1).integers(0, 16, ppm - 5))
    assert decoder.parse_header(nibs)[3]
    return nibs


def chirps(sf, bw):
    N = 1 << sf
    t = np.arange(N, dtype=np.float64) / bw
    up = np.exp(1j * np.pi * bw / (N / bw) * t ** 2).astype(np.complex64)
    return up, np.conj(up).astype(np.complex64)


def frame(sf, bw, payload_len, cr, crc_present, data_symbols, preamble_start,
          length, seed=3, noise=0.02):
    """BW-rate capture holding one frame whose header checks and whose
    payload symbols are arbitrary (geometry and grid tests never need a CRC)."""
    N = 1 << sf
    ppm = sf - 2
    hdr = [4 * lvl for lvl in decoder.header_expected_levels(
        header_nibbles(payload_len, cr, crc_present, ppm), ppm)]
    rng = np.random.default_rng(seed)
    pay = list(rng.integers(0, N, max(0, data_symbols - 8)))
    iq = decoder.reconstruct_lora_packet(length, preamble_start, hdr + pay, sf, bw, bw)
    iq = iq + noise * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    return iq.astype(np.complex64)


def attempt(iq1, sf, bw):
    diag = {}
    result = decoder._decode_attempt(iq1, sf, bw, 1 << sf, sf - 2, bw * 8, 8,
                                     'synthetic.cf32', Counter, diag_out=diag)
    return result, diag


# ---------------------------------------------------------------- truncation
def test_frame_cut_before_either_ldro_mode_completes_is_terminal():
    sf, bw, N = 7, 125000, 128
    # PL20+CRC at CR4/5, LDRO off: 44 nibbles / 7 -> 7 blocks -> 43 symbols;
    # LDRO on would need 53.  The capture ends 20 symbols into the data.
    start = 2 * N
    data_start = start + int(round(12.25 * N))
    length = data_start + 20 * N
    result, diag = attempt(frame(sf, bw, 20, 1, True, 43, start, length), sf, bw)
    assert result[0] == 'FAIL'
    assert diag['truncated'] and diag['no_rescue']
    assert diag['header_decoded']
    assert abs(result[1] - start) < N // 4
    assert result[1] + result[2] == length          # masked to the capture end
    assert 'crc_ok' not in diag                     # no payload work was spent


def test_complete_frame_is_not_called_truncated():
    sf, bw, N = 7, 125000, 128
    start = 2 * N
    length = start + int(round(12.25 * N)) + 43 * N + 3 * N
    result, diag = attempt(frame(sf, bw, 20, 1, True, 43, start, length), sf, bw)
    assert diag['header_decoded']
    assert 'truncated' not in diag
    assert result[1] + result[2] < length


def test_opposite_ldro_fit_is_truncated_when_the_burst_runs_past_its_end():
    # SF11/125k is LDRO-on (16.384 ms symbols).  PL56 no-CRC: 108 nibbles
    # after the header's 4 -> 12 blocks of 9 -> 68 symbols; LDRO-off would be
    # 58.  Only the LDRO-off frame fits a capture cut 64 symbols in, but the
    # burst is still on through the cut: six symbols of LoRa past the
    # LDRO-off end prove the LDRO-on frame is what was transmitted.
    sf, bw, N = 11, 125000, 2048
    start = 2 * N
    data_start = start + int(round(12.25 * N))
    length = data_start + 64 * N
    result, diag = attempt(frame(sf, bw, 56, 1, False, 68, start, length), sf, bw)
    assert result[0] == 'FAIL'
    assert diag['truncated'] and diag['no_rescue']
    assert result[1] + result[2] == length


def test_opposite_ldro_frame_that_really_ends_keeps_its_payload_attempt():
    sf, bw, N = 11, 125000, 2048
    start = 2 * N
    data_start = start + int(round(12.25 * N))
    length = data_start + 64 * N
    # Same header, but the transmitter used LDRO off: 58 symbols then noise.
    result, diag = attempt(frame(sf, bw, 56, 1, False, 58, start, length), sf, bw)
    assert diag['header_decoded']
    assert 'truncated' not in diag


# ------------------------------------------------------- first-pass ladder
@pytest.fixture(scope='module')
def production_loop():
    tree = ast.parse(Path(decoder.__file__).read_text())
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_process_file_label')
    maximum = next(n for n in process.body if assigned(n, 'MAX_ATTEMPTS'))
    start = next(i for i, n in enumerate(process.body) if assigned(n, '_need_rescue'))
    end = next(i for i, n in enumerate(process.body) if isinstance(n, ast.For)
               and isinstance(n.target, ast.Name) and n.target.id == 'attempt')
    shell = ast.parse('def run(iq1_work, sf, N, ppm):\n'
                      '    return iq1_work, _need_rescue, skip_bins, _residual_clear')
    shell.body[0].body = [maximum] + process.body[start:end + 1] + shell.body[0].body
    ast.fix_missing_locations(shell)
    return compile(shell, '<production first-pass loop>', 'exec')


def run_ladder(code, second_frame_diag, bw):
    """Frame 1 decodes; frame 2 (starting at sample 90, running to the end)
    fails with the given diagnostics.  Only demodulation is a stub."""
    namespace = dict(vars(decoder))
    recenters, attempts = [], []

    def demod(iq, *args, diag_out=None, **kw):
        attempts.append(dict(skip_bins=list(kw.get('skip_bins', []))))
        present = np.flatnonzero(iq)
        if not len(present):
            return None
        start = int(present[0])
        if start < 90:
            diag_out.update(verified=True, crc_ok=True)
            namespace['_EMIT_COUNT'] += 1
            return ('OK', start, 60, 0)
        diag_out.update(second_frame_diag)
        return ('FAIL', start, len(iq) - start, 5)

    namespace.update(_decode_attempt=demod, _EMIT_COUNT=0,
                     _recenter_and_decode=lambda cfo: recenters.append(cfo),
                     _LAST_EMIT_VERIFIED=[False], _BUDGET_S=0, _process_start=0,
                     time=SimpleNamespace(process_time=lambda: 0.0),
                     skip_bins=[], found_cfos_hz=[], _ddl=None, bw=bw, fs=bw * 8,
                     dec=8, name='synthetic.cf32', Counter=Counter)
    exec(code, namespace)
    iq = np.zeros(200, dtype=np.complex64)
    iq[20:80] = 1
    iq[90:] = 2
    out = namespace['run'](iq, 12, 4096, 10)
    return out, recenters, attempts


def test_truncated_frame_is_masked_banned_and_earns_no_rescue(production_loop):
    diag = dict(hdr_ok=True, header_decoded=True, truncated=True, no_rescue=True)
    (iq, need_rescue, skip_bins, residual_clear), recenters, attempts = run_ladder(
        production_loop, diag, bw=125000)
    assert recenters == []                 # header-OK alone used to recenter
    assert not need_rescue
    assert residual_clear                  # the ladder ended on a clean residual
    assert skip_bins == [5]
    assert not np.any(iq)                  # masked on its first failure
    assert len(attempts) == 3              # decode, truncated frame, empty


def test_ordinary_header_failure_still_recenters(production_loop):
    diag = dict(hdr_ok=True, header_decoded=True)
    (_, need_rescue, _, _), recenters, _ = run_ladder(production_loop, diag, bw=125000)
    assert recenters                       # unchanged behaviour for real failures
    assert need_rescue



# ------------------------------------------------------ header grid descent
@pytest.fixture(scope='module')
def production_descent():
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    i = next(i for i, n in enumerate(fn.body) if assigned(n, '_hdr0'))
    assert isinstance(fn.body[i + 1], ast.If)
    shell = ast.parse('def descend(iq1, preamble_start, pre_last_i, N, ppm, downchirp):\n'
                      '    return preamble_start, iq1')
    shell.body[0].body = fn.body[i:i + 2] + shell.body[0].body
    ast.fix_missing_locations(shell)
    namespace = dict(np=np, _fft=decoder._fft, soft_fft_demod=decoder.soft_fft_demod,
                     soft_deinterleave=decoder.soft_deinterleave,
                     soft_hamming_decode_batch=decoder.soft_hamming_decode_batch,
                     parse_header=decoder.parse_header)
    exec(compile(shell, '<production header grid descent>', 'exec'), namespace)
    return namespace['descend']


def preamble_bin(iq1, start, N, downchirp):
    X = np.abs(decoder._fft(iq1[start:start + N] * downchirp))
    b = int(np.argmax(X))
    return b - N if b > N // 2 else b


def header_valid(iq1, data_start, N, ppm, downchirp, cfo_shift=0):
    llrs = [decoder.soft_fft_demod(iq1[data_start + j * N:data_start + (j + 1) * N],
                                   downchirp, N, N // 4, ppm, 4, cfo_shift=cfo_shift)
            for j in range(8)]
    hb, _, _ = decoder.soft_hamming_decode_batch(decoder.soft_deinterleave(llrs, 8, ppm), 8)
    res = decoder.parse_header([int(hb[k]) for k in range(ppm)])
    return bool(res is not None and res[3] and 4 <= res[0] <= 237 and 1 <= res[1] <= 4)


@pytest.mark.parametrize('offset', [120, -100, 240, -230])
def test_descent_rescues_only_a_grid_whose_header_does_not_check(
        production_descent, offset):
    sf, bw, N = 9, 125000, 512
    true_start = 3 * N
    iq1 = frame(sf, bw, 30, 1, True, 40, true_start, 70 * N)
    _, downchirp = chirps(sf, bw)
    # A misphased grid still dechirps the preamble to a clean tone g bins up,
    # and the preamble CFO correction that precedes the descent has already
    # rotated that tone to bin 0 — the state the descent starts from.
    assert abs(preamble_bin(iq1, true_start + offset, N, downchirp) - offset) <= 1
    iq1 = (iq1 * np.exp(-2j * np.pi * offset * np.arange(len(iq1)) / N)).astype(np.complex64)
    assert abs(preamble_bin(iq1, true_start + offset, N, downchirp)) <= 1
    misphased_valid = header_valid(iq1, true_start + offset + int(12.25 * N), N, sf - 2, downchirp)
    start, out = production_descent(iq1, true_start + offset, 7, N, sf - 2, downchirp)
    if misphased_valid:
        # The old path would decode this attempt; its grid is left alone.
        assert start == true_start + offset
        assert out is iq1
    else:
        assert abs(start - true_start) <= N // 64
        assert abs(preamble_bin(out, start, N, downchirp)) <= 1
        assert header_valid(out, start + int(12.25 * N), N, sf - 2, downchirp)


def test_header_checksum_survives_misphases_up_to_nearly_half_a_symbol():
    # Documents why the descent is rescue-only in practice: the soft header
    # decode keeps checking on grids misphased by up to ~47% of a symbol
    # even at -4 dB per-sample noise, so a grid that fails to read a header
    # is far off or impaired, never "slightly late".  The parametrisation
    # above therefore exercises the leave-alone branch at every offset.
    sf, bw, N = 9, 125000, 512
    true_start = 3 * N
    _, downchirp = chirps(sf, bw)
    for noise_level in (0.02, 0.6):
        iq1 = frame(sf, bw, 30, 1, True, 40, true_start, 70 * N, noise=noise_level)
        for offset in (120, -100, 240, -230):
            rot = (iq1 * np.exp(-2j * np.pi * offset * np.arange(len(iq1)) / N)).astype(np.complex64)
            assert header_valid(rot, true_start + offset + int(12.25 * N), N, sf - 2, downchirp)


def test_descent_leaves_an_aligned_grid_alone(production_descent):
    sf, bw, N = 9, 125000, 512
    true_start = 3 * N
    iq1 = frame(sf, bw, 30, 1, True, 40, true_start, 70 * N)
    _, downchirp = chirps(sf, bw)
    start, out = production_descent(iq1, true_start, 7, N, sf - 2, downchirp)
    assert start == true_start
    assert out is iq1


def test_descent_never_moves_onto_a_grid_without_a_valid_header(production_descent):
    # Noise sharpness fluctuates by more than 10% between grids; without a
    # checksum-valid header on the proposed grid the candidate stays put.
    sf, bw, N = 9, 125000, 512
    _, downchirp = chirps(sf, bw)
    rng = np.random.default_rng(11)
    for seed in range(6):
        iq1 = (rng.standard_normal(70 * N) + 1j * rng.standard_normal(70 * N)).astype(np.complex64)
        start, out = production_descent(iq1, 3 * N + 50, 7, N, sf - 2, downchirp)
        assert start == 3 * N + 50
        assert out is iq1


# ------------------------------------------------------ same-preamble merge
@pytest.fixture(scope='module')
def production_merge():
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    i = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.FunctionDef)
             and n.name == '_alias_header_margin')
    j = next(j for j in range(i, len(fn.body)) if assigned(fn.body[j], 'cand_infos'))
    shell = ast.parse('def merge(cand_infos, N, iq1, downchirp, ppm, diag_out=None,\n'
                      '          skip_bins=(), alias_skip=False):\n'
                      '    return cand_infos')
    shell.body[0].body = fn.body[i:j + 1] + shell.body[0].body
    ast.fix_missing_locations(shell)
    namespace = dict(np=np, soft_fft_demod=decoder.soft_fft_demod,
                     soft_deinterleave=decoder.soft_deinterleave,
                     soft_hamming_decode_batch=decoder.soft_hamming_decode_batch,
                     parse_header=decoder.parse_header)
    exec(compile(shell, '<production alias merge>', 'exec'), namespace)
    return namespace['merge']


def cand(score, start, bin_, pre_last_i=15):
    return (score, 16, 15.0, 2, 30.0, start, bin_, pre_last_i, [], 0.0, pre_last_i + 3)


def noise(n, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_grid_aliases_of_one_preamble_merge_to_the_best_scored(production_merge):
    N = 2048
    best = cand(261.2, 122880, 36)
    aliases = [cand(257.4, 122689, 1893),     # start -191, bin -191
               cand(252.5, 124403, 1559),     # start +1523, bin +1523
               cand(246.3, 124800, 1956)]     # start +1920, bin +1920
    other_cfo = cand(240.0, 122880, 60)       # same time, 24 bins away
    far_same_cfo = cand(230.0, 122880 + 5 * N, 36)   # a later frame
    _, downchirp = chirps(11, 250000)
    kept = production_merge([best] + aliases + [other_cfo, far_same_cfo], N,
                            noise(200 * N), downchirp, 9)
    assert kept == [best, other_cfo, far_same_cfo]   # noise: no header, score order


def test_merge_tolerates_bin_jitter_and_wraps_the_cfo_difference(production_merge):
    N = 1024
    a = cand(244.1, 109568, 413)
    b = cand(238.8, 109144, 1013)             # start -424, bin +600 == -424
    c = cand(224.4, 110128, 975)              # start +560, bin +562 (2 bins off)
    _, downchirp = chirps(10, 250000)
    assert production_merge([a, b, c], N, noise(200 * N), downchirp, 8) == [a]


def test_merge_prefers_the_alias_whose_header_margin_is_clearly_better(production_merge):
    # One preamble, two grids: the score winner sits a quarter symbol late
    # (its header still checks, as on the SF10/250k capture), the other on
    # the true boundary.  Header margin decides; a near-tie keeps the score.
    sf, bw, N = 9, 125000, 512
    true_start = 3 * N
    iq1 = frame(sf, bw, 30, 1, True, 40, true_start, 70 * N, noise=0.05)
    _, downchirp = chirps(sf, bw)
    late = cand(240.0, true_start + 120, 120, pre_last_i=7)
    aligned = cand(235.0, true_start, 0, pre_last_i=7)
    assert production_merge([late, aligned], N, iq1, downchirp, sf - 2) == [aligned]
    nearly = cand(235.0, true_start + 12, 12, pre_last_i=7)
    assert production_merge([late, nearly], N, iq1, downchirp, sf - 2) == [nearly]
    close_pair = [cand(240.0, true_start + 12, 12, pre_last_i=7),
                  cand(235.0, true_start, 0, pre_last_i=7)]
    assert production_merge(close_pair, N, iq1, downchirp, sf - 2) == [close_pair[0]]


def test_runner_up_attempt_hands_a_banned_group_to_its_next_alias(production_merge):
    sf, bw, N = 9, 125000, 512
    true_start = 3 * N
    iq1 = frame(sf, bw, 30, 1, True, 40, true_start, 70 * N, noise=0.05)
    _, downchirp = chirps(sf, bw)
    winner = cand(240.0, true_start + 12, 12, pre_last_i=7)
    other = cand(235.0, true_start - 100, N - 100, pre_last_i=7)   # 112 bins away
    # Ordinary attempt: the banned winner still represents the group (the
    # caller's all-filtered fallback decides), as before.
    assert production_merge([winner, other], N, iq1, downchirp, sf - 2,
                            skip_bins=[12]) == [winner]
    # Runner-up attempt: the group passes to the first unbanned alias.
    assert production_merge([winner, other], N, iq1, downchirp, sf - 2,
                            skip_bins=[12], alias_skip=True) == [other]


# ------------------------------------------------------ runner-up rung ban
@pytest.fixture(scope='module')
def production_recenter():
    tree = ast.parse(Path(decoder.__file__).read_text())
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_process_file_label')
    fn = next(n for n in process.body if isinstance(n, ast.FunctionDef)
              and n.name == '_recenter_and_decode')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, '<production recenter rung>', 'exec')


def run_recenter(code, first_diag, refined_bin=1500):
    """One rung; the first attempt fails after reading a header and returns a
    REFINED bin far from the finder's chosen bin (SFD re-anchor / descent)."""
    calls = []

    def attempt(iq, *args, skip_bins=None, diag_out=None, alias_skip=False, **kw):
        calls.append(dict(skip_bins=list(skip_bins or []), alias_skip=alias_skip))
        if len(calls) == 1:
            diag_out.update(first_diag)
            return ('FAIL', 8192, 20 * 4096, refined_bin)
        return ('FAIL', 8192, 20 * 4096, 7)

    namespace = dict(os=os, time=SimpleNamespace(process_time=lambda: 0.0),
                     _BUDGET_S=0, _process_start=0.0, _ddl=None, skip_bins=[3],
                     recrop_centered=lambda sh: np.zeros(8, dtype=np.complex64),
                     _decode_attempt=attempt, sf=12, bw=62500, N=32768, ppm=10,
                     fs=500000, dec=8, name='synthetic.cf32', Counter=Counter)
    exec(code, namespace)
    namespace['_recenter_and_decode'](27420.0)
    return calls


def test_runner_up_attempt_bans_the_chosen_finder_bin_not_the_refined_one(production_recenter):
    diag = dict(header_decoded=True, n_candidates=4, alt_candidates=1, chosen_bin=2687,
                runner_up_ratio=1.39)
    calls = run_recenter(production_recenter, diag)
    assert len(calls) == 2 and calls[1]['alias_skip']
    assert 2687 in calls[1]['skip_bins'] and 1500 in calls[1]['skip_bins']
    assert calls[0]['skip_bins'] == [3]


def test_runner_up_attempt_needs_a_candidate_outside_the_ban_radius(production_recenter):
    # Four finder candidates that all alias the chosen bin: the retry could
    # only re-pick the winner (the all-filtered fallback), so it is skipped.
    diag = dict(header_decoded=True, n_candidates=4, alt_candidates=0, chosen_bin=2687,
                runner_up_ratio=1.39)
    assert len(run_recenter(production_recenter, diag)) == 1
    # Attempts that never recorded a chosen bin (header-only paths) skip too.
    diag = dict(header_decoded=True, n_candidates=4, alt_candidates=1, runner_up_ratio=1.39)
    assert len(run_recenter(production_recenter, diag)) == 1


def test_runner_up_attempt_is_only_for_an_alias_that_read_at_least_as_well(production_recenter):
    base = dict(header_decoded=True, n_candidates=4, alt_candidates=1, chosen_bin=2687)
    # SF12/62.5k live crops: the absorbed alias read no header (0) or a
    # weaker one (0.48) at the finder grid — no runner-up attempt.
    for ratio in (0.0, 0.48, 0.99):
        assert len(run_recenter(production_recenter, dict(base, runner_up_ratio=ratio))) == 1
    # The near-tie the merge left in score order, and an alias that read a
    # header where the winner's finder-grid header did not check.
    for ratio in (1.0, 1.39, float('inf')):
        assert len(run_recenter(production_recenter, dict(base, runner_up_ratio=ratio))) == 2
    assert len(run_recenter(production_recenter, base)) == 1     # no ratio recorded
