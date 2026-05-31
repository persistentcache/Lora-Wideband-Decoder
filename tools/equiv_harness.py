#!/usr/bin/env python3
"""Decoder equivalence harness.

Dumps a per-attempt fingerprint of decoder intermediate values (preamble
start, header bins, payload nibbles, soft margins, chase flips, final
bytes) to JSONL, then compares two runs symbol-by-symbol so that
candidate algorithmic optimisations can be validated as bit-identical
or provably-equivalent BEFORE running an expensive live test.

Workflow:

  # 1) freeze a reference from the current code on /dev/shm/live_caps
  tools/equiv_harness.py dump --out tests/harness_ref

  # 2) make an optimisation, then check it against the reference
  tools/equiv_harness.py check --ref tests/harness_ref

  # 3) zero diff → safe to live-test; non-zero diff → fix or abandon.

If the optimisation is intentionally non-bit-identical (e.g. a
smaller-N coarse prune), use `--allow-fields margin_med,margin_min` to
ignore drift in soft-margin numerics while still requiring the decoded
nibble + payload-byte sequence to match exactly.
"""

import argparse
import concurrent.futures as _fut
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DECODER = REPO / 'scripts' / 'decode_header_v3.py'
DEFAULT_CAP_DIR = Path('/dev/shm/live_caps')

# Per-event field set that must match exactly between reference and
# current run for the decode path to be considered equivalent.  When a
# user passes --allow-fields, those keys are dropped from comparison.
FINGERPRINT_FIELDS = {
    'attempt_start': ('capture', 'sf', 'bw', 'iq_len', 'skip_bins'),
    'preamble':      ('preamble_start', 'pre_last_i', 'preamble_bin', 'data_start'),
    'header':        ('chosen_td', 'hdr_bins', 'payload_len', 'cr',
                      'crc_present', 'hdr_ok', 'data_start'),
    'primary_decode':('nibs_count', 'first_nibs', 'raw_hex', 'crc_ok',
                      'tau_frac', 'margin_med', 'margin_min', 'margin_n'),
    'chase':         ('used', 'ok', 'method', 'n_flipped', 'flips'),
    'attempt_end':   ('status', 'via', 'payload_hex', 'payload_len', 'crc_ok'),
}


def list_captures(cap_dir):
    return sorted(p for p in Path(cap_dir).glob('*.cf32'))


def run_one_capture(cap, out_path, timeout):
    """Run the decoder on one capture, dumping harness events to out_path."""
    env = os.environ.copy()
    env['LORA_HARNESS_OUT'] = str(out_path)
    # The decoder's PASS 1 / PASS 2 retry loops are gated by a CPU-time
    # budget (LORA_DECODE_BUDGET_S, default 1.5 s × SF scale).  Under
    # variable host load that budget trims attempts non-deterministically
    # — making the harness flap between "5 attempts" and "6 attempts" on
    # the same capture with no code change.  Set a huge budget so every
    # attempt the decoder WOULD make on an idle host runs to completion.
    env['LORA_DECODE_BUDGET_S'] = env.get('LORA_HARNESS_BUDGET_S', '9999')
    # Single-threaded NumPy/BLAS so the harness fingerprint is independent
    # of how many other processes are running on the host.
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '1'
    # Make sure prior run's events don't accumulate.
    if out_path.exists():
        out_path.unlink()
    cmd = [sys.executable, str(DECODER), str(cap)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return cap.name, r.returncode, len(r.stdout), len(r.stderr)
    except subprocess.TimeoutExpired:
        return cap.name, -1, 0, 0


def cmd_dump(args):
    """Run the decoder on every capture and dump the reference set."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = list_captures(args.caps)
    if not caps:
        print('ERROR: no .cf32 captures found in %s' % args.caps)
        return 1
    print('dumping %d captures → %s' % (len(caps), out_dir))
    # Serial — the harness must produce identical output regardless of
    # host load.  Parallel runs slot into the same per-capture file
    # anyway, no speedup.
    for c in caps:
        out_path = out_dir / (c.name + '.jsonl')
        name, rc, no, ne = run_one_capture(c, out_path, args.timeout)
        n_events = sum(1 for _ in open(out_path)) if out_path.exists() else 0
        print('  %-72s rc=%d events=%d' % (name, rc, n_events))
    return 0


def load_events(jsonl_path):
    """Return list of per-attempt event dicts grouped by attempt id."""
    by_attempt = {}
    if not jsonl_path.exists():
        return by_attempt
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_attempt.setdefault(ev['attempt'], []).append(ev)
    return by_attempt


def diff_attempt(ref_events, cur_events, allow_fields):
    """Return list of (event_kind, field, ref_value, cur_value) diffs."""
    diffs = []
    # Build event-kind → ref event dict.
    ref_by_kind = {e['event']: e for e in ref_events}
    cur_by_kind = {e['event']: e for e in cur_events}
    # Did current reach the same depth?
    for kind in ('attempt_start', 'preamble', 'header', 'primary_decode',
                 'chase', 'attempt_end'):
        rk = ref_by_kind.get(kind)
        ck = cur_by_kind.get(kind)
        if rk is None and ck is None:
            continue
        if rk is None and ck is not None:
            diffs.append((kind, '<presence>', 'absent_in_ref', 'present_in_cur'))
            continue
        if ck is None and rk is not None:
            diffs.append((kind, '<presence>', 'present_in_ref', 'absent_in_cur'))
            continue
        for field in FINGERPRINT_FIELDS.get(kind, ()):
            if field in allow_fields:
                continue
            rv = rk.get(field)
            cv = ck.get(field)
            if rv != cv:
                diffs.append((kind, field, rv, cv))
    return diffs


def cmd_check(args):
    """Re-run the decoder and diff each per-capture JSONL vs reference."""
    ref_dir = Path(args.ref)
    if not ref_dir.exists():
        print('ERROR: reference dir %s does not exist (run dump first)' % ref_dir)
        return 2
    cur_dir = Path(args.cur) if args.cur else (REPO / 'tests' / 'harness_cur')
    if cur_dir.exists():
        shutil.rmtree(cur_dir)
    cur_dir.mkdir(parents=True, exist_ok=True)
    caps = list_captures(args.caps)
    allow_fields = set(args.allow_fields.split(',')) if args.allow_fields else set()
    total = 0
    diff_count = 0
    for c in caps:
        ref_path = ref_dir / (c.name + '.jsonl')
        cur_path = cur_dir / (c.name + '.jsonl')
        if not ref_path.exists():
            print('  %-72s SKIP (no reference)' % c.name)
            continue
        run_one_capture(c, cur_path, args.timeout)
        ref = load_events(ref_path)
        cur = load_events(cur_path)
        # Compare attempt-by-attempt.  The attempt ID is monotonic per
        # decoder process; we expect both runs to produce the same set
        # of attempts.
        attempts = sorted(set(ref) | set(cur))
        cap_diffs = []
        for a in attempts:
            d = diff_attempt(ref.get(a, []), cur.get(a, []), allow_fields)
            if d:
                cap_diffs.append((a, d))
        total += 1
        if cap_diffs:
            diff_count += 1
            print('  %-72s DIFF (%d attempts affected)' % (c.name, len(cap_diffs)))
            for a, diffs in cap_diffs[:args.max_per_capture]:
                for kind, field, rv, cv in diffs[:args.max_per_attempt]:
                    print('    attempt=%d %s.%s  ref=%r  cur=%r' % (a, kind, field, rv, cv))
        else:
            print('  %-72s OK' % c.name)
    print()
    print('%s — %d/%d captures equivalent  (%d differ)' % (
        'EQUIVALENT' if diff_count == 0 else 'DRIFT',
        total - diff_count, total, diff_count))
    return 0 if diff_count == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    pd = sub.add_parser('dump', help='Freeze a reference set')
    pd.add_argument('--out', required=True, help='Output dir for reference JSONL')
    pd.add_argument('--caps', default=str(DEFAULT_CAP_DIR),
                    help='Captures dir (default: /dev/shm/live_caps)')
    pd.add_argument('--timeout', type=float, default=120.0,
                    help='Per-capture decode timeout (s)')
    pd.set_defaults(func=cmd_dump)

    pc = sub.add_parser('check', help='Diff current code vs frozen reference')
    pc.add_argument('--ref', required=True, help='Reference dir from dump')
    pc.add_argument('--cur', default=None, help='Where to write current run (default: tests/harness_cur)')
    pc.add_argument('--caps', default=str(DEFAULT_CAP_DIR),
                    help='Captures dir (default: /dev/shm/live_caps)')
    pc.add_argument('--timeout', type=float, default=120.0,
                    help='Per-capture decode timeout (s)')
    pc.add_argument('--allow-fields', default='',
                    help='Comma-list of fields to ignore in diff (e.g. margin_med,margin_min)')
    pc.add_argument('--max-per-capture', type=int, default=3,
                    help='Max differing attempts to print per capture')
    pc.add_argument('--max-per-attempt', type=int, default=5,
                    help='Max differing fields to print per attempt')
    pc.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
