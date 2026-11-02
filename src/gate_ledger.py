"""Gate decisions as data: one JSON line per detector window.

The gate makes several decisions about every energy peak before the detect
stage ever sees it (prescreen, spur-width drop, per-window cap, routing), and
the detect stage then makes its own (which bandwidth classes it evaluated,
the best Schmidl-Cox score, whether a detection came out).  Until now those
decisions existed only as rate-limited stderr lines, so a transmission the
pipeline never published could not be told apart from RF that never arrived
(live runs 2026-10-31: SF7/125k first transmissions lost at the gate with no
line explaining why).  The ledger records every window: counts for all peaks,
and for each strong peak (strong_db over the window floor) its fate and the
detect stage's verdict.  An audit looks up the window covering a transmission
and reads what happened to its carrier.

Written by the gate thread only.  A line is 200-800 bytes; the file rotates
to `.1` when it passes max_bytes.
"""
import json
import os


class GateLedger:
    STRONG_DB = 20.0

    def __init__(self, path, strong_db=STRONG_DB, max_bytes=20 << 20):
        self.path = path
        self.strong_db = float(strong_db)
        self.max_bytes = int(max_bytes)
        self._fh = None
        self.written = 0

    # ---- record construction (gate thread) ----
    def begin(self, seq, t_end_s, win_s, floor_db, fres_hz, center_hz):
        """Start a window record.  t_end_s: absolute stream time at which the
        window ends (the gate's sample count), win_s: its length, so t0..t1 is
        the span a transmission's firmware timestamp is looked up in;
        floor_db: the window's PSD floor; fres_hz: gate bin width."""
        return {'seq': int(seq), 't0': round(float(t_end_s) - float(win_s), 3),
                't1': round(float(t_end_s), 3),
                'floor': round(float(floor_db), 1),
                '_fres': float(fres_hz), '_c': float(center_hz),
                'scan': 0, 'prescreen': 0, 'doomed': 0, 'shed': 0, 'forced': 0,
                'cap': None, 'route': None, 'peaks': []}

    def _strong(self, rec, peaks):
        out = []
        for p in peaks:
            floor = p[3] if len(p) > 3 and p[3] is not None else rec['floor']
            elev = float(p[2]) - float(floor)
            if elev >= self.strong_db:
                out.append({'f': round((rec['_c'] + (float(p[0]) - 2048.0) * rec['_fres']) / 1e6, 4),
                            'w': int(p[1]), 'e': int(round(elev))})
        return out

    def scanned(self, rec, peaks):
        rec['scan'] = len(peaks)

    def removed(self, rec, stage, peaks):
        """Peaks the gate took out at `stage` ('prescreen', 'doomed', 'shed')."""
        rec[stage] += len(peaks)
        for p in self._strong(rec, peaks):
            p['fate'] = stage
            rec['peaks'].append(p)

    def forced(self, rec, n):
        rec['forced'] += int(n)

    def dispatched(self, rec, peaks, route, cap):
        """The list the detect stage receives, and how it travels
        ('inline', 'pool', 'skipped', 'quiet')."""
        rec['route'] = route
        rec['cap'] = cap
        for p in self._strong(rec, peaks):
            p['fate'] = 'kept'
            rec['peaks'].append(p)

    def outcomes(self, rec, outcomes):
        """Attach the detect stage's per-peak verdicts (see
        detector.detect_preamble's `outcomes`)."""
        if not outcomes:
            return
        if isinstance(outcomes, dict):
            if outcomes.get('cap_override'):
                rec['cap_override'] = outcomes['cap_override']
            outcomes = outcomes.get('peaks', [])
        by_f = {p['f']: p for p in rec['peaks'] if p.get('fate') == 'kept'}
        for o in outcomes:
            target = by_f.get(round(o['f'], 4))
            if target is None:
                # a peak below strong_db: its verdict matters only when it
                # became a detection (a weak peak's sc-below is the expected
                # case and would swamp the record)
                if o.get('r') != 'detected':
                    continue
                target = {'f': round(o['f'], 4), 'w': o.get('w'), 'e': o.get('e'), 'fate': 'kept'}
                rec['peaks'].append(target)
            for k in ('r', 'sc', 'lag', 'thr', 'det'):
                if k in o and o[k] is not None:
                    target[k] = o[k]

    # ---- output ----
    def write(self, rec):
        if self._fh is None:
            self._open()
        line = json.dumps({k: v for k, v in rec.items() if not k.startswith('_')},
                          separators=(',', ':'))
        self._fh.write(line + '\n')
        self._fh.flush()
        self.written += len(line) + 1
        if self.written >= self.max_bytes:
            self._rotate()

    def _open(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(self.path, 'a', buffering=1)
        try:
            self.written = os.fstat(self._fh.fileno()).st_size
        except OSError:
            self.written = 0

    def _rotate(self):
        self._fh.close()
        self._fh = None
        try:
            os.replace(self.path, self.path + '.1')
        except OSError:
            pass
        self.written = 0

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def ledger_from_environ(environ, debug, default_dir):
    """The ledger is on at debug >= 1 (the web deployment's level) unless
    LORA_GATE_LEDGER is set to '' ; LORA_GATE_LEDGER=path chooses the file."""
    path = environ.get('LORA_GATE_LEDGER')
    if path is None:
        if debug < 1:
            return None
        path = os.path.join(default_dir, 'lora_gate_ledger.jsonl')
    if not path:
        return None
    return GateLedger(path)
