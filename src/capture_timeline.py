"""Bounded shared live capture storage, independent of detector/decoder imports.

The producer publishes metadata only. A background service reads validated raw
history, crops each band once, and transfers ready views to a nonblocking save
sink. The sink owns its view/workspace lease until release, including failures.
"""
from dataclasses import dataclass, field
import math
import threading
import time

import numpy as np


class CaptureCapacityError(RuntimeError):
    pass


class SharedCaptureFailure(RuntimeError):
    pass


class _Lease:
    def __init__(self, budget, kind, size, slots):
        self._budget, self.kind = budget, kind
        self.size, self.slots = size, slots
        self._released = False

    def resize(self, size):
        self._budget._resize(self, int(size))

    def release(self):
        self._budget._release(self)


class CaptureBudget:
    """Atomic byte/outbox reservations. No reservation waits for capacity."""
    def __init__(self, total_bytes, outbox_limit):
        self.total_bytes = int(total_bytes)
        self.outbox_limit = int(outbox_limit)
        if self.total_bytes <= 0 or self.outbox_limit <= 0:
            raise ValueError('capture limits must be positive')
        self._lock = threading.Lock()
        self._bytes = self._slots = self._peak = 0
        self._kinds = {}

    def reserve(self, kind, size, *, slots=0):
        size, slots = int(size), int(slots)
        if size < 0 or slots < 0:
            raise ValueError('negative capture reservation')
        with self._lock:
            if self._bytes + size > self.total_bytes or self._slots + slots > self.outbox_limit:
                raise CaptureCapacityError(f'{kind} needs {size} bytes/{slots} slots; '
                                           f'used {self._bytes}/{self.total_bytes} bytes, '
                                           f'{self._slots}/{self.outbox_limit} slots')
            lease = _Lease(self, kind, size, slots)
            self._bytes += size; self._slots += slots
            self._kinds[kind] = self._kinds.get(kind, 0) + size
            self._peak = max(self._peak, self._bytes)
            return lease

    def _resize(self, lease, size):
        if size < 0:
            raise ValueError('negative capture reservation')
        with self._lock:
            if lease._released:
                raise ValueError('released capture reservation')
            delta = size - lease.size
            if self._bytes + delta > self.total_bytes:
                raise CaptureCapacityError(f'{lease.kind} growth exceeds capture byte budget')
            self._bytes += delta
            self._kinds[lease.kind] += delta
            self._peak = max(self._peak, self._bytes)
            lease.size = size

    def _release(self, lease):
        with self._lock:
            if lease._released:
                return
            lease._released = True
            self._bytes -= lease.size; self._slots -= lease.slots
            self._kinds[lease.kind] -= lease.size
            lease.size = lease.slots = 0

    def stats(self):
        with self._lock:
            return {'used_bytes': self._bytes, 'peak_bytes': self._peak,
                    'limit_bytes': self.total_bytes, 'outbox_slots': self._slots,
                    'outbox_limit': self.outbox_limit, 'by_kind': dict(self._kinds)}


@dataclass(eq=False)
class _Band:
    center: float
    factor: int
    target_factor: int
    origin: int
    identity: tuple
    credit: object
    refs: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    cursor: int = 0
    produced: int = 0
    decimator: object = None
    started: bool = False
    busy: bool = False
    retired_before: int = 0


@dataclass(eq=False)
class _Reference:
    band: _Band
    index: int
    start: int
    end: int
    context_end: int
    center: float


class SharedSave:
    """One ready batch; the save worker releases it in its finally block."""
    def __init__(self, owner, number, detections, start, end, target_factor,
                 export_bw, metadata, credit, role='shared_full'):
        self._owner = owner
        self.number = number
        self.detections = detections
        self.start, self.end = start, end
        self.target_factor, self.export_bw = target_factor, export_bw
        self.metadata = metadata
        self.credit = credit
        self.role = role
        self.refs = []
        self.state = 'collecting'
        self.complete = False
        self._released = False
        self._render_failed = False
        self.ready_at = None
        self._transfer_at = None
        self._full_deferred_at = None
        self._full_defer_s = 0.0

    @property
    def nb_fs(self):
        return self._owner.rate / self.target_factor

    def stream(self, index):
        """Save-worker-only NB recrop, after ordinary save-side dedup."""
        try:
            return self._owner._render(self, index)
        except Exception as exc:
            self._render_failed = True
            if self.role not in ('header_prefix', 'early'):
                self._owner._fail(f'view {self.number} recrop failed: {exc}')
            raise

    def release(self):
        self._owner._release_view(self)


class _HeaderHold:
    """Pin the full reference while an optional header decision is in flight."""
    state = 'holding'

    def __init__(self, owner, ref, credit):
        self.owner, self.ref, self.credit = owner, ref, credit

    def release(self):
        with self.owner._lock:
            if self.state == 'done':
                return
            self.state = 'done'
            self.ref.band.refs = [(v, r) for v, r in self.ref.band.refs if v is not self]
            self.credit.release()
            self.owner._retire()
            self.owner._release_idle_credits()
        self.owner._wake.set()


class SharedCapture:
    """Shared bands and bounded view outbox for an existing raw history ring.

    register/advance never run FIRs, wait for queues, or perform file I/O.
    step is the worker service boundary and is also usable by deterministic
    tests. submit must be a nonblocking ownership transfer: false retains this
    object's lease, true transfers it until SharedSave.release().
    """
    def __init__(self, *, rate, center, hop_n, reader, decimator_factory,
                 submit, budget, report=None, threaded=True,
                 writer_stall_s=30.0, clock=time.monotonic):
        self.rate, self.center = float(rate), float(center)
        self.hop_n = int(hop_n)
        if self.rate <= 0 or self.hop_n <= 0:
            raise ValueError('invalid shared capture clock')
        self.reader, self.factory, self.submit = reader, decimator_factory, submit
        # Older/fake history readers retain explicit consumed-watermark feeds.
        # The opt-in live reader supplies both ends of the published-copy API.
        self._published_history = all(callable(getattr(reader, name, None)) for name in
            ('received_history_anchor', 'received_history_parts'))
        self.budget = budget
        self.clock = clock
        self.writer_stall_s = float(writer_stall_s)
        if not math.isfinite(self.writer_stall_s) or self.writer_stall_s <= 0:
            raise ValueError('writer stall bound must be finite and positive')
        self.report = report or (lambda message: None)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._bands, self._views = [], []
        self._watermark = None
        self._writing = None
        self._optional_burst = 0
        self._full_write_s = 0.0
        self._next = self._completed = self._incomplete = 0
        self._optional_completed = self._optional_incomplete = 0
        self._closing = False
        self.failure = None
        self.header_controller = None
        self._pending_report = None
        self._pending_notice = None
        self._non_admitted = 0
        self._last_status = self.clock()
        # Includes input conversion, mix/concat/FIR temporaries. Each band's
        # previous full-hop _hist backing owner is charged separately below.
        # _StreamDecimator's mixer builds an8192-sample phasor even for a
        # smaller feed. Charge that fixed block in both processing stages.
        self._workspace_hop_n = max(self.hop_n, 8192)
        self._crop_credit = budget.reserve('crop_workspace', 48 * self._workspace_hop_n + 65536)
        self._writer_credit = budget.reserve('save_workspace', 0)
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name='shared-capture')
            try:
                self._thread.start()
            except Exception:
                self._crop_credit.release(); self._writer_credit.release()
                raise

    @staticmethod
    def _identity(anchor):
        if anchor is None or len(anchor) != 3:
            raise SharedCaptureFailure('missing raw history identity')
        return int(anchor[1]), int(anchor[2])  # drop epoch AND retune generation

    def advance(self, anchor):
        """Publish a validated raw watermark without touching sample arrays."""
        try:
            identity = self._identity(anchor)
            with self._lock:
                if self._watermark is not None and identity != self._identity(self._watermark):
                    self._fail_locked('raw stream skipped or retuned; shared capture identity ended')
                if self._watermark is None or int(anchor[0]) >= int(self._watermark[0]):
                    self._watermark = tuple(anchor)
        except Exception as exc:
            self._fail(str(exc))
        self._wake.set()

    def _poll_received_history(self):
        if not self._published_history:
            return
        with self._lock:
            if self._closing or self.failure:
                return
        try:
            anchor = self.reader.received_history_anchor()
            identity = self._identity(anchor)
            received = int(anchor[0])
        except Exception as exc:
            self._fail(f'published history snapshot failed: {exc}')
            return
        with self._lock:
            # close may have frozen the final published boundary while this
            # snapshot was in flight. Never extend a closed generation.
            # advance wakes the service. Do not wake our own idle loop for an
            # unchanged/stale position; identity changes must still propagate.
            changed = (self._watermark is None
                or identity != self._identity(self._watermark)
                or received > int(self._watermark[0]))
            if not self._closing and not self.failure and changed:
                self.advance(anchor)

    def _spec(self, target_factor, frequency):
        target_factor = int(target_factor)
        if target_factor < 1:
            raise ValueError('invalid target factor')
        shared = next((d for d in range(target_factor // 2, 0, -1)
                       if target_factor % d == 0), 1)
        wanted_half = self.rate / (2 * target_factor)
        if target_factor == 1:
            return 1, self.center
        if shared == 1 and target_factor > 1:
            # A prime D (e.g. D5 at10Msps) would otherwise expand a short
            # 2Msps view into a 10Msps retained band. Keep its original FIR
            # and exact center instead; only identical-center views share it.
            shared, center, support = target_factor, frequency, wanted_half
        elif shared == 1:
            center, support = self.center, self.rate / 2
        else:
            center = round(frequency / wanted_half) * wanted_half
            support = .75 * self.rate / (2 * shared)
        if (abs(frequency - center) + wanted_half > support + 1e-6
                or abs(center - self.center) + support > self.rate / 2 + 1e-6):
            # The payload can be in-band while its oversampled export or
            # wider shared plan crosses Nyquist. Preserve the original
            # exact-rate/center filter instead of rejecting that real signal.
            return target_factor, frequency
        return shared, center

    def _support(self, band):
        return ((1 if band.factor in (1, band.target_factor) else .75)
                * self.rate / (2 * band.factor))

    def register(self, detections, *, base_start, base_end, end, anchor,
                 target_factor, export_bw, pre_hop_n, wb_n, sample_pos):
        """Admit complete future-span reservations before publishing a view."""
        if not detections:
            return None
        lease = None
        view = None
        try:
            identity = self._identity(anchor)
            with self._lock:
                if self.failure or self._closing:
                    return None
                if self._watermark is not None and identity != self._identity(self._watermark):
                    raise SharedCaptureFailure('capture registration has stale raw identity')
                if end <= base_start or base_end < base_start:
                    raise ValueError('invalid capture range')
                useful = []
                for det in detections:
                    signal_half = float(det.get('_capture_signal_bw', det['bw'])) / 2
                    if (math.isfinite(float(det['freq_hz'])) and signal_half > 0
                            and abs(det['freq_hz'] - self.center) + signal_half
                            <= self.rate / 2 + 1e-6):
                        useful.append(det)
                    else:
                        self._non_admitted += 1
                        self._pending_notice = ('[SHARED-CAPTURE NON-ADMISSION] '
                            f"RF footprint outside input: freq={det['freq_hz']} "
                            f"bw={2*signal_half:g}; count={self._non_admitted}")
                detections = useful
                if not detections:
                    self._wake.set()
                    return None
                # Exact request duplicates share the existing descriptor. No
                # frequency/time heuristic discards a distinct packet span.
                signature = tuple(tuple(sorted((key, repr(value))
                                      for key, value in d.items()))
                                  for d in detections)
                for old in self._views:
                    if (old.state not in ('done', 'cancelled') and old.start == base_start
                            and old.end == end and old.metadata['signature'] == signature):
                        return old.number
                lease = self.budget.reserve('view_metadata', 4096 + 4096 * len(detections), slots=1)
                metadata = dict(pre_hop_n=pre_hop_n, wb_n=wb_n, sample_pos=sample_pos,
                                base_start=base_start,
                                base_end=base_end, signature=signature, identity=identity)
                view = SharedSave(self, self._next, [dict(d) for d in detections],
                                  int(base_start), int(end), int(target_factor),
                                  export_bw, metadata, lease)
                self._next += 1
                # One save worker at a time owns this reusable workspace.
                # Covers original/corrected arrays and existing save recenter
                # temporaries, plus a bounded chunk-wise second-stage FIR.
                out_bytes = math.ceil((end - base_start) / target_factor) * 8
                # Shared saves mix their private output in <=min(65536, hop_n)
                # tiles. Retain two complete outputs for the prior preamble
                # during the next render, or the same-dtype file-write copy.
                # The fixed allowance covers second-stage FIR and tile phase/
                # complex128 mix temporaries, even for small input hops.
                needed_workspace = 2 * out_bytes + 64 * self._workspace_hop_n + 65536
                if needed_workspace > self._writer_credit.size:
                    self._writer_credit.resize(needed_workspace)
                for index, det in enumerate(view.detections):
                    factor, center = self._spec(target_factor, float(det['freq_hz']))
                    wanted_half = self.rate / (2 * target_factor)
                    band = next((b for b in self._bands
                                 if b.identity == identity and b.target_factor == target_factor
                                 and b.factor == factor
                                 and base_start >= b.origin + b.retired_before * b.factor
                                 and (factor == 1 or abs(det['freq_hz'] - b.center) + wanted_half
                                      <= self._support(b) + 1e-6)), None)
                    if band is None:
                        credit = self.budget.reserve('band_retention', 0)
                        band = _Band(center, factor, target_factor, int(base_start), identity,
                                     credit, cursor=int(base_start))
                        self._bands.append(band)
                    start = -(-(int(base_start) - band.origin) // factor)
                    stop = -(-(int(end) - band.origin) // factor)
                    # Second-stage real right context; shared FIR itself also
                    # needs10D raw lookahead before emitting these samples.
                    second_factor = target_factor // factor
                    context = stop + (10 * second_factor if second_factor > 1 else 0)
                    ref = _Reference(band, index, start, stop, context, det['freq_hz'])
                    band.refs.append((view, ref)); view.refs.append(ref)
                    self._reserve_band(band)
                    det['_nb_ix'] = index
                    det['_capture_start_offset_s'] = (band.origin + start * factor - base_start) / self.rate
                self._views.append(view)
                if self._watermark is None or int(anchor[0]) >= int(self._watermark[0]):
                    self._watermark = tuple(anchor)
            self._wake.set()
            return view.number
        except Exception as exc:
            with self._lock:
                if view is not None:
                    for ref in view.refs:
                        ref.band.refs = [(v, r) for v, r in ref.band.refs if v is not view]
                if lease is not None:
                    lease.release()
                self._incomplete += 1
                self._fail_locked(f'capture admission failed: {exc}')
            return None

    def _reserve_band(self, band):
        live = [(v, r) for v, r in band.refs if v.state not in ('done', 'cancelled')]
        if not live:
            return
        earliest = min(r.start for _, r in live)
        latest = max(r.context_end for _, r in live)
        # Chunks may straddle the earliest view by at most one input hop.
        span = latest - min(earliest, band.retired_before)
        wanted = ((span + math.ceil(self.hop_n / band.factor)) * 8
                  + self._band_overhead(band))
        band.credit.resize(max(wanted, band.credit.size))

    def header_candidates(self):
        """Small descriptors only; never read or copy samples on this boundary."""
        with self._lock:
            if self.failure or self._closing:
                return []
            return [(view, ref) for view in self._views
                    if view.role == 'shared_full' and view.state == 'collecting'
                    for ref in view.refs]

    def hold_header_reference(self, view, ref):
        with self._lock:
            if self.failure or self._closing or view._released or ref not in view.refs:
                raise SharedCaptureFailure('header source no longer available')
            hold = _HeaderHold(self, ref, self.budget.reserve(
                'header_hold', 4096 + 4096 * len(view.detections)))
            ref.band.refs.append((hold, ref))
            return hold

    def header_token(self, view, ref, begin_raw, end_raw):
        band = ref.band
        return (id(self), view.number, ref.index, *band.identity, band.origin,
                band.factor, band.center, view.target_factor, ref.center,
                self.rate, int(view.detections[ref.index]['sf']),
                float(view.detections[ref.index]['bw']), int(begin_raw), int(end_raw))

    def header_source_valid(self, hold, token):
        with self._lock:
            band = hold.ref.band
            return (not self.failure and not self._closing and hold.state != 'done'
                    and band in self._bands and self._watermark is not None
                    and self._identity(self._watermark) == band.identity
                    and tuple(token[3:5]) == band.identity
                    and (token[0], token[5], token[6], token[7], token[10])
                        == (id(self), band.origin, band.factor, band.center, self.rate))

    def add_slice_view(self, parent, ref, begin_raw, end_raw, *, role, request_id):
        """Optional prefix/early sibling; original full retention never changes."""
        if role not in ('header_prefix', 'early'):
            raise ValueError('invalid internal slice role')
        with self._lock:
            band = ref.band
            if self.failure or self._closing or band not in self._bands:
                raise SharedCaptureFailure('slice source identity ended')
            begin = -(-(int(begin_raw) - band.origin) // band.factor)
            stop = -(-(int(end_raw) - band.origin) // band.factor)
            if begin < ref.start or stop > ref.end or stop <= begin or begin < band.retired_before:
                raise SharedCaptureFailure('slice lies outside retained full reference')
            credit = self.budget.reserve('header_view_metadata', 8192, slots=1)
            try:
                actual_start = band.origin + begin * band.factor
                detection = dict(parent.detections[ref.index])
                detection['_nb_ix'] = 0
                detection['_capture_start_offset_s'] = (
                    actual_start - parent.metadata.get('base_start', parent.start)) / self.rate
                metadata = dict(parent.metadata, header_request_id=request_id,
                                header_parent_id=parent.number,
                                signature=(tuple(sorted((k, repr(v)) for k, v in detection.items())),))
                child = SharedSave(self, self._next, [detection], actual_start, int(end_raw),
                                   parent.target_factor, parent.export_bw, metadata, credit, role)
                factor = child.target_factor // band.factor
                child_ref = _Reference(band, 0, begin, stop,
                                       stop + (10 * factor if factor > 1 else 0), ref.center)
                child.refs.append(child_ref)
                band.refs.append((child, child_ref))
                self._views.append(child)
                self._next += 1
            except Exception:
                credit.release()
                raise
        self._wake.set()
        return child

    def _band_overhead(self, band):
        # Retained FIR history backing owner plus taps/polyphase state and
        # bounded Python band/chunk bookkeeping. D1 has no FIR state.
        return self.hop_n * 8 + 512 * band.factor + 4096

    def _fail_locked(self, message):
        if self.failure is None:
            self.failure = str(message)
            self._pending_report = '[SHARED-CAPTURE FATAL] ' + self.failure
        for view in self._views:
            if view.state in ('done', 'cancelled'):
                continue
            if view is self._writing:
                view._render_failed = True
                continue  # the sink still owns its arrays/lease
            view.state = 'cancelled'; view.complete = False
            if view.role in ('header_prefix', 'early'):
                self._optional_incomplete += 1
            else:
                self._incomplete += 1
            view.credit.release()
        self._wake.set()

    def _fail(self, message):
        with self._lock:
            self._fail_locked(message)

    def raise_if_failed(self):
        if self.failure:
            raise SharedCaptureFailure(self.failure)

    def _retire(self):
        for band in list(self._bands):
            band.refs = [(v, r) for v, r in band.refs if v.state not in ('done', 'cancelled')]
            if band.busy:
                continue
            if not band.refs:
                band.chunks.clear(); band.decimator = None
                band.credit.release(); self._bands.remove(band)
                continue
            earliest = min(r.start for _, r in band.refs)
            band.chunks = [(offset, a) for offset, a in band.chunks if offset + len(a) > earliest]
            band.retired_before = band.chunks[0][0] if band.chunks else band.produced
            latest = max(r.context_end for _, r in band.refs)
            wanted = ((latest - band.retired_before + math.ceil(self.hop_n / band.factor)) * 8
                      + self._band_overhead(band))
            band.credit.resize(max(0, wanted))
        self._views = [v for v in self._views if v.state not in ('done', 'cancelled')]

    def _defer_full_for_early(self, view, available):
        """Leave the idle writer free briefly for an almost-collected early view.

        Only measured successful full-save service time justifies this wait.
        The first wait deadline belongs to the full view, never to later arrivals.
        Caller holds the metadata lock; no sample work or sleeping occurs here.
        """
        cost = self._full_write_s
        if self._closing or self.failure or self._optional_burst >= 2 or cost <= 0:
            return False
        now = self.clock()
        wait = min(cost, 5.0, self.writer_stall_s / 4)
        if view._full_deferred_at is not None:
            wait = view._full_defer_s - (now - view._full_deferred_at)
        if wait <= 0:
            return False
        # Do not voluntarily consume time needed to finish this already-aged
        # full view before the existing writer-stall deadline.
        if now - view.ready_at + wait + cost >= self.writer_stall_s:
            return False
        imminent = any(
            early.role == 'early' and early.state == 'collecting'
            and max((max(0, ref.band.origin + ref.context_end * ref.band.factor
                         + (10 * ref.band.factor if ref.band.factor > 1 else 0)
                         - available) for ref in early.refs), default=math.inf)
                / self.rate <= cost
            for early in self._views)
        if not imminent:
            return False
        if view._full_deferred_at is None:
            view._full_deferred_at = now
            view._full_defer_s = wait
        return True

    def step(self):
        """One bounded crop service turn plus a nonblocking save transfer."""
        self._poll_received_history()
        if self.header_controller is not None:
            self.header_controller.step()
            for event in self.header_controller.drain_events():
                try:
                    import json
                    self.report('[HEADER-CHECKPOINT] ' + json.dumps(event, separators=(',', ':')))
                except Exception:
                    pass
        with self._lock:
            report, self._pending_report = self._pending_report, None
            notice, self._pending_notice = self._pending_notice, None
            report = report or notice
            if (report is None and self._views
                    and self.clock() - self._last_status >= 5):
                self._last_status = self.clock()
                state = self.stats()
                budget = state['budget']
                debt = state['crop_debt_s']
                report = (f"[SHARED-CAPTURE] bands={state['bands']} "
                          f"views={state['active_views']} ready={state['ready_views']} "
                          f"done={self._completed} incomplete={self._incomplete} "
                          f"debt={debt:.3f}s bytes={budget['used_bytes']} "
                          f"peak={budget['peak_bytes']} slots={budget['outbox_slots']} "
                          f"non_admitted={self._non_admitted}")
        if report is not None:
            # A blocked diagnostic pipe must never hold the producer lock.
            try:
                self.report(report)
            except Exception:
                pass
        selected = None
        with self._lock:
            self._retire()
            stalled = next((v for v in self._views
                            if v.ready_at is not None and not v._released
                            and self.clock() - v.ready_at > self.writer_stall_s), None)
            if stalled is not None:
                self._fail_locked(f'writer stalled beyond {self.writer_stall_s:g}s '
                                  f'for admitted view {stalled.number}')
            if self.failure:
                self._release_idle_credits()
                return False
            if self._watermark is None:
                return False
            available = int(self._watermark[0])
            # A delayed slow result may request older history than a newer
            # fast band. Recover the oldest missing raw interval first so
            # insertion order cannot cause an avoidable ring lap.
            for band in sorted(self._bands, key=lambda b: b.cursor):
                if not band.refs or band.busy:
                    continue
                through = band.origin + max(r.context_end for _, r in band.refs) * band.factor
                through += 10 * band.factor if band.factor > 1 else 0
                target = min(through, available)
                if band.cursor < target:
                    n = min(self.hop_n, target - band.cursor)
                    band.busy = True; band.started = True
                    selected = band, band.cursor, n
                    break
        if selected is not None:
            band, start, n = selected
            try:
                if band.factor > 1 and band.decimator is None:
                    band.decimator = self.factory(band.center - self.center, self.rate, band.factor)
                history_parts = (self.reader.received_history_parts if self._published_history
                                 else self.reader.history_parts)
                data = next(history_parts((start, *band.identity), n, self.hop_n))
                if band.factor == 1:
                    output = data  # D1 is the actual full-rate original band
                else:
                    output = band.decimator.feed(data)
                # Own the exact output allocation: contiguous views can retain
                # a much larger FIR temporary and must not evade the ledger.
                if output.base is not None:
                    output = output.copy()
                output.flags.writeable = False
                with self._lock:
                    if not self.failure:
                        if len(output):
                            band.chunks.append((band.produced, output))
                            band.produced += len(output)
                        band.cursor += n
                    band.busy = False
            except Exception as exc:
                with self._lock:
                    band.busy = False
                    self._fail_locked(f'raw history/crop interval {start}:{start+n} failed: {exc}')
                return False
        transfer = None
        with self._lock:
            for view in self._views:
                if view.state == 'collecting' and all(r.band.produced >= r.context_end for r in view.refs):
                    view.state = 'ready'; view.complete = True
                    view.ready_at = self.clock()
            if self._writing is None:
                ready = [v for v in self._views if v.state == 'ready']
                optional = sorted((v for v in ready if v.role in ('early', 'header_prefix')),
                                  key=lambda v: (v.role != 'early', v.number))
                full = [v for v in ready if v.role not in ('early', 'header_prefix')]
                # At most two optional transfers ahead of a waiting full view.
                # One existing writer is never preempted; acquisition stays free.
                expired = (full and full[0]._full_deferred_at is not None
                    and self.clock() - full[0]._full_deferred_at >= full[0]._full_defer_s)
                if full and (expired or self._optional_burst >= 2):
                    transfer = full[0]
                elif optional:
                    transfer = optional[0]
                elif full and not self._defer_full_for_early(full[0], available):
                    transfer = full[0]
                if transfer is not None:
                    self._writing = transfer; transfer.state = 'transferring'
                    transfer._transfer_at = self.clock()
        if transfer is not None:
            try:
                accepted = self.submit(transfer)  # never under producer/ledger lock
            except Exception as exc:
                self._fail(f'save sink rejected view {transfer.number}: {exc}')
                accepted = False
            with self._lock:
                if accepted:
                    self._optional_burst = (self._optional_burst + 1
                        if transfer.role in ('early', 'header_prefix') else 0)
                    if transfer.state != 'done':
                        transfer.state = 'writing'
                else:
                    self._writing = None
                    transfer._transfer_at = None
                    transfer.state = 'ready' if not self.failure else 'cancelled'
                    if self.failure:
                        self._release_view(transfer)
        with self._lock:
            if self._closing and selected is None:
                incomplete = [v for v in self._views if v.state == 'collecting']
                if incomplete:
                    self._fail_locked('input ended before admitted capture endpoints/right context')
            self._retire(); self._release_idle_credits()
        return selected is not None or bool(transfer is not None and accepted)

    def _render(self, view, index):
        with self._lock:
            if view is not self._writing or not view.complete or view._released:
                raise SharedCaptureFailure('save view is not owned/stable')
            ref = next(r for r in view.refs if r.index == index)
            band = ref.band
            parts = []
            cursor = ref.start
            for offset, data in band.chunks:
                lo, hi = max(ref.start, offset), min(ref.context_end, offset + len(data))
                if hi <= lo:
                    continue
                if lo != cursor:
                    raise SharedCaptureFailure('view crosses retired or missing NB samples')
                parts.append(data[lo-offset:hi-offset]); cursor = hi
            if cursor != ref.context_end:
                raise SharedCaptureFailure('view lacks emitted right context')
            raw_start = band.origin + ref.start * band.factor
            size = math.ceil((view.end - raw_start) / view.target_factor)
            factor = view.target_factor // band.factor
            phase = np.complex64(np.exp(2j * np.pi * (((band.center-self.center)
                                 * (raw_start-band.origin) / self.rate) % 1)))
            offset = ref.center - band.center
        output = np.empty(size, np.complex64)
        written = 0
        if factor > 1:
            decimator = self.factory(offset, self.rate / band.factor, factor)
            for part in parts:
                chunk = decimator.feed(part * phase)
                take = min(len(chunk), size - written)
                output[written:written+take] = chunk[:take]; written += take
                if written == size:
                    break
        else:
            for part in parts:
                take = min(len(part), size-written)
                n = np.arange(written, written+take, dtype=np.float64)
                output[written:written+take] = part[:take] * phase * np.exp(
                    -2j*np.pi*((n*offset/(self.rate/band.factor)) % 1)).astype(np.complex64)
                written += take
                if written == size:
                    break
        if written != size:
            raise SharedCaptureFailure('recrop cannot finish without invented zero padding')
        # A concurrent skip/retune must not publish a result under the old
        # identity. Its writer-owned allocations remain charged until release.
        self.raise_if_failed()
        return output

    def _release_view(self, view):
        with self._lock:
            if view._released:
                return
            view._released = True
            if view._render_failed or self.failure:
                if view.role in ('header_prefix', 'early'):
                    self._optional_incomplete += 1
                else:
                    self._incomplete += 1
            else:
                if view.role in ('header_prefix', 'early'):
                    self._optional_completed += 1
                else:
                    self._completed += 1
                    if view._transfer_at is not None:
                        elapsed = self.clock() - view._transfer_at
                        if math.isfinite(elapsed) and elapsed > 0:
                            self._full_write_s = elapsed
            view.state = 'done'; view.credit.release()
            if self._writing is view:
                self._writing = None
            self._retire(); self._release_idle_credits()
        self._wake.set()

    def _release_idle_credits(self):
        if (self._closing or self.failure) and not self._bands and self._writing is None:
            self._crop_credit.release(); self._writer_credit.release()

    def _run(self):
        while True:
            try:
                did_work = self.step()
            except Exception as exc:
                self._fail(f'capture service failed: {exc}')
                did_work = False
            with self._lock:
                done = (self._closing or self.failure) and not self._bands and self._writing is None
            if done:
                return
            if not did_work:
                self._wake.wait(.02); self._wake.clear()

    def close(self, *, wait=True):
        if self.header_controller is not None:
            self.header_controller.close(wait=wait)
        with self._lock:
            if self._published_history and not self._closing and not self.failure:
                try:
                    # Freeze once at close, including already-published final
                    # partial-hop context. Later producer writes are excluded.
                    self.advance(self.reader.received_history_anchor())
                except Exception as exc:
                    self._fail_locked(f'final published history snapshot failed: {exc}')
            self._closing = True
            self._retire(); self._release_idle_credits()
        self._wake.set()
        if wait and self._thread is not None and not self.failure:
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                self._fail('capture writer/service did not drain within shutdown bound')

    def stats(self):
        with self._lock:
            available = int(self._watermark[0]) if self._watermark is not None else 0
            debt = max((max(0, min(available, b.origin
                + max((r.context_end for _, r in b.refs), default=0) * b.factor
                + (10 * b.factor if b.factor > 1 else 0)) - b.cursor)
                for b in self._bands if b.refs), default=0) / self.rate
            return {'bands': len(self._bands),
                    'crop_debt_s': debt,
                    'active_views': sum(v.state not in ('done','cancelled') for v in self._views),
                    'ready_views': sum(v.state == 'ready' for v in self._views),
                    'completed_views': self._completed, 'incomplete_views': self._incomplete,
                    'optional_completed_views': self._optional_completed,
                    'optional_incomplete_views': self._optional_incomplete,
                    'non_admitted': self._non_admitted,
                    'failure': self.failure, 'budget': self.budget.stats(),
                    'band_details': [{'center': b.center, 'factor': b.factor,
                                      'origin': b.origin, 'cursor': b.cursor,
                                      'produced': b.produced, 'views': len(b.refs)}
                                     for b in self._bands]}
