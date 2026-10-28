"""Optional header-driven early siblings; full shared captures remain authoritative.

No acquisition-loop DSP or I/O. The existing save worker renders immutable
prefix files; one bounded header service returns geometry. This controller
only polls small descriptors and adds sample-range references to shared bands.
"""
from dataclasses import dataclass
import math
import os
import tempfile
import threading
import time

from header_checkpoint_service import HeaderRequest, file_identity


class _PrefixLease:
    def __init__(self, credit):
        self.credit, self.path = credit, None
        self._lock = threading.Lock()
        self.released = False

    def release(self):
        with self._lock:
            if self.released:
                return
            if self.path is not None:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
            self.credit.release()
            self.released = True


@dataclass
class _Entry:
    identifier: int
    parent: object
    ref: object
    hold: object
    token: tuple
    prefix: object
    lease: object
    state: str = 'preparing'
    request: object = None


class HeaderCaptureController:
    """Bound optional checkpoint work; a header never shortens a full view."""
    def __init__(self, owner, service, *, directory, max_prefix_bytes,
                 max_pending=2, cpu_budget_s=2., wall_timeout_s=10.):
        self.owner, self.service = owner, service
        self.directory = os.fspath(directory)
        self.max_prefix_bytes = int(max_prefix_bytes)
        self.max_pending = int(max_pending)
        self.cpu_budget_s, self.wall_timeout_s = cpu_budget_s, wall_timeout_s
        self._lock = threading.RLock()
        self._entries = {}
        self._attempted = set()
        self._next = 0
        self._closed = False
        self._events = []
        self.counts = dict(requested=0, headers=0, early=0, fallback=0)

    def _event(self, kind, **values):
        # Bounded descriptor telemetry, emitted by the capture service thread.
        self._events.append(dict(event=kind, ts=time.time(), **values))
        self._events = self._events[-32:]

    def drain_events(self):
        with self._lock:
            events, self._events = self._events, []
            return events

    def _geometry(self, view, ref):
        detection = view.detections[ref.index]
        if detection.get('patience_trial'):
            return None
        sf, bw = int(detection['sf']), float(detection['bw'])
        if bw in (41667., 41670.):
            bw = 125000. / 3
        if not 7 <= sf <= 12 or not math.isfinite(bw) or bw <= 0:
            return None
        hint = float(detection.get('preamble_t_s', float('nan')))
        if not math.isfinite(hint) or hint < 0:
            return None
        symbol = (2 ** sf) / bw
        origin = view.metadata.get('base_start', view.start)
        preamble = origin + view.metadata['pre_hop_n'] + hint * self.owner.rate
        # Per-hint origin prevents an earlier complete packet in the same
        # 32s span taking the only opportunity for a later frame. Eight symbols
        # of leading context preserve the preamble through ordinary gate jitter.
        start = max(ref.band.origin + ref.start * ref.band.factor,
                    math.floor(preamble - 8 * symbol * self.owner.rate))
        start = ref.band.origin + math.ceil((start-ref.band.origin) / ref.band.factor) * ref.band.factor
        end = math.ceil(preamble + (32 + 4.25 + 8 + 4) * symbol * self.owner.rate)
        if end >= view.end or end <= start:
            return None  # ordinary full capture is already the shorter path
        count = math.ceil((end-start) / view.target_factor)
        if count * 8 > self.max_prefix_bytes:
            return None
        factor = view.target_factor // ref.band.factor
        context_end = math.ceil((end-ref.band.origin) / ref.band.factor)
        context_end += 10 * factor if factor > 1 else 0
        if ref.band.produced < context_end:
            return None  # consumed raw samples do not imply emitted FIR samples
        return start, end, count, sf, bw

    def step(self):
        try:
            self._step()
        except Exception as error:
            # Checkpoint/control failure cannot cancel independently retained
            # full captures. Stop optional work and retain explicit diagnostics.
            with self._lock:
                self._event('header-disabled', reason=str(error)[:200])
            self.close(wait=False)

    def _step(self):
        with self._lock:
            if self._closed:
                return
            if self.owner.failure or self.owner._closing:
                self.close(wait=False)
                return
            # At most the configured outstanding descriptor count per turn.
            for _ in range(self.max_pending):
                result = self.service.poll()
                if result is None:
                    break
                entry = self._entries.get(result.request.request_id)
                if entry is None:
                    continue
                self._event('header-result', request=entry.identifier, status=result.status,
                            cpu_s=getattr(result, 'cpu_seconds', None),
                            wall_s=getattr(result, 'wall_seconds', None))
                try:
                    good = (entry.state == 'requested' and result.status == 'ok'
                            and result.request == entry.request
                            and result.metadata is not None
                            and result.metadata.get('identity_token') == entry.token
                            and self.owner.header_source_valid(entry.hold, entry.token)
                            and self.owner.header_token(entry.parent, entry.ref,
                                  entry.prefix.start, entry.prefix.end) == entry.token)
                    if good:
                        self.counts['headers'] += 1
                        meta = result.metadata
                        canonical_bw = float(meta['bw'])
                        rate = entry.parent.nb_fs
                        relative = math.ceil(float(meta['ready_end_bw']) * rate / canonical_bw)
                        end = entry.prefix.start + relative * entry.parent.target_factor
                        preamble = entry.prefix.start + (float(meta['preamble_start_bw'])
                                                        * self.owner.rate / canonical_bw)
                        # Never claim completeness by clipping a larger required
                        # endpoint to the maximum span. Such a header abstains.
                        if (relative > 0 and math.isfinite(preamble)
                                and entry.prefix.start < end <= entry.parent.end):
                            child = self.owner.add_slice_view(entry.parent, entry.ref,
                                entry.prefix.start, end, role='early', request_id=entry.identifier)
                            child.metadata['header_preamble_raw'] = preamble
                            self.counts['early'] += 1
                            self._event('early-planned', request=entry.identifier,
                                        view=child.number, start=child.start, end=child.end)
                        else:
                            self.counts['fallback'] += 1
                    else:
                        self.counts['fallback'] += 1
                except Exception as error:
                    # An optional header/sibling allocation cannot cancel a
                    # previously admitted useful full span.
                    self.counts['fallback'] += 1
                    self._event('header-fallback', request=entry.identifier, reason=str(error)[:200])
                finally:
                    entry.hold.release()
                    self._entries.pop(entry.identifier, None)
            if len(self._entries) >= self.max_pending:
                return
            candidates = self.owner.header_candidates()
            live = {(view.number, ref.index) for view, ref in candidates}
            self._attempted.intersection_update(live | {
                (e.parent.number, e.ref.index) for e in self._entries.values()})
            for view, ref in candidates:
                key = view.number, ref.index
                if key in self._attempted:
                    continue
                geometry = self._geometry(view, ref)
                if geometry is None:
                    continue
                start, end, count, _, _ = geometry
                lease = hold = None
                try:
                    lease = _PrefixLease(self.owner.budget.reserve('header_prefix_file', count * 8))
                    hold = self.owner.hold_header_reference(view, ref)
                    identifier = self._next
                    self._next += 1
                    prefix = self.owner.add_slice_view(view, ref, start, end,
                        role='header_prefix', request_id=identifier)
                    token = self.owner.header_token(view, ref, prefix.start, prefix.end)
                    self._entries[identifier] = _Entry(identifier, view, ref, hold, token, prefix, lease)
                    self._attempted.add(key)
                    self._event('prefix-planned', request=identifier, view=view.number,
                                start=prefix.start, end=prefix.end, bytes=count*8)
                except Exception as error:
                    if hold is not None:
                        hold.release()
                    if lease is not None:
                        lease.release()
                    self._event('header-capacity-fallback', view=view.number, reason=str(error)[:200])
                break  # one optional preparation per service turn, no queue burst

    def write_prefix(self, view):
        """Called only by the existing save worker; never by acquisition/crop."""
        identifier = view.metadata['header_request_id']
        with self._lock:
            entry = self._entries.get(identifier)
            if self._closed or entry is None:
                return
            entry.state = 'writing'
        try:
            prefix = view.stream(0)
            if prefix.nbytes > entry.lease.credit.size:
                raise RuntimeError('prefix allocation exceeds its reservation')
            fd, path = tempfile.mkstemp(prefix='.header-', suffix='.prefix', dir=self.directory)
            os.close(fd)
            entry.lease.path = path
            prefix.tofile(path)
            # Drop the render buffer before handing the descriptor to a second
            # process. The existing writer workspace remains charged separately.
            count = len(prefix)
            del prefix
            detection = entry.parent.detections[entry.ref.index]
            request = HeaderRequest(identifier, path, 0, count, file_identity(os.stat(path)),
                int(detection['sf']), float(detection['bw']), view.nb_fs,
                float(entry.ref.center), entry.token, self.cpu_budget_s,
                self.wall_timeout_s, f'header-view-{entry.parent.number}-{entry.ref.index}')
            with self._lock:
                if self._closed or not self.owner.header_source_valid(entry.hold, entry.token):
                    raise RuntimeError('prefix identity ended before checkpoint')
                if not self.service.submit(request, entry.lease):
                    raise RuntimeError('header service has no descriptor capacity')
                entry.request, entry.state = request, 'requested'
                self.counts['requested'] += 1
                self._event('header-requested', request=identifier, bytes=count*8)
        except Exception as error:
            with self._lock:
                entry.lease.release()
                entry.hold.release()
                self._entries.pop(identifier, None)
                self.counts['fallback'] += 1
                self._event('header-write-fallback', request=identifier, reason=str(error)[:200])

    def close(self, *, wait=True):
        with self._lock:
            self._closed = True
            entries = list(self._entries.values())
            for entry in entries:
                if entry.state == 'requested':
                    self.service.cancel(entry.identifier)
        # Live service turns only signal cancellation. An optional worker's
        # termination must never delay the sole shared-crop producer. Explicit
        # EOF/shutdown may wait, including after earlier live initiation.
        stopped = self.service.close(timeout_s=2 if wait else 0)
        with self._lock:
            for entry in entries:
                if entry.state == 'writing':
                    # The writer still reads parent metadata and owns its
                    # prefix lease; its completion path releases both.
                    continue
                entry.hold.release()
                if entry.state != 'requested':
                    entry.lease.release()
                # Requested prefixes belong exclusively to the service until
                # the worker stops. It retains/releases that lease itself;
                # retaining a controller entry would only strand parent data.
                self._entries.pop(entry.identifier, None)
            if wait and not stopped:
                self._event('header-close-failed', reason=str(self.service.failure))
        return stopped  # worker termination confirmation, not capture draining
