"""Bounded, metadata-only header checkpoints; deliberately not wired to RX.

A background writer prepares an immutable cf32 prefix and reserves its backing
storage. submit() transfers that lease only on True. A separate lifetime lease
covers the worker's maximum FFT workspace, including allocator retention. This
module never materializes a prefix in the producer and never closes a capture
or admits a payload. The controller must validate generation/drop/grid tokens
before using readiness; a failed checkpoint always leaves the full fallback.
"""
from collections import deque
from dataclasses import dataclass
import math
import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time


@dataclass(frozen=True)
class HeaderRequest:
    request_id: int
    path: str
    offset_samples: int
    sample_count: int
    file_identity: tuple
    sf: int
    bw: float
    nb_fs: float
    mix_center_hz: float
    identity_token: tuple
    cpu_budget_s: float = 2.0
    wall_timeout_s: float = 10.0
    name: str = ''


@dataclass(frozen=True)
class HeaderResult:
    request: HeaderRequest
    status: str
    metadata: object = None
    error: str = None
    cpu_seconds: float = None
    wall_seconds: float = None


def file_identity(stat):
    """Prepare outside acquisition; prefix must remain immutable while leased."""
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _probe_request(request):
    # Import/worker startup happens before the job CPU duration begins. FFT and
    # prefix setup are charged to this job; the independent wall guard covers
    # imports, mapping stalls and any native function that ignores CPU limits.
    import numpy as np
    from decoder import probe_frame_header
    started = time.process_time()
    try:
        with open(request.path, 'rb', buffering=0) as stream:
            before = file_identity(os.fstat(stream.fileno()))
            if (before != request.file_identity
                    or (request.offset_samples + request.sample_count) * 8 > before[2]):
                return 'stale', None, 'prefix identity or extent changed'
            prefix = np.memmap(stream, dtype=np.complex64, mode='r',
                               offset=request.offset_samples * 8,
                               shape=(request.sample_count,))
            try:
                remaining = request.cpu_budget_s - (time.process_time() - started)
                metadata = (probe_frame_header(
                    prefix, request.sf, request.bw, request.nb_fs, request.name,
                    budget_s=remaining, identity_token=request.identity_token)
                    if remaining > 0 else None)
                if file_identity(os.fstat(stream.fileno())) != before:
                    return 'stale', None, 'prefix changed during checkpoint'
                if time.process_time() - started > request.cpu_budget_s:
                    return 'abstained', None, 'CPU duration expired'
                return ('ok', metadata, None) if metadata is not None else ('abstained', None, None)
            finally:
                prefix._mmap.close()
    except Exception as error:
        return 'error', None, str(error)[:500]


def _worker_main(connection):
    # One dedicated process: diagnostics cannot enter the parent's stdout or
    # share its result protocol. No thread-global stdout redirection is used.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import atexit
    import decoder
    if hasattr(decoder, 'pyfftw'):
        decoder.pyfftw.interfaces.cache.disable()
    if hasattr(decoder, '_save_wisdom'):
        atexit.unregister(decoder._save_wisdom)
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            cpu_start = time.process_time()
            status, metadata, error = _probe_request(request)
            connection.send((request.request_id, status, metadata, error,
                             time.process_time() - cpu_start))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def _worker_command(fd):
    # A standalone entrypoint avoids multiprocessing spawn re-importing the
    # caller's __main__ (which may load NumPy before our thread settings).
    return [sys.executable, os.path.abspath(__file__), str(fd)]


def _immutable_token(value):
    # Bound IPC metadata independently of prefix bytes. No arbitrary objects or
    # mutable containers whose identity can change after producer submission.
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 128 or depth > 8:
            return False
        if isinstance(item, tuple):
            if len(item) > 32:
                return False
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            text_bytes += len(item)
            if len(item) > 256 or text_bytes > 4096:
                return False
        elif isinstance(item, int):
            if item.bit_length() > 128:
                return False
        elif item is not None and not (isinstance(item, float) and math.isfinite(item)):
            return False
    return True


class HeaderCheckpointService:
    """One persistent process with bounded queued + active + uncollected work.

    submit/cancel/poll only touch small descriptors under a short lock. The
    manager owns all process I/O and waits. An accepted lease's release() runs
    exactly once after the worker stops touching that prefix. Call close() at
    EOF to cancel checkpoints; it never waits for a decoder CPU deadline.
    Request leases cover prefix backing; the required workspace_lease covers
    one maximum-sized worker for the service lifetime. Both provide nonblocking,
    idempotent release(). Refused requests remain the caller's responsibility.
    """
    def __init__(self, *, workspace_lease, max_pending=2, max_prefix_bytes=64 * 1024 * 1024,
                 max_cpu_s=5.0, max_wall_s=30.0):
        if (max_pending < 1 or max_prefix_bytes < 8
                or not math.isfinite(max_cpu_s) or max_cpu_s <= 0
                or not math.isfinite(max_wall_s) or max_wall_s <= 0
                or not callable(getattr(workspace_lease, 'release', None))):
            raise ValueError('invalid checkpoint capacity')
        self.max_pending = int(max_pending)
        self.max_prefix_bytes = int(max_prefix_bytes)
        self.max_cpu_s, self.max_wall_s = max_cpu_s, max_wall_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._states = {}
        self._queue = deque()
        self._results = deque()
        self._closing = False
        self._active = None
        self._process = self._connection = None
        self._unreaped = None
        self._failure = None
        self._workspace_lease = workspace_lease
        self._thread = threading.Thread(target=self._manage, daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._workspace_lease = None
            workspace_lease.release()
            raise

    def _valid(self, request):
        try:
            return (isinstance(request, HeaderRequest)
                    and isinstance(request.request_id, int)
                    and request.request_id.bit_length() <= 128
                    and isinstance(request.path, str) and 0 < len(request.path) <= 4096
                    and isinstance(request.name, str) and len(request.name) <= 512
                    and isinstance(request.offset_samples, int) and request.offset_samples >= 0
                    and isinstance(request.sample_count, int) and request.sample_count > 0
                    and request.sample_count * 8 <= self.max_prefix_bytes
                    and isinstance(request.file_identity, tuple) and len(request.file_identity) == 4
                    and all(isinstance(v, int) and v >= 0 for v in request.file_identity)
                    and (request.offset_samples + request.sample_count) * 8 <= request.file_identity[2]
                    and int(request.sf) == request.sf and 7 <= request.sf <= 12
                    and all(math.isfinite(v) and v > 0 for v in
                            (request.bw, request.nb_fs, request.mix_center_hz))
                    # Bound FFT padding even for an almost-empty prefix with
                    # an adversarial clock ratio. Covers normal NB grids and
                    # up to 32Msps/31.25k wideband input; no arbitrary resampler.
                    and 0 < request.nb_fs / request.bw <= 1024
                    and 0 < request.cpu_budget_s <= self.max_cpu_s
                    and 0 < request.wall_timeout_s <= self.max_wall_s
                    and isinstance(request.identity_token, tuple)
                    and _immutable_token(request.identity_token)
                    and len(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)) <= 4096)
        except (TypeError, ValueError, OverflowError):
            return False

    def submit(self, request, lease):
        if not self._valid(request) or not callable(getattr(lease, 'release', None)):
            return False
        with self._lock:
            if (self._closing or len(self._states) >= self.max_pending
                    or request.request_id in self._states):
                return False
            self._states[request.request_id] = [request, lease, False, False]
            self._queue.append(request.request_id)
        self._wake.set()
        return True

    @property
    def pending(self):
        with self._lock:
            return len(self._states)

    @property
    def active_request_id(self):
        with self._lock:
            return self._active

    def cancel(self, request_id):
        with self._lock:
            state = self._states.get(request_id)
            if state is None or state[3]:
                return False
            state[2] = True
        self._wake.set()
        return True

    def poll(self):
        with self._lock:
            if not self._results:
                return None
            result = self._results.popleft()
            self._states.pop(result.request.request_id, None)
            return result

    def close(self, *, timeout_s=1.0):
        with self._lock:
            self._closing = True
            for state in self._states.values():
                if not state[3]:
                    state[2] = True
        self._wake.set()
        self._thread.join(timeout=max(0., timeout_s))
        return (not self._thread.is_alive() and self._unreaped is None
                and self._failure is None)

    @property
    def failure(self):
        return self._failure

    def _stop_worker(self):
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            connection.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                # Do not release memory still mapped by an unkillable process.
                self._unreaped = process
                self._failure = 'header worker could not be reaped'
                return False
        return True

    def _start_worker(self):
        if os.name != 'posix':
            raise OSError('header checkpoints require POSIX file-descriptor handoff')
        parent, child = multiprocessing.Pipe()
        environment = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            environment[key] = '1'
        try:
            process = subprocess.Popen(_worker_command(child.fileno()),
                pass_fds=(child.fileno(),), env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._process, self._connection = process, parent

    def _finish(self, state, status, metadata=None, error=None, *,
                cpu_seconds=None, wall_seconds=None):
        request, lease, _, _ = state
        try:
            lease.release()
        except Exception as exc:
            status, metadata, error = 'error', None, 'lease release failed: ' + str(exc)[:400]
            self._failure = error
        with self._lock:
            if state[2]:
                status, metadata = 'cancelled', None
            state[1], state[3] = None, True
            self._active = None
            self._results.append(HeaderResult(request, status, metadata, error,
                                              cpu_seconds, wall_seconds))

    def _manage(self):
        try:
            while True:
                with self._lock:
                    state = self._states[self._queue.popleft()] if self._queue else None
                    closing = self._closing
                    if state is not None:
                        self._active = state[0].request_id
                if state is None:
                    if closing:
                        return
                    self._wake.wait(.05)
                    self._wake.clear()
                    continue
                if state[2]:
                    self._finish(state, 'cancelled')
                    continue
                request = state[0]
                wall_start = time.monotonic()
                deadline = wall_start + request.wall_timeout_s
                try:
                    if self._process is None:
                        self._start_worker()
                    self._connection.send(request)
                    while True:
                        if state[2] or time.monotonic() >= deadline:
                            status = 'cancelled' if state[2] else 'timeout'
                            if not self._stop_worker():
                                return
                            self._finish(state, status, wall_seconds=time.monotonic() - wall_start)
                            break
                        if self._connection.poll(.01):
                            identifier, status, metadata, error, cpu_seconds = self._connection.recv()
                            if identifier != request.request_id:
                                raise RuntimeError('worker response identity mismatch')
                            self._finish(state, status, metadata, error,
                                         cpu_seconds=cpu_seconds,
                                         wall_seconds=time.monotonic() - wall_start)
                            break
                        if self._process.poll() is not None:
                            raise RuntimeError('header worker exited before result')
                except Exception as error:
                    if not self._stop_worker():
                        return
                    self._finish(state, 'error', error=str(error)[:500],
                                 wall_seconds=time.monotonic() - wall_start)
        finally:
            if self._stop_worker() and self._unreaped is None:
                lease, self._workspace_lease = self._workspace_lease, None
                try:
                    lease.release()
                except Exception as error:
                    self._failure = 'workspace release failed: ' + str(error)[:400]


if __name__ == '__main__':
    from multiprocessing.connection import Connection
    _worker_main(Connection(int(sys.argv[1])))
