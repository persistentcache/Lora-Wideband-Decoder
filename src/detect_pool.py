"""Multiprocess Schmidl-Cox detection pool.

The gate's per-window detect_preamble (1Msps extract + multi-lag SC + dechirp
confirm) is the only thing that can't sustain 28 Msps in one process — the
producer (read + Welch + slide) runs at ~57 Msps with headroom, but detect is
~0.8x real-time serial.  Detection of each time-window is independent, so this
pool fans windows out to N single-threaded worker PROCESSES (true parallelism,
no GIL), fed via shared-memory slots so the 224 MB window is never pickled.

Results are identical to calling detect_preamble directly — same algorithm, run
on different cores.  The caller collects in window order.

Usage:
    pool = DetectPool(n_workers=6, n_slots=10, win_n=28_000_000, params={...})
    slot = pool.acquire_slot()              # blocks until a slot is free
    pool.slot_array(slot)[:n] = window_iq   # write the window into the slot
    pool.dispatch(slot, seq, n, psd, peaks) # hand to a worker
    ...
    dets = pool.result(seq)                 # blocks until window `seq` is done
    pool.release_slot(slot)                 # return slot to the free pool
    pool.close()
"""
import os
import time
import queue
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


def _worker_main(shm_names, win_n, task_q, result_q, params, cap_override=None):
    # Single-threaded inside each worker so N workers ≈ N cores (no
    # oversubscription).  detect_preamble reads these envs.
    os.environ['LORA_FFT_WORKERS'] = '1'
    os.environ['LORA_DETECT_SERIAL'] = '1'
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import detector as L

    # Attach to every shared slot once; build a numpy view per slot.
    shms = [shared_memory.SharedMemory(name=nm) for nm in shm_names]
    views = [np.ndarray((win_n,), dtype=np.complex64, buffer=s.buf) for s in shms]

    wb_fs = params['wb_fs']; wb_bw = params['wb_bw']; center = params['center']
    sc_thr = params['sc_threshold']; ethr = params['ethresh']
    dc_notch = params['dc_notch']; spur_notch = params['spur_notch']
    spur_default = params.get('spur_db_default', None)
    dechirp_chans = params.get('dechirp_chans', None)   # channelizer dechirp matched-filter
    # Honor LORA_DEBUG so users diagnosing detection issues actually see the
    # per-peak energy / SC / dechirp output the gate already knows how to print.
    # The hardcoded debug=0 here previously made --debug N on the parent silently
    # a no-op in multi-worker mode (the default).
    try:
        _dbg = int(os.environ.get('LORA_DEBUG', '0') or '0')
    except ValueError:
        _dbg = 0

    while True:
        try:
            task = task_q.get()
        except KeyboardInterrupt:
            # SIGINT at pipeline shutdown reaches the fork workers too —
            # exit quietly instead of spraying a traceback per worker.
            break
        if task is None:
            break
        slot, seq, n, psd, peaks, spur_db, task_center, task_chans = task
        # Emergency cap from the gate, applied at TASK START so it reaches
        # windows that were already queued: under a sustained deficit the
        # table fills with 10-24-peak windows (seconds each) dispatched
        # before the budget clamp engaged, and dispatch then blocks the
        # main thread until one drains.  Truncating here (strongest peaks
        # first — the same shed-weakest rule the gate applies) or skipping
        # (-1) lets that backlog drain in milliseconds instead.
        if cap_override is not None and peaks:
            _co = cap_override.value
            if _co < 0:
                peaks = []
            elif 0 < _co < len(peaks):
                peaks = sorted(peaks, key=lambda p: -p[2])[:_co]
        iq = views[slot][:n]
        try:
            kw = dict(sc_threshold=sc_thr, ethresh=ethr,
                      dc_notch_mhz=dc_notch, spur_notch_hz=spur_notch,
                      debug=_dbg, cached_psd=psd, cached_peaks=peaks,
                      # Per-task channels (same pattern as task_center): the
                      # live ctl-file channel apply and the patience gate's
                      # rotating (sf,bw) trials change the fed set MID-RUN —
                      # spawn-time params are stale the moment either fires.
                      dechirp_chans=(task_chans if task_chans is not None
                                     else dechirp_chans))
            if spur_db is not None:
                kw['spur_db'] = spur_db
            _c = task_center if task_center is not None else center
            _t0 = time.perf_counter()
            dets = L.detect_preamble(iq, wb_fs, wb_bw, _c, **kw)
            # Measured detect cost rides along with the result: the gate's
            # proactive peak-budget clamp needs the WORKER's wall time per
            # peak, not the ~15 ms the main loop spends dispatching (which it
            # used to read as the per-peak cost in pooled mode — the cap
            # then climbed to its maximum and 20-peak windows took seconds
            # each here, wrapping the ring).
            _cost = (time.perf_counter() - _t0, len(peaks) if peaks else 0)
        except KeyboardInterrupt:
            break   # shutdown SIGINT mid-detect — exit quietly
        except Exception as e:
            dets = []
            result_q.put((seq, dets, 'ERR:%s' % e, None))
            continue
        result_q.put((seq, dets, None, _cost))

    for s in shms:
        s.close()


class DetectPool:
    def __init__(self, n_workers, n_slots, win_n, params):
        self._aborted = False
        self.win_n = win_n
        self.n_slots = n_slots
        # Shared-memory slots, each holds one full window (complex64).
        self._shms = [shared_memory.SharedMemory(create=True,
                                                 size=win_n * 8)
                      for _ in range(n_slots)]
        self._views = [np.ndarray((win_n,), dtype=np.complex64, buffer=s.buf)
                       for s in self._shms]
        self._free = list(range(n_slots))
        names = [s.name for s in self._shms]

        # 'fork' so workers inherit the already-imported numpy/scipy/detector
        # (fast startup).  CRITICAL: the caller must build this pool BEFORE
        # starting any other threads (recorder save-workers, decoder managers,
        # the stdin drainer) — forking a multithreaded process inherits their
        # held locks and the workers deadlock.  Built first, the process is
        # single-threaded and fork is safe.
        ctx = mp.get_context('fork')
        self._task_q = ctx.Queue()
        self._result_q = ctx.Queue()
        # Shared emergency cap read by workers at task start (0 = none,
        # N>0 = keep the N strongest peaks, -1 = skip detection).
        self._cap_override = ctx.Value('i', 0)
        self._workers = []
        for _ in range(n_workers):
            p = ctx.Process(target=_worker_main,
                            args=(names, win_n, self._task_q,
                                  self._result_q, params,
                                  self._cap_override),
                            daemon=True)
            p.start()
            self._workers.append(p)
        self._results = {}   # seq -> dets, filled as results arrive
        self._costs = []     # (detect_seconds, n_peaks) per finished window,
                             # drained by the gate's budget clamp

    def acquire_slot(self):
        # Single-threaded caller (the gate producer), so no lock needed.
        return self._free.pop() if self._free else None

    def slot_array(self, slot):
        return self._views[slot]

    def release_slot(self, slot):
        self._free.append(slot)

    def n_free(self):
        return len(self._free)

    def dispatch(self, slot, seq, n, psd, peaks, spur_db=None, center=None,
                 dechirp_chans=None):
        # center (MHz) and dechirp_chans are passed per-task so a live
        # center-frequency change / live channel apply / patience trial takes
        # effect without re-spawning workers; None → use the spawn-time params.
        if self._aborted:
            raise RuntimeError('detect pool has been aborted')
        self._task_q.put((slot, seq, n, psd, peaks, spur_db, center,
                          dechirp_chans))

    def _absorb(self, rseq, dets, err, cost=None):
        if err is not None:
            # Worker-side detect exceptions used to vanish silently (the
            # third tuple field was read and dropped): a systematic failure
            # looked identical to "no signal" (UC audit b).
            import sys
            print(f"[POOL] worker error on window {rseq}: {err}",
                  file=sys.stderr, flush=True)
        self._results[rseq] = dets
        if cost is not None:
            self._costs.append(cost)

    def set_cap_override(self, cap):
        """Emergency cap for queued windows: 0 none, N keep the N strongest
        peaks, -1 skip detection.  Takes effect at each worker's next task."""
        if self._cap_override.value != int(cap):
            self._cap_override.value = int(cap)

    def drain_costs(self):
        """Return and clear the (detect_seconds, n_peaks) samples reported by
        workers since the last drain — the clamp's per-peak cost feed."""
        c, self._costs = self._costs, []
        return c

    def result(self, seq):
        """Block until window `seq`'s detection is available; return dets.

        Bounded: if no result arrives for 10 s and a worker has died, the
        window is abandoned as no-detections — a dead worker's tasks never
        produce a result, and the unbounded get() froze the whole gate loop
        forever (UC audit b).  Losing one window beats losing the pipeline.
        """
        while seq not in self._results:
            _all_alive = all(p.is_alive() for p in self._workers)
            try:
                # Short wait once a worker is down: at SIGINT shutdown ALL
                # workers die and the EOF drain would otherwise pay the full
                # 10 s per still-in-flight window.
                _item = self._result_q.get(
                    timeout=10.0 if _all_alive else 0.5)
            except queue.Empty:
                if _all_alive:
                    continue   # slow window (big SF sweep) — keep waiting
                import sys
                print(f"[POOL] worker died; abandoning window {seq} "
                      f"(no detections)", file=sys.stderr, flush=True)
                self._results[seq] = []
                break
            self._absorb(*_item)
        return self._results.pop(seq)

    def ready(self, seq):
        """Non-blocking: drain any arrived results, return True if `seq` is done.
        Lets the gate make a tail-aware commit decision without blocking the
        realtime read on a not-yet-finished detection."""
        try:
            while True:
                self._absorb(*self._result_q.get_nowait())
        except queue.Empty:
            pass
        return seq in self._results

    def peek(self, seq):
        """Return `seq`'s dets without removing them (call after ready())."""
        return self._results.get(seq)

    def abort(self, timeout=2.0):
        """Failure-only discard of queued work; never wait on queue feeders."""
        self._aborted = True
        deadline = time.monotonic() + max(0.0, float(timeout))
        clean = True
        # Consumers are about to stop. Joining a task feeder whose pipe is full
        # would hang interpreter exit even after every worker has been reaped.
        for q in (self._task_q, self._result_q):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                clean = False
        for worker in self._workers:
            if worker.is_alive():
                try:
                    worker.terminate()
                except ProcessLookupError:
                    pass
                except Exception:
                    clean = False
        term_deadline = time.monotonic() + max(0.0, deadline-time.monotonic()) / 2
        for worker in self._workers:
            try:
                worker.join(timeout=max(0.0, term_deadline-time.monotonic()))
            except Exception:
                clean = False
        for worker in self._workers:
            if worker.is_alive():
                try:
                    worker.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    clean = False
        for worker in self._workers:
            try:
                worker.join(timeout=max(0.0, deadline-time.monotonic()))
                clean = clean and not worker.is_alive()
            except Exception:
                clean = False
        self._views.clear()
        for memory in self._shms:
            try:
                memory.close()
            except Exception:
                clean = False
            try:
                memory.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                clean = False
        return clean

    def close(self):
        for _ in self._workers:
            try:
                self._task_q.put(None)
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for s in self._shms:
            # unlink() must run even when close() raises, or a partial
            # teardown leaks the shm segments in /dev/shm until reboot
            # (UC audit b) — real RAM on a Pi, gone across restarts.
            try:
                s.close()
            except Exception:
                pass
            try:
                s.unlink()
            except Exception:
                pass
