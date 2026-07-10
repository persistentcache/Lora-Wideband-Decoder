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
import queue
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


def _worker_main(shm_names, win_n, task_q, result_q, params):
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
        task = task_q.get()
        if task is None:
            break
        slot, seq, n, psd, peaks, spur_db, task_center, task_chans = task
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
            dets = L.detect_preamble(iq, wb_fs, wb_bw, _c, **kw)
        except Exception as e:
            dets = []
            result_q.put((seq, dets, 'ERR:%s' % e))
            continue
        result_q.put((seq, dets, None))

    for s in shms:
        s.close()


class DetectPool:
    def __init__(self, n_workers, n_slots, win_n, params):
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
        self._workers = []
        for _ in range(n_workers):
            p = ctx.Process(target=_worker_main,
                            args=(names, win_n, self._task_q,
                                  self._result_q, params),
                            daemon=True)
            p.start()
            self._workers.append(p)
        self._results = {}   # seq -> dets, filled as results arrive

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
        self._task_q.put((slot, seq, n, psd, peaks, spur_db, center,
                          dechirp_chans))

    def _absorb(self, rseq, dets, err):
        if err is not None:
            # Worker-side detect exceptions used to vanish silently (the
            # third tuple field was read and dropped): a systematic failure
            # looked identical to "no signal" (UC audit b).
            import sys
            print(f"[POOL] worker error on window {rseq}: {err}",
                  file=sys.stderr, flush=True)
        self._results[rseq] = dets

    def result(self, seq):
        """Block until window `seq`'s detection is available; return dets.

        Bounded: if no result arrives for 10 s and a worker has died, the
        window is abandoned as no-detections — a dead worker's tasks never
        produce a result, and the unbounded get() froze the whole gate loop
        forever (UC audit b).  Losing one window beats losing the pipeline.
        """
        while seq not in self._results:
            try:
                rseq, dets, err = self._result_q.get(timeout=10.0)
            except queue.Empty:
                if all(p.is_alive() for p in self._workers):
                    continue   # slow window (big SF sweep) — keep waiting
                import sys
                print(f"[POOL] worker died; abandoning window {seq} "
                      f"(no detections)", file=sys.stderr, flush=True)
                self._results[seq] = []
                break
            self._absorb(rseq, dets, err)
        return self._results.pop(seq)

    def ready(self, seq):
        """Non-blocking: drain any arrived results, return True if `seq` is done.
        Lets the gate make a tail-aware commit decision without blocking the
        realtime read on a not-yet-finished detection."""
        try:
            while True:
                rseq, dets, err = self._result_q.get_nowait()
                self._absorb(rseq, dets, err)
        except queue.Empty:
            pass
        return seq in self._results

    def peek(self, seq):
        """Return `seq`'s dets without removing them (call after ready())."""
        return self._results.get(seq)

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
