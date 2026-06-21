"""Save subprocess for the wideband decoder pipeline.

Runs the per-batch FFT-extract + capture-file-write work in a child process
so it stops eating master-process GIL time.  Background:

The master's recorder threads (4 of them) were measured holding ~60% of the
master GIL during burst load (live py-spy snapshots, see commit 9af8416).
iq-reader was being shut out — 0 of 20 active-GIL snapshots — and that
starvation back-pressured the kernel pipe to soapy_rx, overflowing the
HackRF's internal buffer and dropping samples.  v2.1.0 pinned iq-reader to
its own core + tightened the GIL handoff to claw back some of that, but
the structural fix is to move the recorder work out of the master process
entirely — that's what this file does.

IPC contract with master (set up in SignalRecorder):
  - shm pool: N pre-allocated mp.shared_memory blocks for IQ transfer.
    Master writes extended_full into a free block, sends the block index
    + length + metadata via save_q.  Subproc reads the IQ from shm,
    processes, writes capture files to /dev/shm/live_caps/, sends results
    via submit_q, then returns the block index to free_q.
  - save_q (mp.Queue, master→subproc): batched save events
  - submit_q (mp.Queue, subproc→master): decode-submit requests (file paths)
  - free_q (mp.Queue, subproc→master): freed shm block indices

The actual _save_worker logic stays in detector.py for Phase 1 — this file
provides the IPC scaffolding and a stub worker that just verifies the
channels work end-to-end without behaviour change.  Phase 2 ports the real
worker logic in here.
"""
import os
import sys
import time
import numpy as np
from multiprocessing import shared_memory


def save_worker_main(save_q, submit_q, free_q, shm_names, shm_bytes, debug=0):
    """Entry point for the save subprocess.

    Args mirror what the master set up in SignalRecorder:
      save_q     — inbound save events (block_idx, n_samples, metadata...)
      submit_q   — outbound decode-submit requests (fpath, fname, bucket...)
      free_q     — outbound shm block-index returns (just the int index)
      shm_names  — list of mp.shared_memory block names (we attach by name)
      shm_bytes  — size in bytes of each block (uniform)
      debug      — debug verbosity level

    Loop:
      1. event = save_q.get()
      2. attach to shm_blocks[event['block_idx']]
      3. read extended_full from shm as np.complex64
      4. process (Phase 1: just a sanity ack; Phase 2: real save_worker logic)
      5. submit_q.put({'fpath': ..., ...}) for each file the subproc wrote
      6. free_q.put(event['block_idx']) — release the shm block
      7. None on save_q = shutdown sentinel
    """
    # Pin this subprocess off the gate cores (0-3) and off iq-reader's
    # core 0 specifically.  Use cores 1-3 — the same range master's
    # MainThread / recorders / managers contend on, but this subprocess
    # has its OWN process-level GIL so it doesn't compete with master's
    # GIL.  That's the whole point of moving this out.
    try:
        os.sched_setaffinity(0, {1, 2, 3})
    except (AttributeError, OSError):
        pass

    # Attach to each shm block once.  Lifetime of these attachments lasts
    # the lifetime of this subprocess; master cleans up the blocks on exit.
    shm_blocks = []
    for name in shm_names:
        try:
            shm_blocks.append(shared_memory.SharedMemory(name=name))
        except FileNotFoundError as e:
            print(f"[save_subproc] shm attach failed for {name}: {e}",
                  file=sys.stderr, flush=True)
            return

    if debug >= 1:
        print(f"[save_subproc] up, pid={os.getpid()}, "
              f"{len(shm_blocks)} shm blocks × {shm_bytes/1024/1024:.0f}MB",
              file=sys.stderr, flush=True)

    while True:
        event = save_q.get()
        if event is None:
            # Shutdown.
            if debug >= 1:
                print(f"[save_subproc] shutdown sentinel received",
                      file=sys.stderr, flush=True)
            break
        block_idx = event['block_idx']
        n_samples = event['n_samples']
        try:
            shm = shm_blocks[block_idx]
            # View into the shm: complex64 (8 bytes/sample), n_samples long.
            iq = np.ndarray(
                (n_samples,), dtype=np.complex64,
                buffer=shm.buf[:n_samples * 8])

            # PHASE 1 STUB: just ack receipt to prove the IPC path works.
            # Phase 2 will replace this block with the actual save_worker
            # body — FFT extraction per preamble, file write, etc.
            if debug >= 2:
                print(f"[save_subproc] event block={block_idx} "
                      f"n={n_samples} sf={event.get('sf')} "
                      f"detections={len(event.get('detections', []))}",
                      file=sys.stderr, flush=True)
            # Don't submit anything yet — master keeps its old in-process
            # path for actual saves while we exercise the IPC plumbing.
        except Exception as e:
            print(f"[save_subproc] error processing event: {e}",
                  file=sys.stderr, flush=True)
        finally:
            # ALWAYS release the shm block, even on error, or master
            # blocks indefinitely on free_q.get().
            try:
                free_q.put(block_idx)
            except Exception:
                pass

    # Cleanup on shutdown: close (not unlink) our attachments.  Master
    # owns the blocks (created them) and unlinks them.
    for shm in shm_blocks:
        try: shm.close()
        except Exception: pass
