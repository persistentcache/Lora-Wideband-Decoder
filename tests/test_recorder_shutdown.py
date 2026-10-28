"""EOF must submit the final deferred capture before decoder drain is tested."""
import ast
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import detector
from detector import SignalRecorder


def production_footer():
    tree = ast.parse(Path(detector.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    # Preserve actual source order, exercising the calls rather than asserting
    # the textual order. Queue joins and finalization below are real methods.
    body = [n for n in main.body if isinstance(n, ast.If) and any(
        token in ast.unparse(n) for token in (
            'recorder.flush()', 'recorder.flush_pending(wait=True)',
            'decoder.drain(timeout=3600.0)'))]
    assert len(body) in (2, 3)
    return compile(ast.Module(body=body, type_ignores=[]), detector.__file__, 'exec')


@pytest.mark.parametrize('narrowband', [True, False])
def test_final_pending_capture_is_saved_and_submitted_before_decoder_drain(narrowband):
    # Delay the saver until shutdown actually joins its outstanding work.
    # This is a valid scheduling interleaving, with no sleeps or CPU races.
    permit_saver = threading.Event()

    class SaveQueue(queue.Queue):
        def join(self):
            if self.unfinished_tasks:
                permit_saver.set()
            return super().join()

    recorder = SignalRecorder.__new__(SignalRecorder)
    recorder._save_queue = SaveQueue(maxsize=8)
    recorder._crop_queue = queue.Queue(maxsize=48) if narrowband else None
    job = {'base_parts': [np.ones(16, np.complex64)], 'tail': [], 'need': 0,
           'dets': [{'sf': 12, 'bw': 62500, 'freq_hz': 915e6}],
           'pre_hop_n': 0, 'wb_n': 160, 'sample_pos': 160}
    if narrowband:
        job['nb'] = {
            'decs': {0: SimpleNamespace(finish=lambda: np.empty(0, np.complex64))},
            'parts': {0: [np.ones(8, np.complex64)]},
            'nb_fs': 500000.0, 'export_bw': 250000}
    recorder._pending = [job]
    submitted, calls = [], []

    def save():
        permit_saver.wait()
        item = recorder._save_queue.get()
        submitted.append(item)
        recorder._save_queue.task_done()

    def drain(**kwargs):
        calls.append(('drain', len(submitted), recorder._save_queue.unfinished_tasks))

    decoder = SimpleNamespace(drain_slow_into_fast=lambda: calls.append(('fold',)),
                              pending=lambda: len(submitted), drain=drain)
    saver = threading.Thread(target=save)
    cropper = threading.Thread(target=recorder._crop_worker) if narrowband else None
    saver.start()
    if cropper:
        cropper.start()
    try:
        exec(production_footer(), dict(recorder=recorder, decoder=decoder, sys=sys))
        assert calls == [('fold',), ('drain', 1, 0)]
        assert len(submitted) == 1 and recorder._pending == []
        assert recorder._save_queue.unfinished_tasks == 0
    finally:
        permit_saver.set()
        saver.join(timeout=2)
        if cropper:
            recorder._crop_queue.put(None)
            cropper.join(timeout=2)
            assert not cropper.is_alive()
        assert not saver.is_alive()
