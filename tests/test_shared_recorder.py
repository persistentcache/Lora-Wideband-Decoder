"""Explicit shared-capture flag routes actual recorder boundaries safely."""
import queue
import time

import numpy as np
import pytest

import detector
from test_capture_timeline import reader


@pytest.fixture
def rec(tmp_path,monkeypatch):
    monkeypatch.setenv('LORA_SHARED_CAPTURE','1')
    monkeypatch.setenv('LORA_SHARED_CAPTURE_MB','8')
    monkeypatch.setenv('LORA_EXPORT_DEC','2')
    monkeypatch.setenv('LORA_PAYLOAD_SYM_BUDGET','120')
    monkeypatch.setattr(detector.SignalRecorder,'_save_worker',lambda self:None)
    r=detector.SignalRecorder(str(tmp_path),1000,915e6,500)
    yield r
    if getattr(r,'_shared',None):
        r._shared.close(wait=False)
    if r._crop_queue is not None:
        r._crop_queue.put(None);r._crop_thread.join(2)


def det():
    return {'sf':2,'bw':62.5,'freq_hz':915e6,'peak_power_db':20,'detect_conf':.99}


def test_actual_recorder_routes_metadata_without_consuming_history_generator(rec):
    history=reader();history._rp=history._wp=4000
    rec.set_history_reader(history)
    def forbidden():
        pytest.fail('main thread must not enumerate seed history')
        yield
    rec.update_deferred([det()],np.zeros(4000,np.complex64),sample_pos=4000,
                        need_tail_n=2000,owned_buf=True,tail_parts=forbidden(),
                        history_anchor=history.history_anchor(4000))
    assert not rec._pending and rec._crop_queue.empty()
    history._rp=history._wp=7000
    rec.feed_tail(np.zeros(500,np.complex64),sample_pos=7000)
    item=rec._save_queue.get(timeout=2)
    assert item[0][0]=='NB_SHARED'
    shared=item[0][1]
    assert shared.complete and len(shared.stream(0))==750
    shared.release();rec._save_queue.task_done()


def test_actual_recorder_exposes_failed_capture_before_healthy_loop_can_continue(rec):
    history=reader();history._rp=history._wp=4000
    rec.set_history_reader(history)
    rec.update_deferred([det()],np.zeros(4000,np.complex64),sample_pos=4000,
                        need_tail_n=2000,history_anchor=history.history_anchor(4000))
    history._history_generation+=1
    with pytest.raises(RuntimeError,match='skipped|retuned'):
        rec.feed_tail(np.zeros(500,np.complex64),sample_pos=4000)


def test_shared_flag_is_explicit_and_does_not_change_default_recorder(rec,monkeypatch):
    monkeypatch.delenv('LORA_SHARED_CAPTURE')
    rec.set_history_reader(reader())
    assert rec._shared is None


def test_only_shared_recorder_selects_tiled_capture_fir(rec):
    rec.set_history_reader(reader())
    assert rec._shared.factory(35000, 10e6, 5)._use_gemm
    assert not detector._StreamDecimator(35000, 10e6, 5)._use_gemm


def test_actual_save_worker_handles_shared_view_and_releases_after_write(rec,monkeypatch):
    # Constructor save threads are disabled; execute the real worker below.
    import ast
    import threading
    from pathlib import Path
    tree=ast.parse(Path(detector.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SignalRecorder')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_save_worker')
    scope=dict(vars(detector));exec(compile(ast.Module([node],type_ignores=[]),detector.__file__,'exec'),scope)
    monkeypatch.setattr(detector,'_nb_plateau_offset',lambda *args,**kwargs:None)
    scope['_nb_plateau_offset']=detector._nb_plateau_offset
    history=reader();history._rp=history._wp=4000
    rec.set_history_reader(history)
    d=det();d['sf']=7
    rec.update_deferred([d],np.zeros(4000,np.complex64),sample_pos=4000,
        need_tail_n=2000,history_anchor=history.history_anchor(4000))
    history._rp=history._wp=7000;rec.feed_tail(np.zeros(500,np.complex64),sample_pos=7000)
    end=time.monotonic()+2
    while rec._save_queue.empty() and time.monotonic()<end:time.sleep(.005)
    rec._save_queue.put(None)
    worker=threading.Thread(target=scope['_save_worker'],args=(rec,))
    worker.start();worker.join(2);assert not worker.is_alive()
    files=list(Path(rec.export_dir).glob('*.cf32'))
    assert len(files)==1
    assert len(np.fromfile(files[0],np.complex64))==750
    assert rec._shared.stats()['completed_views']==1
    assert rec._shared.stats()['budget']['outbox_slots']==0
    assert not rec._shared.failure


def test_shared_flush_never_waits_for_failed_writer(rec):
    rec.set_history_reader(reader())
    rec._shared._fail('writer failed')
    # Poison legacy joins to make waiting visible without a hanging test.
    rec._save_queue.join=lambda:pytest.fail('failed shared writer must not be joined')
    rec._crop_queue.join=lambda:pytest.fail('shared path must not join legacy crop queue')
    with pytest.raises(RuntimeError,match='writer failed'):rec.flush()


def test_actual_pool_registration_uses_anchor_without_copying_trailing_slots(rec):
    import ast
    from collections import deque
    from pathlib import Path
    from types import SimpleNamespace
    tree=ast.parse(Path(detector.__file__).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    commit=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='_commit_oldest')
    node=next(n for n in commit.body if isinstance(n,ast.If) and ast.unparse(n.test)=='recorder')
    history=reader();history._rp=history._wp=8000
    rec.set_history_reader(history)
    def slot(index):
        assert index==0,'main must not copy trailing pool history for shared captures'
        return np.zeros(4000,np.complex64)
    env=dict(vars(detector),recorder=rec,dets0=[det()],win_n=4000,hop_n=500,
        _inflight=deque([{'seq':1,'slot':1}]),_pool=SimpleNamespace(slot_array=slot),
        it={'seq':0,'slot':0,'tot_s':4000,'pre_hop':None,'history_anchor':(4000,0,0)})
    exec(compile(ast.Module([node],type_ignores=[]),detector.__file__,'exec'),env)
    assert rec._shared.stats()['active_views']==1
    assert not rec._pending
    function=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='_tail_windows_needed')
    env['_tail_seg_n']=500
    exec(compile(ast.Module([function],type_ignores=[]),detector.__file__,'exec'),env)
    assert env['_tail_windows_needed']([det()])==0


def test_slow_registration_defers_history_copy_to_service(rec):
    history=reader();history._rp=history._wp=12000
    rec.set_history_reader(history)
    detector._record_slow_capture(rec,[det()],np.zeros(4000,np.complex64),4000,
        sample_pos=12000,hop_n=500,win_n=4000,recent_hops=[],
        history_reader=history,history_anchor=(4000,0,0))
    assert rec._shared.stats()['active_views']==1
    assert not rec._pending


def test_shared_writer_releases_mix_backing_before_lease_and_propagates_role(rec):
    import ast
    import gc
    import threading
    import weakref
    from pathlib import Path
    tree=ast.parse(Path(detector.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SignalRecorder')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_save_worker')
    owners=[]
    class Numpy:
        def __getattr__(self,name):return getattr(np,name)
        def arange(self,*args,**kw):
            a=np.arange(*args,**kw);owners.append(weakref.ref(a));return a
    scope=dict(vars(detector),np=Numpy(),_CROP_CENTER_FIX=True,
               _nb_plateau_offset=lambda *a,**k:10000.)
    exec(compile(ast.Module([node],type_ignores=[]),detector.__file__,'exec'),scope)
    calls=[]
    class Decoder:
        _pressure_n=8
        def pending(self):return 0
        def _cap_try_reserve(self,*args):return True,0,0
        def submit(self,*args,**kwargs):calls.append((args,kwargs))
    rec.set_decoder(Decoder())
    history=reader();history._rp=history._wp=4000;rec.set_history_reader(history)
    d=det();d['preamble_t_s']=1.
    rec.update_deferred([d],np.zeros(4000,np.complex64),sample_pos=4000,
        need_tail_n=2000,history_anchor=history.history_anchor(4000))
    history._rp=history._wp=7000;rec.feed_tail(np.zeros(500,np.complex64),sample_pos=7000)
    worker=threading.Thread(target=scope['_save_worker'],args=(rec,));worker.start()
    try:
        end=time.monotonic()+2
        while rec._shared.stats()['completed_views']<1 and time.monotonic()<end:time.sleep(.005)
        assert rec._shared.stats()['completed_views']==1
        gc.collect()
        assert owners and all(ref() is None for ref in owners)
        assert len(calls)==2  # full primary plus existing time-isolated derivative
        assert all(k.get('role')=='shared_full' for _,k in calls)
        assert all('_shared_full_' in args[1] for args,_ in calls)
    finally:
        rec._save_queue.put(None);worker.join(2);assert not worker.is_alive()
