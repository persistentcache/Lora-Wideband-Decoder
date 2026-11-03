"""Shared capture may copy published raw samples without advancing detection."""
import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import detector
from capture_timeline import CaptureBudget, SharedCapture, SharedCaptureFailure
from test_slow_capture import ring_reader, samples
from test_capture_timeline import engine, register


def api(reader, name):
    method=getattr(reader,name,None)
    assert callable(method), 'StreamBuffer lacks the explicit '+name+' boundary'
    return method


def pump(capture):
    for _ in range(1000):
        if not capture.step():break
    else:pytest.fail('capture did not quiesce')


def test_published_unconsumed_samples_leave_legacy_mapping_and_counters_unchanged():
    r=ring_reader(4000,received=6000)
    received_anchor=api(r,'received_history_anchor')
    received_parts=api(r,'received_history_parts')
    before=r.progress()
    assert received_anchor()==(6000,0,0)
    assert r.history_anchor(4000)==(4000,0,0)
    assert r.history_anchor(6000) is None
    with pytest.raises(detector.StreamHistoryUnavailable):
        list(r.history_parts((4000,0,0),2000,500))
    out=list(received_parts((4000,0,0),2000,500))
    assert all(len(part)<=500 for part in out)
    np.testing.assert_array_equal(np.concatenate(out),samples(4000,2000))
    assert r.progress()==before
    with pytest.raises(detector.StreamHistoryUnavailable):
        list(received_parts((6000,0,0),1,500))


def test_reader_publishes_wp_only_after_both_wrap_copy_segments(monkeypatch):
    r=ring_reader(8,capacity=10)
    anchor=api(r,'received_history_anchor'); parts=api(r,'received_history_parts')
    entered,release,snapshot_done=threading.Event(),threading.Event(),threading.Event()
    snapshots=[]
    class PausedRing(np.ndarray):
        paused=False
        def __setitem__(self,key,value):
            super().__setitem__(key,value)
            if not self.paused:
                self.paused=True;entered.set()
                assert release.wait(2)
    r._ring=r._ring.view(PausedRing)
    raw=np.empty(8,np.int16);raw[::2]=np.arange(8,12);raw[1::2]=-np.arange(8,12)
    r._fp=io.BytesIO(raw.tobytes());r.sc16=True;r.bps=4;r._chunk_bytes=16;r._eof=False
    import cpu_affinity
    monkeypatch.setattr(cpu_affinity,'pin_reader',lambda:None)
    monkeypatch.setattr(detector.os,'sched_setscheduler',lambda *args:None)
    writer=threading.Thread(target=r._reader_loop)
    snapshot=threading.Thread(target=lambda:(snapshots.append(anchor()),snapshot_done.set()))
    writer.start()
    try:
        assert entered.wait(1)
        assert r._wp==8, 'write pointer exposed a half-written wrap chunk'
        snapshot.start()
        assert not snapshot_done.wait(.02), 'snapshot did not synchronize with the ring writer'
    finally:
        release.set();writer.join(2)
        if snapshot.ident is not None:snapshot.join(2)
    assert not writer.is_alive() and not snapshot.is_alive()
    assert snapshots==[(12,0,0)]
    np.testing.assert_array_equal(np.concatenate(list(parts((8,0,0),4,2))),samples(8,4))
    assert r._rp==8 and r._total_drops==0


@pytest.mark.parametrize('mutation',['lap','skip','retune'])
@pytest.mark.parametrize('when',['before','during'])
def test_published_copy_rejects_lap_skip_and_retune_before_or_during_copy(monkeypatch,mutation,when):
    r=ring_reader(4000,received=6000)
    parts=api(r,'received_history_parts')
    original=r._convert_span
    def mutate():
        if mutation=='lap':r._wp+=r._ring_n
        elif mutation=='skip':assert r.skip_to_latest(500)==1500
        else:r.invalidate_history()
    if when=='before':mutate()
    else:
        def copy_then_change(pos,count):
            out=original(pos,count);mutate();return out
        monkeypatch.setattr(r,'_convert_span',copy_then_change)
    with pytest.raises(detector.StreamHistoryUnavailable):
        list(parts((4000,0,0),500,500))
    assert r._rp==(5500 if mutation=='skip' else 4000)


def test_safe_new_publication_during_copy_is_not_mistaken_for_a_lap(monkeypatch):
    r=ring_reader(4000,received=6000)
    parts=api(r,'received_history_parts');original=r._convert_span
    def copy_then_publish(pos,count):
        out=original(pos,count);r._wp+=500;return out
    monkeypatch.setattr(r,'_convert_span',copy_then_publish)
    np.testing.assert_array_equal(next(parts((4000,0,0),500,500)),samples(4000,500))
    assert r._rp==4000 and r._wp==6500 and r._total_drops==0


def test_shared_early_readiness_follows_published_data_without_main_advance():
    submitted=[];e,r,_=engine(submit=lambda view:submitted.append(view) or True)
    r._rp=4000;r._wp=6000
    register(e,4000,tail=2000)
    parent=e._views[0]
    early=e.add_slice_view(parent,parent.refs[0],0,5500,role='early',request_id='test')
    before=r.progress();pump(e)
    assert submitted==[early], 'shared capture still waits for main-loop consumption'
    assert len(early.stream(0))==688
    early.release();submitted.clear()
    assert r.progress()==before and parent.state=='collecting'
    r._wp=6120;pump(e)
    assert submitted==[parent] and len(parent.stream(0))==750
    parent.release();e.close(wait=False)
    assert r._rp==4000 and r._total_drops==0
    assert e.stats()['budget']['used_bytes']==0


@pytest.mark.parametrize('capability',['none','anchor_only','parts_only'])
def test_legacy_reader_fallback_keeps_consumed_clock(capability):
    r=ring_reader(4000,received=7000)
    legacy=SimpleNamespace(history_parts=r.history_parts)
    if capability=='anchor_only':legacy.received_history_anchor=lambda:pytest.fail('half API used')
    if capability=='parts_only':legacy.received_history_parts=lambda *args:pytest.fail('half API used')
    sent=[]
    e=SharedCapture(rate=1000,center=915e6,hop_n=500,reader=legacy,
        decimator_factory=detector._StreamDecimator,submit=lambda v:sent.append(v) or True,
        budget=CaptureBudget(2_000_000,16),threaded=False)
    register(e,4000,tail=2000);pump(e)
    assert not sent
    r._rp=7000;e.advance(r.history_anchor());pump(e)
    assert len(sent)==1 and len(sent[0].stream(0))==750
    sent[0].release();e.close(wait=False)
    assert e.stats()['budget']['used_bytes']==0


@pytest.mark.parametrize('received,complete',[(6000,False),(6120,True)])
def test_close_drains_exact_final_publication_including_partial_hop_context(received,complete):
    sent=[];e,r,_=engine(submit=lambda v:sent.append(v) or True)
    r._rp=4000;r._wp=received;register(e,4000,tail=2000)
    e.close(wait=False);pump(e)
    assert bool(sent)==complete
    if complete:
        assert len(sent[0].stream(0))==750
        assert e.stats()['budget']['used_bytes']>0
        sent[0].release();pump(e)
    else:assert 'input ended' in e.failure
    assert e.stats()['budget']['used_bytes']==0
    assert e.stats()['budget']['outbox_slots']==0
    assert r._rp==4000


def test_close_does_not_chase_later_publication_or_accept_a_racing_poll(monkeypatch):
    sent=[];e,r,_=engine(submit=lambda v:sent.append(v) or True)
    received=api(r,'received_history_anchor')
    r._rp=4000;r._wp=6000;register(e,4000,tail=2000)
    entered,release=threading.Event(),threading.Event()
    def paused_snapshot():
        if threading.current_thread().name=='poll-test':
            entered.set();assert release.wait(2)
        return received()
    monkeypatch.setattr(r,'received_history_anchor',paused_snapshot)
    worker=threading.Thread(target=e.step,name='poll-test');worker.start()
    try:
        assert entered.wait(1)
        e.close(wait=False)
        assert e._watermark[0]==6000
        r._wp=7000
    finally:release.set();worker.join(2)
    assert not worker.is_alive()
    monkeypatch.setattr(r,'received_history_anchor',lambda:pytest.fail('polled after closing'))
    pump(e)
    assert e._watermark[0]==6000 and not sent
    assert 'input ended' in e.failure
    assert e.stats()['budget']['used_bytes']==0


def test_published_epoch_change_cancels_held_writer_without_reclaiming_its_lease():
    sent=[];e,r,_=engine(submit=lambda v:sent.append(v) or True)
    r._rp=4000;r._wp=7000;register(e,4000,tail=2000);pump(e)
    assert len(sent)==1
    held=e.stats()['budget']['used_bytes']
    r.invalidate_history();e.step()
    assert e.failure and e.stats()['budget']['used_bytes']==held
    with pytest.raises(SharedCaptureFailure):sent[0].stream(0)
    sent[0].release();e.close(wait=False)
    assert e.stats()['budget']['used_bytes']==0


@pytest.mark.parametrize('stale',[False,True])
def test_real_service_waits_when_publication_is_unchanged_then_captures_new_data(monkeypatch,stale):
    r=ring_reader(4000,received=4000)
    snapshot=api(r,'received_history_anchor')
    calls=[];sent=[];received=threading.Event();old_snapshot=snapshot();return_old=[stale]
    def counted():
        calls.append(1)
        return old_snapshot if return_old[0] else snapshot()
    monkeypatch.setattr(r,'received_history_anchor',counted)
    def submit(view):
        sent.append(view);received.set();return True
    e=SharedCapture(rate=1000,center=915e6,hop_n=500,reader=r,
        decimator_factory=detector._StreamDecimator,submit=submit,
        budget=CaptureBudget(2_000_000,16),threaded=True)
    try:
        if stale:
            with r._lock:r._wp=4500
            e.advance((4500,0,0))
        # This runs the actual service loop, not a stub wait implementation.
        threading.Event().wait(.12)
        assert 1<=len(calls)<=20, 'unchanged publication hot-spins instead of yielding'
        return_old[0]=False
        register(e,4000,tail=2000)
        with r._lock:r._wp=6120
        assert received.wait(1), 'new published samples did not wake/progress the view'
        assert len(sent[0].stream(0))==750 and r._rp==4000
        sent[0].release();sent.clear()
    finally:
        if sent:
            for view in sent:view.release()
        e.close(wait=True)
    assert not e._thread.is_alive()
    assert e.stats()['budget']['used_bytes']==0
