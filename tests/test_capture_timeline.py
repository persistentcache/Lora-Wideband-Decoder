"""Shared live captures preserve admitted spans without producer backpressure."""
import queue
import threading
import time

import numpy as np
import pytest

import detector
from capture_timeline import CaptureBudget, CaptureCapacityError, SharedCapture, SharedCaptureFailure


def test_budget_tracks_outbox_and_workspace_until_idempotent_release():
    budget = CaptureBudget(1000, 2)
    view = budget.reserve('view', 100, slots=1)
    workspace = budget.reserve('workspace', 700)
    with pytest.raises(CaptureCapacityError):
        budget.reserve('snapshot', 201)
    with pytest.raises(CaptureCapacityError):
        budget.reserve('view', 0, slots=2)
    assert budget.stats()['used_bytes'] == 800
    workspace.resize(500)
    assert budget.stats()['used_bytes'] == 600
    workspace.release(); workspace.release()
    view.release(); view.release()
    assert budget.stats()['used_bytes'] == budget.stats()['outbox_slots'] == 0


def test_failed_resize_is_atomic():
    budget = CaptureBudget(1000, 1)
    lease = budget.reserve('band', 900)
    with pytest.raises(CaptureCapacityError):
        lease.resize(1001)
    assert lease.size == budget.stats()['used_bytes'] == 900
    lease.release()


def reader(capacity=60000):
    # Actual production ring/history methods, with no input/device thread.
    r = detector.StreamBuffer.__new__(detector.StreamBuffer)
    r._lock = threading.Lock()
    r._ring_n = capacity; r._ring_elems = capacity * 2
    r._ring = np.empty(capacity * 2, np.int16)
    n = np.arange(capacity)
    r._ring[::2] = (n % 30000).astype(np.int16)
    r._ring[1::2] = -(n % 30000).astype(np.int16)
    r._scale = np.float32(1)
    r._rp = r._wp = r._total_drops = r._history_generation = 0
    return r


def detection():
    return {'sf': 2, 'bw': 62.5, 'freq_hz': 915e6,
            'peak_power_db': 20, 'detect_conf': .99}


def engine(*, capacity=60000, submit=None, budget_bytes=2_000_000, max_views=16):
    r = reader(capacity)
    calls = []
    def factory(*args):
        calls.append(args)
        return detector._StreamDecimator(*args)
    e = SharedCapture(rate=1000, center=915e6, hop_n=500, reader=r,
                      decimator_factory=factory, submit=submit or (lambda item: False),
                      budget=CaptureBudget(budget_bytes, max_views), threaded=False)
    return e, r, calls


def advance(e, r, end):
    r._rp = r._wp = end
    e.advance(r.history_anchor(end))
    # The synchronous test drives the actual worker service, never producer DSP.
    for _ in range(200):
        if not e.step():
            break
    else:
        pytest.fail('crop service did not quiesce')


def register(e, end, *, tail=28000):
    return e.register([detection()], base_start=end-4000, base_end=end,
                      end=end+tail, anchor=(end, 0, 0), target_factor=8,
                      export_bw=62.5, pre_hop_n=0, wb_n=4000,
                      sample_pos=end)


def test_more_than_three_overlapping_views_share_one_fir_and_keep_full_tails():
    submitted = []
    e, r, calls = engine(submit=lambda item: submitted.append(item) or True)
    jobs = []
    for end in range(4000, 43001, 500):
        r._rp = r._wp = end
        if end in (4000, 6500, 9000, 11500, 14000):
            jobs.append(register(e, end))
        advance(e, r, end)
        if submitted:
            item = submitted.pop(0)
            output = item.stream(0)
            assert len(output) == 4000  # (4s base +28s tail)*125samples/s
            assert item.complete and item.end - item.start == 32000
            item.release()
    for _ in range(20):
        e.step()
        if submitted:
            item = submitted.pop(0)
            assert len(item.stream(0)) == 4000
            item.release()
    assert sum(args[1] == 1000 for args in calls) == 1  # one raw-rate FIR
    assert e.stats()['completed_views'] == 5
    assert e.stats()['incomplete_views'] == 0
    assert not e.failure
    e.close(wait=False)


def test_full_writer_queue_never_blocks_producer_or_discards_ready_spans():
    blocked = queue.Queue(maxsize=1)
    blocked.put('occupied')
    def submit(item):
        try: blocked.put_nowait(item); return True
        except queue.Full: return False
    e, r, calls = engine(submit=submit)
    r._rp = r._wp = 4000
    register(e, 4000, tail=2000)
    advance(e, r, 7000)
    before = e.stats()
    t = time.perf_counter()
    for _ in range(100): e.advance((7000,0,0))
    assert time.perf_counter()-t < .1
    assert e.stats()['ready_views'] == 1
    assert e.stats()['incomplete_views'] == 0
    assert before['budget']['outbox_slots'] == 1
    blocked.get_nowait()
    e.step()
    item = blocked.get_nowait()
    assert item.complete and len(item.stream(0)) == 750
    held = e.stats()['budget']['used_bytes']
    e.close(wait=False)
    assert e.stats()['budget']['used_bytes'] > 0  # writer still owns its lease
    assert held > 0
    item.release(); item.release()
    assert e.stats()['budget']['used_bytes'] == 0


@pytest.mark.parametrize('failure', ['drop', 'retune', 'lap'])
def test_lost_history_is_fatal_and_never_splices_future_samples(failure):
    submitted = []
    e, r, _ = engine(capacity=16000, submit=lambda item: submitted.append(item) or True)
    r._rp = r._wp = 4000
    register(e, 4000)
    advance(e, r, 5000)
    if failure == 'drop':
        r._total_drops = 500; r._rp = r._wp = 6000
    elif failure == 'retune':
        r._history_generation = 1; r._rp = r._wp = 6000
    else:
        r._rp = r._wp = 24000
    e.advance(r.history_anchor())
    e.step()
    with pytest.raises(SharedCaptureFailure): e.raise_if_failed()
    assert not submitted
    assert e.stats()['incomplete_views'] == 1
    e.close(wait=False)


def test_retention_capacity_failure_is_explicit_main_loop_state():
    e, r, _ = engine(budget_bytes=100000)
    r._rp = r._wp = 4000
    register(e, 4000, tail=1_000_000)
    with pytest.raises(SharedCaptureFailure): e.raise_if_failed()
    assert e.stats()['incomplete_views'] == 1
    e.close(wait=False)


def test_late_registration_never_rolls_back_current_watermark():
    submitted = []
    e, r, _ = engine(submit=lambda item: submitted.append(item) or True)
    r._rp = r._wp = 8000
    e.advance((8000,0,0))
    register(e, 4000, tail=2000)
    for _ in range(30): e.step()
    assert len(submitted) == 1
    submitted[0].release(); e.close(wait=False)


def test_save_sink_exception_releases_nontransferred_view_and_reports_fatal():
    def broken(item): raise OSError('disk sink failed')
    e, r, _ = engine(submit=broken)
    r._rp = r._wp = 4000
    register(e,4000,tail=1000)
    advance(e,r,6000)
    assert 'disk sink failed' in e.failure
    assert e.stats()['incomplete_views'] == 1
    e.close(wait=False)
    e.step()
    assert e.stats()['budget']['used_bytes'] == 0


def test_done_view_metadata_does_not_accumulate_forever():
    submitted = []
    e, r, _ = engine(submit=lambda item: submitted.append(item) or True)
    for end in (4000,10000,16000):
        r._rp = r._wp = end
        register(e,end,tail=1000)
        advance(e,r,end+2000)
        assert len(submitted)==1
        submitted.pop().release()
    assert e.stats()['completed_views']==3
    assert len(e._views)==0
    e.close(wait=False)


def test_real_writer_output_contains_every_contiguous_sample_through_required_end():
    from scipy.signal import resample_poly, firwin
    submitted = []
    e, r, _ = engine(submit=lambda item: submitted.append(item) or True)
    r._rp = r._wp = 4000
    register(e,4000,tail=28000)
    advance(e,r,33000)
    item=submitted.pop()
    actual=item.stream(0)
    raw=np.arange(33000,dtype=np.float32)%30000
    raw=(raw-1j*raw).astype(np.complex64)
    shared=resample_poly(raw,1,4,window=firwin(81,.25,window=('kaiser',5)).astype(np.float32))
    expected=resample_poly(shared[:8020],1,2,
                          window=firwin(41,.5,window=('kaiser',5)).astype(np.float32))[:4000]
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=.02)
    item.release();e.close(wait=False)


def test_emitted_samples_not_raw_watermark_control_readiness():
    submitted=[]
    e,r,_=engine(submit=lambda item: submitted.append(item) or True)
    r._rp=r._wp=4000
    register(e,4000,tail=2000)
    advance(e,r,6000)
    assert not submitted  # shared and per-view FIR still need real right context
    advance(e,r,6120)
    assert len(submitted)==1 and len(submitted[0].stream(0))==750
    submitted[0].release();e.close(wait=False)


def test_producer_does_not_wait_while_worker_copies_history():
    e,r,_=engine()
    entered,proceed=threading.Event(),threading.Event()
    original=r.history_parts
    def paused(*args):
        entered.set()
        assert proceed.wait(2)
        yield from original(*args)
    r.history_parts=paused
    r._rp=r._wp=8000
    e.advance((8000,0,0));register(e,4000,tail=2000)
    worker=threading.Thread(target=e.step)
    worker.start();assert entered.wait(1)
    t=time.perf_counter()
    e.advance((8000,0,0))
    register(e,6500,tail=2000)
    assert time.perf_counter()-t<.1
    proceed.set();worker.join(2);assert not worker.is_alive()
    e._fail('test shutdown');e.close(wait=False)


def test_full_outbox_slot_limit_is_reserved_before_capture_completion():
    e,r,_=engine(max_views=1)
    r._rp=r._wp=4000
    register(e,4000,tail=28000)
    assert e.stats()['budget']['outbox_slots']==1
    register(e,6500,tail=28000)
    with pytest.raises(SharedCaptureFailure):e.raise_if_failed()
    assert 'slots' in e.failure
    e.close(wait=False)


def test_writer_stall_sets_fatal_state_without_reclaiming_live_lease():
    now=[10.0];submitted=[]
    r=reader()
    e=SharedCapture(rate=1000,center=915e6,hop_n=500,reader=r,
        decimator_factory=detector._StreamDecimator,
        submit=lambda item: submitted.append(item) or True,
        budget=CaptureBudget(2_000_000,4),threaded=False,
        writer_stall_s=5,clock=lambda:now[0])
    r._rp=r._wp=4000;register(e,4000,tail=1000)
    advance(e,r,6000)
    assert len(submitted)==1
    before=e.stats()['budget']['used_bytes']
    now[0]=16;e.step()
    with pytest.raises(SharedCaptureFailure):e.raise_if_failed()
    assert 'writer' in e.failure
    assert e.stats()['budget']['used_bytes']==before
    e.close(wait=False)
    assert e.stats()['budget']['used_bytes']>0
    submitted[0].release()
    assert e.stats()['budget']['used_bytes']==0


def test_blocked_failure_report_cannot_hold_producer_or_cleanup_lock():
    entered,proceed=threading.Event(),threading.Event()
    def report(_):
        entered.set();assert proceed.wait(2)
    e,r,_=engine()
    e.report=report
    # Failure publication itself must only set state, never call a reporter.
    t=time.perf_counter();e._fail('simulated writer failure')
    assert time.perf_counter()-t<.1
    worker=threading.Thread(target=e.step);worker.start()
    assert entered.wait(1)
    t=time.perf_counter();e.advance((0,1,0));e.close(wait=False)
    with pytest.raises(SharedCaptureFailure):e.raise_if_failed()
    assert time.perf_counter()-t<.1
    proceed.set();worker.join(2);assert not worker.is_alive()


def test_skip_while_render_is_paused_retains_writer_lease_and_rejects_result():
    submitted=[]
    e,r,_=engine(submit=lambda view:submitted.append(view) or True)
    r._rp=r._wp=4000;register(e,4000,tail=2000);advance(e,r,7000)
    view=submitted.pop()
    entered,proceed=threading.Event(),threading.Event()
    original=e.factory
    def paused(*args):
        entered.set();assert proceed.wait(2)
        return original(*args)
    e.factory=paused
    errors=[]
    def render():
        try:view.stream(0)
        except SharedCaptureFailure as exc:errors.append(str(exc))
    worker=threading.Thread(target=render);worker.start();assert entered.wait(1)
    used=e.stats()['budget']['used_bytes']
    t=time.perf_counter();e.advance((7500,500,0));e.close(wait=False)
    assert time.perf_counter()-t<.1
    assert e.stats()['budget']['used_bytes']==used
    proceed.set();worker.join(2);assert not worker.is_alive()
    assert errors
    view.release();assert e.stats()['budget']['used_bytes']==0


def test_idle_retune_cannot_reuse_old_frequency_metadata():
    e,r,_=engine()
    e.advance((4000,0,0));e.advance((4500,0,1))
    with pytest.raises(SharedCaptureFailure,match='retuned'):e.raise_if_failed()
    e.close(wait=False)


def test_unavailable_signal_is_explicit_non_admission_not_wrapped_alias():
    e,r,_=engine()
    d=detection();d['freq_hz']+=600  # outside the original +/-500Hz window
    e.register([d],base_start=0,base_end=4000,end=6000,anchor=(4000,0,0),
        target_factor=8,export_bw=62.5,pre_hop_n=0,wb_n=4000,sample_pos=4000)
    assert not e.failure
    assert e.stats()['non_admitted'] == 1
    assert e.stats()['active_views'] == 0
    e.close(wait=False)


def test_delayed_older_band_history_is_served_before_newer_band():
    e,r,_=engine()
    r._rp=r._wp=16000;e.advance((16000,0,0))
    for base,frequency in [(10000,915e6+300),(0,915e6-300)]:
        d=detection();d['freq_hz']=frequency
        e.register([d],base_start=base,base_end=base+4000,end=base+6000,
            anchor=(base+4000,0,0),target_factor=8,export_bw=62.5,
            pre_hop_n=0,wb_n=4000,sample_pos=base+4000)
    history=r.history_parts;copied=[]
    def measured(anchor,*args):
        copied.append(anchor[0]);yield from history(anchor,*args)
    r.history_parts=measured
    e.step()
    assert copied==[0]
    e._fail('test shutdown');e.close(wait=False)


def test_prime_target_factor_retains_exact_rate_without_expanding_to_full_wideband():
    submitted=[]
    e,r,calls=engine(submit=lambda v:submitted.append(v) or True)
    r._rp=r._wp=4000
    e.register([detection()],base_start=0,base_end=4000,end=6000,
        anchor=(4000,0,0),target_factor=5,export_bw=100,
        pre_hop_n=0,wb_n=4000,sample_pos=4000)
    advance(e,r,7000)
    view=submitted.pop()
    assert e.stats()['band_details'][0]['factor']==5
    actual=view.stream(0)
    raw=np.arange(7000,dtype=np.float32);raw=(raw-1j*raw).astype(np.complex64)
    dec=detector._StreamDecimator(0,1000,5)
    expected=dec.feed(raw)[:1200]
    np.testing.assert_array_equal(actual,expected)
    view.release();e.close(wait=False)


def test_same_span_different_preamble_hint_is_not_an_exact_duplicate():
    e,r,_=engine();r._rp=r._wp=4000
    for timing in (0.25,1.25):
        d=detection();d['preamble_t_s']=timing
        e.register([d],base_start=0,base_end=4000,end=6000,anchor=(4000,0,0),
            target_factor=8,export_bw=62.5,pre_hop_n=0,wb_n=4000,sample_pos=4000)
    assert e.stats()['active_views']==2
    assert e.stats()['bands']==1
    e._fail('test shutdown');e.close(wait=False)


@pytest.mark.parametrize('offset',[-460.,460.])
def test_real_edge_signal_uses_original_center_filter_when_export_crosses_input(offset):
    submitted=[]
    e,r,_=engine(submit=lambda v:submitted.append(v) or True)
    d=detection();d['freq_hz']+=offset  # payload +/-31.25Hz fits; export +/-62.5 does not
    r._rp=r._wp=4000
    e.register([d],base_start=0,base_end=4000,end=6000,anchor=(4000,0,0),
        target_factor=8,export_bw=62.5,pre_hop_n=0,wb_n=4000,sample_pos=4000)
    advance(e,r,7000)
    assert not e.failure
    assert e.stats()['band_details'][0]['factor']==8
    view=submitted.pop();actual=view.stream(0)
    raw=np.arange(7000,dtype=np.float32);raw=(raw-1j*raw).astype(np.complex64)
    dec=detector._StreamDecimator(offset,1000,8)
    np.testing.assert_allclose(actual,dec.feed(raw)[:750],rtol=1e-6,atol=.002)
    view.release();e.close(wait=False)


@pytest.mark.parametrize('end',[5000,6000])
def test_eof_before_endpoint_or_real_right_context_never_claims_complete(end):
    submitted=[]
    e,r,_=engine(submit=lambda v:submitted.append(v) or True)
    r._rp=r._wp=4000;register(e,4000,tail=2000)
    advance(e,r,end);e.close(wait=False)
    for _ in range(20):
        if not e.step():break
    assert not submitted
    assert e.stats()['completed_views']==0
    assert e.stats()['incomplete_views']==1
    with pytest.raises(SharedCaptureFailure,match='input ended'):e.raise_if_failed()
    assert e.stats()['budget']['used_bytes']==0


def test_complete_eof_drains_and_releases_only_after_writer_done():
    submitted=[]
    e,r,_=engine(submit=lambda v:submitted.append(v) or True)
    r._rp=r._wp=4000;register(e,4000,tail=2000);advance(e,r,6120)
    assert len(submitted)==1
    e.close(wait=False);e.step()
    assert not e.failure
    assert e.stats()['budget']['used_bytes']>0
    assert len(submitted[0].stream(0))==750
    submitted[0].release();e.step()
    assert e.stats()['budget']['used_bytes']==0
    assert e.stats()['completed_views']==1


def test_failed_service_thread_start_releases_constructor_reservations(monkeypatch):
    budget=CaptureBudget(2_000_000,4)
    def fail(*args):raise RuntimeError('thread start failed')
    monkeypatch.setattr(threading.Thread,'start',fail)
    with pytest.raises(RuntimeError,match='thread start failed'):
        SharedCapture(rate=1000,center=915e6,hop_n=500,reader=reader(),
            decimator_factory=detector._StreamDecimator,submit=lambda v:False,
            budget=budget)
    assert budget.stats()['used_bytes']==0
