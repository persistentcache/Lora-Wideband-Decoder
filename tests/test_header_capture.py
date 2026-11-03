"""Optional early capture views preserve the independently retained full span."""
import importlib.util


def test_header_capture_controller_exists():
    assert importlib.util.find_spec('header_capture_controller') is not None, (
        'shared capture has no bounded header/early-view controller')

import math
from types import SimpleNamespace
import pytest


class FakeHeaders:
    def __init__(self):
        self.requests={};self.results=[];self.failure=None
    def submit(self, request, lease):
        self.requests[request.request_id]=(request,lease);return True
    def poll(self):
        return self.results.pop(0) if self.results else None
    def complete(self, identifier, *, status='ok', endpoint=12000, token=None):
        request,lease=self.requests.pop(identifier)
        lease.release()
        metadata={'bw':125.,'ready_end_bw':endpoint,'preamble_start_bw':1024,
                  'identity_token':request.identity_token if token is None else token}
        self.results.append(SimpleNamespace(request=request,status=status,metadata=metadata))
    def cancel(self, identifier):
        if identifier in self.requests:self.complete(identifier,status='cancelled')
    def close(self,**kwargs):
        for identifier in list(self.requests):self.cancel(identifier)
        return True


@pytest.fixture
def fixture(tmp_path):
    import detector
    from capture_timeline import CaptureBudget,SharedCapture
    from header_capture_controller import HeaderCaptureController
    from test_capture_timeline import reader
    history=reader(400000)
    submitted=[]
    capture=SharedCapture(rate=1000,center=915e6,hop_n=500,reader=history,
        decimator_factory=detector._StreamDecimator,submit=lambda view:submitted.append(view) or True,
        budget=CaptureBudget(12*1024*1024,32),threaded=False)
    headers=FakeHeaders()
    controller=HeaderCaptureController(capture,headers,directory=tmp_path,max_prefix_bytes=100000)
    capture.header_controller=controller
    detections=[dict(sf=7,bw=125.,freq_hz=915e6,peak_power_db=20,detect_conf=.99,preamble_t_s=20.)]
    history._wp=history._rp=4000
    capture.register(detections,base_start=0,base_end=4000,end=200000,anchor=(4000,0,0),
        target_factor=8,export_bw=62.5,pre_hop_n=0,wb_n=4000,sample_pos=4000)
    yield SimpleNamespace(capture=capture,history=history,headers=headers,
                          controller=controller,submitted=submitted,path=tmp_path)
    controller.close()
    for view in list(submitted):view.release()
    capture.close(wait=False)


def advance(f,end):
    f.history._rp=f.history._wp=end
    f.capture.advance(f.history.history_anchor())
    for _ in range(1200):
        if not f.capture.step():break
    else:pytest.fail('capture failed to quiesce')


def prepare(f):
    advance(f,71000)
    assert len(f.submitted)==1
    prefix=f.submitted.pop()
    assert prefix.role=='header_prefix'
    f.controller.write_prefix(prefix)
    prefix.release()
    assert len(f.headers.requests)==1
    return next(iter(f.headers.requests))


def test_prefix_uses_per_hint_origin_and_preserves_original_full_span(fixture):
    f=fixture;original=f.capture._views[0];identifier=prepare(f)
    request,_=f.headers.requests[identifier]
    assert request.identity_token[-2]==11808 # hint20 -8*(128/125), on grid
    assert request.sample_count*8<=100000
    assert original.start==0 and original.end==200000 and original.state=='collecting'
    assert f.capture.stats()['budget']['by_kind']['header_hold']==8192
    assert f.capture.stats()['budget']['by_kind']['header_prefix_file']>0


def test_valid_header_creates_early_sibling_without_shortening_full(fixture):
    f=fixture;original=f.capture._views[0];identifier=prepare(f)
    f.headers.complete(identifier)
    f.capture.step()
    early=next(v for v in f.capture._views if v.role=='early')
    assert early.start==11808 and early.end==107808
    assert original.start==0 and original.end==200000 and original.state=='collecting'
    advance(f,109000)
    item=f.submitted.pop();assert item is early
    assert len(item.stream(0))==12000
    item.release()
    advance(f,202000)
    full=f.submitted.pop();assert full is original and full.role=='shared_full'
    assert len(full.stream(0))==25000
    full.release()
    assert f.capture.stats()['incomplete_views']==0
    assert f.controller.counts['early']==1


def test_short_frame_already_complete_inside_prefix_is_saved_immediately(fixture):
    f=fixture;identifier=prepare(f)
    f.headers.complete(identifier,endpoint=4000)
    f.capture.step()
    assert len(f.submitted)==1,'short valid frame was incorrectly delayed until maximum capture'
    early=f.submitted.pop()
    assert early.role=='early' and early.end==43808
    assert len(early.stream(0))==4000
    early.release()
    assert any(v.role=='shared_full' and v.end==200000 for v in f.capture._views)


@pytest.mark.parametrize('status,token,endpoint',[
    ('abstained',None,12000),('ok',('stale',),12000),('ok',None,1000000)])
def test_failed_stale_or_outside_header_leaves_full_fallback(fixture,status,token,endpoint):
    f=fixture;identifier=prepare(f)
    f.headers.complete(identifier,status=status,token=token,endpoint=endpoint)
    f.capture.step()
    assert not any(v.role=='early' for v in f.capture._views)
    advance(f,202000)
    full=f.submitted.pop();assert full.role=='shared_full'
    assert len(full.stream(0))==25000
    full.release()
    assert not list(f.path.glob('.header-*'))


def test_full_can_finish_while_header_pending_without_releasing_held_samples(fixture):
    f=fixture;identifier=prepare(f)
    advance(f,202000)
    full=f.submitted.pop();full.release()
    assert f.capture.stats()['bands']==1
    assert f.capture.stats()['budget']['by_kind']['header_hold']==8192
    f.headers.complete(identifier,endpoint=4000)
    f.capture.step()
    early=f.submitted.pop();assert early.role=='early'
    assert len(early.stream(0))==4000
    early.release()
    assert f.capture.stats()['bands']==0


def test_changed_drop_identity_cancels_checkpoint_without_early_publication(fixture):
    f=fixture;prepare(f)
    f.history._total_drops=1
    f.capture.advance(f.history.history_anchor())
    f.capture.step()
    assert f.controller._closed and not f.headers.requests
    assert not f.controller.counts['early']
    assert not list(f.path.glob('.header-*'))


def test_prefix_write_failure_does_not_cancel_full_capture(fixture,monkeypatch):
    import header_capture_controller as module
    f=fixture;advance(f,71000)
    prefix=f.submitted.pop()
    monkeypatch.setattr(module.tempfile,'mkstemp',lambda **kw:(_ for _ in ()).throw(OSError('disk full')))
    f.controller.write_prefix(prefix);prefix.release()
    assert not f.capture.failure and f.controller.counts['fallback']==1
    advance(f,202000)
    full=f.submitted.pop();assert len(full.stream(0))==25000;full.release()


def test_no_zero_padding_or_request_before_emitted_prefix_context(fixture):
    f=fixture
    advance(f,69408)
    assert not f.headers.requests and not f.submitted
    advance(f,69600)
    assert f.submitted and f.submitted[0].role=='header_prefix'


def test_ready_early_priority_has_bounded_full_view_fairness(fixture):
    f=fixture
    f.capture.header_controller=None
    original=f.capture._views[0];ref=original.refs[0]
    early=[f.capture.add_slice_view(original,ref,0,10000,role='early',request_id=i) for i in range(4)]
    # Everything is ready at the same service boundary, with full inserted first.
    advance(f,202000)
    order=[]
    for _ in range(1000):
        f.capture.step()
        if f.submitted:
            item=f.submitted.pop();order.append(item.role);item.release()
        if len(order)==5:break
    assert order[:3]==['early','early','shared_full']


def test_completed_filter_work_reports_zero_debt_while_writer_is_held(fixture):
    f=fixture
    f.capture.header_controller=None
    advance(f,202000)
    assert f.submitted and f.submitted[0].role=='shared_full'
    assert f.capture.stats()['crop_debt_s']==0


def test_header_hold_charges_source_metadata_after_full_release(fixture):
    f=fixture;original=f.capture._views[0];metadata_bytes=original.credit.size
    prepare(f);advance(f,202000)
    full=f.submitted.pop();full.release()
    assert f.capture.stats()['budget']['by_kind']['header_hold']>=metadata_bytes


def test_changed_source_phy_cannot_use_an_old_header_result(fixture):
    f=fixture;identifier=prepare(f)
    f.capture._views[0].detections[0]['sf']=8
    f.headers.complete(identifier)
    f.capture.step()
    assert not any(v.role=='early' for v in f.capture._views)


def test_optional_worker_poll_failure_retains_full_capture(fixture,monkeypatch):
    f=fixture
    def failed_poll():raise RuntimeError('header manager unavailable')
    monkeypatch.setattr(f.headers,'poll',failed_poll)
    advance(f,202000)
    assert not f.capture.failure
    full=f.submitted.pop();assert full.role=='shared_full'
    assert len(full.stream(0))==25000
    full.release()


@pytest.mark.parametrize('role',['header_prefix','early'])
def test_optional_render_failure_does_not_cancel_independent_full_span(fixture,monkeypatch,role):
    f=fixture
    if role=='header_prefix':
        advance(f,71000);optional=f.submitted.pop()
    else:
        identifier=prepare(f);f.headers.complete(identifier,endpoint=4000)
        f.capture.step();optional=f.submitted.pop()
    render=f.capture._render
    def fail(view,index):
        if view.role==role:raise MemoryError('optional renderer unavailable')
        return render(view,index)
    monkeypatch.setattr(f.capture,'_render',fail)
    if role=='header_prefix':f.controller.write_prefix(optional)
    else:
        with pytest.raises(MemoryError):optional.stream(0)
    optional.release()
    assert not f.capture.failure
    advance(f,202000)
    full=f.submitted.pop();assert len(full.stream(0))==25000;full.release()
    assert f.capture.stats()['incomplete_views']==0
    assert f.capture.stats()['optional_incomplete_views']==1


def test_delayed_optional_worker_stop_does_not_stall_full_crop_or_release_worker_lease(
        fixture,monkeypatch):
    import threading
    f=fixture;identifier=prepare(f)
    request,prefix_lease=f.headers.requests[identifier]
    workspace=f.capture.budget.reserve('test_worker_workspace',4096)
    entered=threading.Event();allow_stop=threading.Event();finished=threading.Event()
    timeouts=[]
    def close(*,timeout_s):
        timeouts.append(timeout_s);entered.set()
        if timeout_s>0:allow_stop.wait(10)
        if not allow_stop.is_set():return False
        for _,lease in f.headers.requests.values():lease.release()
        f.headers.requests.clear();workspace.release();return True
    monkeypatch.setattr(f.headers,'cancel',lambda identifier:True)
    monkeypatch.setattr(f.headers,'close',close)
    monkeypatch.setattr(f.headers,'poll',lambda:(_ for _ in ()).throw(RuntimeError('optional poll failed')))
    def run():
        advance(f,202000);finished.set()
    thread=threading.Thread(target=run);thread.start()
    try:
        assert entered.wait(.5)
        assert timeouts==[0],'live service turn requested a worker termination wait'
        assert finished.wait(5),'full crop did not progress while optional termination stayed blocked'
        assert not prefix_lease.released and not workspace._released
        assert not f.controller._entries,'closed controller stranded parent metadata'
        full=f.submitted.pop();assert full.role=='shared_full'
        assert len(full.stream(0))==25000;full.release()
        assert not f.capture.failure and timeouts==[0]
    finally:
        allow_stop.set();thread.join(2)
        f.controller.close()
    assert prefix_lease.released and workspace._released
    assert timeouts[-1]>0,'explicit shutdown must still wait after live initiation'


def test_close_during_prefix_render_keeps_parent_hold_until_writer_finishes(fixture,monkeypatch):
    import threading
    f=fixture;advance(f,71000);prefix=f.submitted.pop()
    entered=threading.Event();release=threading.Event();render=f.capture._render
    def paused(view,index):
        entered.set();assert release.wait(2);return render(view,index)
    monkeypatch.setattr(f.capture,'_render',paused)
    thread=threading.Thread(target=f.controller.write_prefix,args=(prefix,));thread.start()
    try:
        assert entered.wait(.5)
        f.controller.close()
        assert f.capture.stats()['budget']['by_kind'].get('header_hold',0)>0
        assert f.controller._entries,'writer still owns its entry and parent metadata'
    finally:
        release.set();thread.join(2);prefix.release()
    assert not thread.is_alive() and not f.controller._entries
    assert not f.capture.stats()['budget']['by_kind'].get('header_hold',0)
