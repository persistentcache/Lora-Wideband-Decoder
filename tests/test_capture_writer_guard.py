"""Real shared-timeline dispatch protects imminent early views without starving fulls."""
from types import SimpleNamespace

import numpy as np
import pytest

import detector
from capture_timeline import CaptureBudget, SharedCapture
from test_capture_timeline import reader


class Rig:
    def __init__(self, stall=30):
        self.now = 100.0
        self.history = reader(200000)
        self.sent = []
        self.capture = SharedCapture(rate=1000, center=915e6, hop_n=500,
            reader=self.history, decimator_factory=detector._StreamDecimator,
            submit=lambda view: self.sent.append(view) or True,
            budget=CaptureBudget(8*1024*1024, 32), threaded=False,
            writer_stall_s=stall, clock=lambda: self.now)

    def full(self, start, end):
        base_end = start + 500
        number = self.capture.register([dict(sf=7, bw=62.5, freq_hz=915e6,
            peak_power_db=20, detect_conf=.99)], base_start=start,
            base_end=base_end, end=end, anchor=(base_end,0,0), target_factor=8,
            export_bw=62.5, pre_hop_n=0, wb_n=500, sample_pos=base_end)
        assert number is not None
        return next(v for v in self.capture._views if v.number == number)

    def early(self, parent, end, *, role='early'):
        return self.capture.add_slice_view(parent, parent.refs[0], parent.start, end,
            role=role, request_id=len(self.capture._views))

    def advance(self, end):
        self.history._rp = self.history._wp = end
        self.capture.advance(self.history.history_anchor())
        self.pump()

    def pump(self):
        for _ in range(1000):
            if not self.capture.step():
                return
        pytest.fail('timeline did not quiesce')

    def release(self, view, seconds=0):
        assert view.complete
        out = view.stream(0)
        assert len(out) == (view.end-view.start+7)//8
        assert np.isfinite(out).all()
        self.now += seconds
        view.release()
        self.sent.remove(view)

    def train(self, seconds=3, *, queue_delay=0, failed=False):
        full = self.full(0,1000)
        if queue_delay:
            self.capture.submit = lambda view: False
        self.advance(2000)
        if queue_delay:
            self.now += queue_delay
            self.capture.submit = lambda view: self.sent.append(view) or True
            self.capture.step()
        assert self.sent == [full]
        if failed:
            self.now += seconds
            full._render_failed = True
            self.capture._fail('simulated failed full save')
            full.release(); self.sent.remove(full)
        else:
            self.release(full, seconds)
        return full

    def pair(self, early_end=8500, *, role='early'):
        waiting = self.full(4000,6000)
        parent = self.full(4000,20000)
        early = self.early(parent, early_end, role=role)
        return waiting, parent, early

    def finish(self):
        # At full EOF all original and optional spans are available. Drain every
        # accepted lease, including any still-held writer, through the real API.
        self.advance(25000)
        self.capture.close(wait=False)
        for _ in range(1000):
            if self.sent:
                self.release(self.sent[0])
            self.capture.step()
            if not self.capture._views and self.capture._writing is None:
                break
        assert self.capture.stats()['budget']['used_bytes'] == 0
        assert self.capture.stats()['budget']['outbox_slots'] == 0
        assert not self.capture.failure


def test_imminent_early_transfers_before_waiting_full_without_shortening_either():
    f=Rig(); f.train(3)
    waiting,parent,early=f.pair()
    f.advance(6500)
    assert not f.sent, 'full dispatched even though early needs only2.12s including FIR context'
    assert waiting.state=='ready' and early.state=='collecting'
    f.now+=1; f.advance(8700)
    assert f.sent==[early]
    f.release(early,.1); f.capture.step()
    assert f.sent==[waiting]
    assert parent.end==20000 and waiting.end==6000 and early.end==8500
    f.finish()


@pytest.mark.parametrize('case',['no_early','distant','prefix','no_estimate'])
def test_unjustified_guard_never_delays_a_full(case):
    f=Rig()
    if case!='no_estimate':f.train(3)
    waiting=f.full(4000,6000)
    if case!='no_early':
        parent=f.full(4000,20000)
        f.early(parent,13000 if case=='distant' else 8500,
                role='header_prefix' if case=='prefix' else 'early')
    f.advance(6500)
    assert f.sent==[waiting]
    f.finish()


def test_real_transfer_to_release_time_excludes_prior_ready_queue_wait():
    f=Rig(); f.train(2,queue_delay=10)
    waiting,_,_=f.pair(9000)  #2.62s away: longer than the2s service, less than12s age
    f.advance(6500)
    assert f.sent==[waiting], 'ready queue time was misclassified as writer service time'
    f.finish()


def test_both_fir_right_contexts_count_against_imminence():
    f=Rig(); f.train(2.1)
    waiting,_,_=f.pair(8501) #endpoint2.001s away, grid+bothFIR contexts2.124s
    f.advance(6500)
    assert f.sent==[waiting], 'guard ignored real FIR lookahead and endpoint grid'
    f.finish()


@pytest.mark.parametrize('cost,stall,cap',[(1.5,30,1.5),(8,40,5),(8,12,3)])
def test_stalled_future_dispatches_full_at_fixed_wait_cap(cost,stall,cap):
    f=Rig(stall); f.train(cost)
    waiting,_,_=f.pair(7000)
    f.advance(6500); started=f.now
    assert not f.sent
    f.now=started+cap-.001; f.capture.step()
    assert not f.sent
    f.now=started+cap; f.capture.step()
    assert f.sent==[waiting], 'stationary input must not extend the deferral deadline'
    f.finish()


def test_new_optional_arrivals_do_not_reset_expired_full_deadline():
    f=Rig(); f.train(3)
    waiting,parent,early=f.pair()
    f.advance(6500); started=f.now
    assert not f.sent
    f.now=started+2.7; f.advance(8700)
    assert f.sent==[early]
    f.release(early,.1)
    later=f.early(parent,8500)
    f.now=started+3; f.capture.step()
    assert f.sent==[waiting], 'new ready optional work reset/starved the waiting full'
    assert later.state=='ready'
    f.finish()


def test_two_optional_burst_bound_still_dispatches_full_before_guard_expiry():
    f=Rig(); f.train(3)
    waiting,parent,_=f.pair(10000)
    first=f.early(parent,6000); second=f.early(parent,6000)
    f.advance(6500)
    assert f.sent==[first]
    f.release(first); f.capture.step(); assert f.sent==[second]
    f.release(second); f.capture.step(); assert f.sent==[waiting]
    f.finish()


def test_near_stall_full_is_not_voluntarily_delayed():
    f=Rig(); f.train(3)
    waiting,parent,_=f.pair(13000)
    # Queue pressure ages the ready full before the guard can own a deferral.
    f.capture.submit=lambda view:False
    f.advance(6500)
    f.now+=25
    f.early(parent,8500)
    f.capture.submit=lambda view:f.sent.append(view) or True
    f.capture.step()
    assert f.sent==[waiting], 'guard consumed the remaining writer-stall/service margin'
    f.finish()


def test_optional_save_duration_does_not_replace_full_service_measurement():
    f=Rig(); f.train(2)
    parent=f.full(4000,20000)
    early=f.early(parent,6000)
    f.advance(6500); assert f.sent==[early]
    f.release(early,10)
    waiting=f.full(6500,7000)
    f.early(parent,10500)
    f.advance(7500)
    assert f.sent==[waiting], 'optional transfer duration became the full-save estimate'
    f.finish()


def test_eof_bypasses_deferral_and_incomplete_future_is_not_claimed_complete():
    f=Rig(); f.train(3)
    waiting,_,_=f.pair()
    f.advance(6500); assert not f.sent
    f.capture.close(wait=False); f.capture.step()
    assert f.sent==[waiting], 'EOF must not wait for a future early endpoint'
    assert 'input ended' in f.capture.failure
    waiting.release(); f.sent.remove(waiting); f.capture.step()
    assert f.capture.stats()['budget']['used_bytes']==0
    assert f.capture.stats()['budget']['outbox_slots']==0


def test_failure_cancels_guard_and_failed_full_duration_is_not_trusted():
    f=Rig(); f.train(3)
    waiting,_,_=f.pair(); f.advance(6500)
    assert not f.sent
    f.capture._fail('input lost'); f.capture.step(); f.capture.close(wait=False)
    assert not f.sent
    assert f.capture.stats()['budget']['used_bytes']==0
    broken=Rig(); broken.train(11,failed=True)
    assert broken.capture._full_write_s==0
    broken.capture.step(); broken.capture.close(wait=False)
    assert broken.capture.stats()['budget']['used_bytes']==0
