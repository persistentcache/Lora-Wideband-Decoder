"""Direct measured-drift recovery precedes sweep without changing failure votes."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import decoder


def production_rescue():
    tree=ast.parse(Path(decoder.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_decode_attempt')
    body=[]
    for node in fn.body:
        if not isinstance(node,ast.If):continue
        text=ast.unparse(node)
        # Keep the production direct trial intact. Its two existing later
        # fallback stages retain real eligibility but use observable sinks.
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='_dd_resid_curve' for n in ast.walk(node)):
            if '_retry_bases' in text:
                n=copy.deepcopy(node);n.body=ast.parse('late_drift()').body;body.append(n)
            elif 'sym_time_ms' in ast.unparse(node.test):body.append(copy.deepcopy(node))
        elif '=== SWEEP (timing' in text:
            n=copy.deepcopy(node);n.body=ast.parse('timing_sweep()').body;body.append(n)
    return compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),decoder.__file__,'exec')


def run_trial(*,crc_match=True,plausible=True,curve=(0.,.2),deadline=None,clock=0.,short=False,**changes):
    calls=[];original_raw=[11]*99;original_payload=original_raw[:97];original_soft=[('original',)]
    vote=['original_vote'];trial_raw=[23]*(98 if short else 99);trial_payload=trial_raw[:97];trial_soft=[('trial',)]
    def track(ds,slope,tau):calls.append(('dd',ds,slope,tau));return list(curve)
    def demod(ds,slope,*,frac_tau,resid_curve):
        calls.append(('demod',ds,slope,frac_tau,tuple(resid_curve)))
        return trial_raw,trial_payload,trial_soft,[7]*200
    def crc(raw,length,strict=False):calls.append(('crc',strict,length));return crc_match,'xor-CCITT-LE'
    def shape(raw):calls.append(('plausibility',));return plausible
    env=dict(crc_present=True,crc_ok=False,hdr_ok=True,_fluke_junk=False,sym_time_ms=65.536,
        deadline=deadline,time=SimpleNamespace(process_time=lambda:clock),tau_frac=.125,
        payload_len=97,_clean_crc=False,crc_method='',raw_bytes=original_raw,payload=original_payload,
        pay_soft_info=original_soft,_skip_sweep=False,_c2_skip=False,_sc_att=vote,
        _dd_resid_curve=track,soft_decode_payload=demod,check_crc=crc,
        _is_chase_acceptable_uni=shape,timing_sweep=lambda:calls.append(('sweep',)),
        late_drift=lambda:calls.append(('late',)),print=lambda *a,**kw:None)
    env.update(changes);exec(production_rescue(),env)
    return SimpleNamespace(env=env,calls=calls,raw=original_raw,payload=original_payload,
        soft=original_soft,votes=vote,trial=trial_raw)


def test_direct_received_crc_wins_before_sweep_and_late_fallback():
    result=run_trial()
    assert [c[0] for c in result.calls]==['dd','demod','crc','plausibility']
    assert result.calls[2]==('crc',True,97)
    assert result.env['crc_ok'] and result.env['_clean_crc']
    assert result.env['raw_bytes'] is result.trial


@pytest.mark.parametrize('changes',[{'crc_match':False},{'plausible':False},{'short':True}])
def test_failed_direct_trial_preserves_original_payload_votes_and_fallback(changes):
    result=run_trial(**changes)
    assert result.calls[0][0]=='dd' and result.calls[-2:]==[('sweep',),('late',)]
    assert result.env['raw_bytes'] is result.raw
    assert result.env['payload'] is result.payload
    assert result.env['pay_soft_info'] is result.soft
    assert result.env['_sc_att'] is result.votes and result.votes==['original_vote']
    assert not result.env['crc_ok'] and not result.env['_clean_crc']
    assert result.env['tau_frac']==.125


@pytest.mark.parametrize('changes',[{'hdr_ok':False},{'crc_present':False},{'crc_ok':True},
    {'_fluke_junk':True},{'sym_time_ms':16.0},{'deadline':0.,'clock':0.},
    {'deadline':1.,'clock':2.}])
def test_ineligible_or_expired_attempt_never_gets_new_trial(changes):
    result=run_trial(**changes)
    assert not any(c[0] in ('dd','demod','crc','plausibility') for c in result.calls)


@pytest.mark.parametrize('curve',[(),(1.,1.119),(float('nan'),float('nan'))])
def test_unmeasured_or_small_curve_does_not_add_demod_or_votes(curve):
    result=run_trial(curve=curve)
    assert result.calls==[('dd',0,0.,.125),('sweep',),('late',)]
    assert result.env['raw_bytes'] is result.raw and result.votes==['original_vote']


def test_exact_existing_curve_threshold_is_eligible():
    assert run_trial(curve=(0.,.12)).env['crc_ok']


def test_deadline_after_tracker_prevents_new_payload_demod():
    clock=iter((0.,2.))
    result=run_trial(deadline=1.,time=SimpleNamespace(process_time=lambda:next(clock)))
    assert result.calls==[('dd',0,0.,.125),('sweep',),('late',)]
