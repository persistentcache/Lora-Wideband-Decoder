"""Failure-only cleanup: real queues/processes and real main exception handlers."""
import ast, copy, json, os, subprocess, sys, textwrap, time
from pathlib import Path
import pytest
import detector

SRC=Path(detector.__file__).parent

@pytest.mark.parametrize('resist,signal_error',[(False,False),(True,False),(True,True)])
def test_real_pool_abort_reaps_stalled_consumer_and_cancels_full_feeder(resist,signal_error,tmp_path):
    code=r'''
import os,signal,sys,time,json
import numpy as np
import detect_pool
ready=detect_pool.mp.get_context('fork').Event()
def stalled(*args):
 if RESIST:signal.signal(signal.SIGTERM,signal.SIG_IGN)
 ready.set();time.sleep(30)
detect_pool._worker_main=stalled
pool=detect_pool.DetectPool(1,13,16,{})
assert ready.wait(2)
names=[s.name for s in pool._shms]
for seq in range(13):pool.dispatch(seq,seq,16,np.zeros(4096,np.float32),[(2048.,30.,20.,0.)])
time.sleep(.05)
started=time.monotonic()
if SIGNAL_ERROR:
 def failed_term():raise OSError('injected first signal failure')
 pool._workers[0].terminate=failed_term
try:
 assert pool.abort(timeout=1.5) is (not SIGNAL_ERROR), 'cleanup result not truthful'
 assert time.monotonic()-started<2.5
 assert all(not p.is_alive() and p.exitcode is not None for p in pool._workers)
 assert not any(os.path.exists('/dev/shm/'+n) for n in names)
 assert pool.abort(timeout=.2), 'abort must be idempotent'
 print(json.dumps({'elapsed':time.monotonic()-started,'reaped':True}))
finally:
 # Test-owned safety cleanup also makes a missing implementation fail promptly.
 for q in (pool._task_q,pool._result_q):q.cancel_join_thread();q.close()
 for p in pool._workers:
  if p.is_alive():p.kill()
  p.join(timeout=1)
 for s in pool._shms:
  try:s.close();s.unlink()
  except FileNotFoundError:pass
'''.replace('RESIST',repr(resist)).replace('SIGNAL_ERROR',repr(signal_error))
    env={**os.environ,'PYTHONPATH':str(SRC),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    run=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=env,timeout=7)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout.splitlines()[-1])['reaped']


def main_exception_path():
    tree=ast.parse(Path(detector.__file__).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    index,node=next((i,n) for i,n in enumerate(main.body) if isinstance(n,ast.Try)
        and any(isinstance(h.type,ast.Name) and h.type.id=='KeyboardInterrupt' for h in n.handlers))
    node=copy.deepcopy(node)
    loop=next(n for n in node.body if isinstance(n,ast.While))
    # Execute the production failure check within the production try/handlers;
    # only downstream acquisition DSP is removed. Normal branch exits the loop.
    loop.body=[loop.body[0],ast.Break()]
    node.body=[loop]
    footer=[]
    for n in main.body[index+1:]:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='total_drops' for t in n.targets):break
        footer.append(copy.deepcopy(n))
    return compile(ast.fix_missing_locations(ast.Module([node]+footer,type_ignores=[])),str(SRC/'detector.py'),'exec')


def test_main_real_failure_handler_aborts_without_new_work_and_reraises_original():
    calls=[];problem=RuntimeError('actual raw history failure')
    class Pool:
        def abort(self,**kwargs):calls.append('pool-abort');return True
        def close(self):calls.append('pool-close')
    class Decoder:
        def abort(self,**kwargs):calls.append('decoder-abort');return True
        def drain_slow_into_fast(self):calls.append('decode-new-work')
    class Recorder:
        def raise_shared_failure(self):raise problem
        def flush_pending(self,**kwargs):calls.append('flush-new-work')
        _shared=None
    env={**vars(detector),'_pool':Pool(),'decoder':Decoder(),'recorder':Recorder(),
         '_inflight':[1],'_commit_oldest':lambda:calls.append('commit-new-work')}
    with pytest.raises(RuntimeError) as error:exec(main_exception_path(),env)
    assert error.value is problem
    assert sorted(calls)==['decoder-abort','pool-abort']


def test_main_real_normal_eof_still_uses_normal_drain():
    calls=[]
    class Pool:
        def abort(self,**kwargs):calls.append('abort')
        def close(self):calls.append('pool-close')
    class Decoder:
        def drain_slow_into_fast(self):calls.append('decode-drain')
        def pending(self):return 0
    class Recorder:
        _save_queue=type('Q',(),{'qsize':lambda self:0})()
        def raise_shared_failure(self):pass
        def flush_pending(self,**kwargs):calls.append('capture-finalize')
        def flush(self):calls.append('capture-flush')
    env={**vars(detector),'_pool':Pool(),'decoder':Decoder(),'recorder':Recorder(),
         '_inflight':[],'_commit_oldest':lambda:calls.append('commit')}
    exec(main_exception_path(),env)
    assert calls==['pool-close','capture-finalize','capture-flush','decode-drain']


@pytest.mark.parametrize('signal_error',[False,True])
def test_decoder_abort_stops_active_worker_wakes_managers_and_preserves_files(tmp_path,monkeypatch,signal_error):
    monkeypatch.setenv('LORA_DECODE_WORKERS','1')
    monkeypatch.setenv('LORA_PKT_LOG','')
    monkeypatch.setenv('LORA_DECODE_RAWLOG',str(tmp_path/'jobs.log'))
    monkeypatch.delenv('LORA_KEEP_IQ',raising=False)
    module=tmp_path/'decoder.py';marker=tmp_path/'started'
    module.write_text('import os,signal,time\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        'def process_file(*args,**kwargs):\n'
        f' Path({str(marker)!r}).write_text(str(os.getpid()))\n time.sleep(30)\n')
    decoder=detector.BackgroundDecoder()
    decoder._decoder_script=str(module);decoder._worker_affinity=None
    files=[tmp_path/f'SF7_125.00k_915.0000MHz_1000ksps_pwr20_{i:03}.cf32' for i in range(2)]
    for f in files:f.write_bytes(b'retained forensic input')
    try:
        for f in files:decoder.submit(str(f),f.name)
        limit=time.monotonic()+3
        while not marker.exists() and time.monotonic()<limit:time.sleep(.01)
        assert marker.exists(),'no-DSP worker did not start'
        if signal_error:
            def failed_term():raise OSError('injected first signal failure')
            decoder._workers[0].terminate=failed_term
        started=time.monotonic()
        assert decoder.abort(timeout=1.5) is (not signal_error)
        assert time.monotonic()-started<2.5
        assert all(p.poll() is not None for p in decoder._workers)
        assert all(not t.is_alive() for t in decoder._threads)
        count=len(decoder._workers)
        assert decoder._start_worker() is None
        decoder.submit(str(files[1]),files[1].name)
        assert len(decoder._workers)==count
        assert all(f.read_bytes()==b'retained forensic input' for f in files)
        assert decoder.abort(timeout=.2)
    finally:
        # RED safety cleanup; only fixture-owned subprocess handles.
        for p in decoder._workers:
            if p.poll() is None:p.kill()
            p.wait(timeout=1)


def test_decoder_abort_waits_for_in_progress_spawn_registration(tmp_path,monkeypatch):
    import threading
    monkeypatch.setenv('LORA_DECODE_WORKERS','1');monkeypatch.setenv('LORA_PKT_LOG','')
    decoder=detector.BackgroundDecoder();decoder._worker_affinity=None
    module=tmp_path/'decoder.py';module.write_text('def process_file(*args,**kwargs):pass\n');decoder._decoder_script=str(module)
    real=detector.subprocess.Popen;entered=threading.Event();release=threading.Event();spawned=[];aborted=[]
    def delayed(*a,**kw):
        entered.set();assert release.wait(2)
        return real(*a,**kw)
    monkeypatch.setattr(detector.subprocess,'Popen',delayed)
    spawn=threading.Thread(target=lambda:spawned.append(decoder._start_worker()))
    abort=threading.Thread(target=lambda:aborted.append(decoder.abort(timeout=1.5)))
    try:
        spawn.start();assert entered.wait(2);abort.start()
        assert decoder._abort_event.wait(1)
        release.set();spawn.join(3);abort.join(3)
        assert not spawn.is_alive() and not abort.is_alive()
        assert aborted==[True] and len(spawned)==1 and spawned[0] is not None
        assert spawned[0].poll() is not None
        assert len(decoder._workers)==1 and decoder._start_worker() is None
    finally:
        release.set()
        for p in decoder._workers:
            if p.poll() is None:p.kill()
            p.wait(timeout=1)


def test_main_failure_during_eof_flush_also_aborts_and_preserves_original():
    calls=[];problem=OSError('writer failed at EOF')
    class Pool:
        def close(self):calls.append('pool-close')
        def abort(self,**kwargs):calls.append('pool-abort');return True
    class Decoder:
        def abort(self,**kwargs):calls.append('decoder-abort');return True
        def drain_slow_into_fast(self):calls.append('unexpected-new-work')
    class Recorder:
        _shared=None
        _save_queue=type('Q',(),{'qsize':lambda self:0})()
        def raise_shared_failure(self):pass
        def flush_pending(self,**kwargs):pass
        def flush(self):raise problem
    env={**vars(detector),'_pool':Pool(),'decoder':Decoder(),'recorder':Recorder(),'_inflight':[]}
    with pytest.raises(OSError) as error:exec(main_exception_path(),env)
    assert error.value is problem
    assert calls==['pool-close','decoder-abort','pool-abort']


def test_cleanup_failure_never_masks_original_or_joins_capture_queues():
    calls=[];problem=RuntimeError('raw ring lapped')
    class Shared:
        def close(self,**kw):assert kw=={'wait':False};calls.append('shared-close')
    class Recorder:
        _shared=Shared()
        def raise_shared_failure(self):raise problem
    class Owner:
        def abort(self,**kwargs):calls.append('abort-error');raise OSError('reap failure')
    env={**vars(detector),'_pool':Owner(),'decoder':Owner(),'recorder':Recorder(),'_inflight':[]}
    with pytest.raises(RuntimeError) as error:exec(main_exception_path(),env)
    assert error.value is problem
    assert calls==['abort-error','abort-error','shared-close']


def test_fatal_helper_waits_header_child_but_never_capture_queues():
    calls=[]
    class Header:
        def close(self,*,wait):assert wait is True;calls.append('header-reap')
    class Shared:
        header_controller=Header()
        def close(self,*,wait):assert wait is False;calls.append('shared-close-no-wait')
    class Recorder:_shared=Shared()
    detector._abort_failed_pipeline(None,None,Recorder())
    assert calls==['header-reap','shared-close-no-wait']
