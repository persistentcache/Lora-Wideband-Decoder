"""Idle decoder children move between managers without overlapping pipe owners."""
import json,os,subprocess,threading,time
from pathlib import Path
import pytest
import detector
from detector import BackgroundDecoder


def wait_until(predicate,timeout=3):
    limit=time.monotonic()+timeout
    while not predicate() and time.monotonic()<limit:time.sleep(.005)
    assert predicate(), 'bounded fixture condition did not arrive'


@pytest.fixture
def instance(tmp_path,monkeypatch):
    monkeypatch.setenv('LORA_DECODE_WORKERS','2');monkeypatch.setenv('LORA_PKT_LOG','')
    monkeypatch.setenv('LORA_DECODE_RAWLOG',str(tmp_path/'raw.log'))
    monkeypatch.setenv('LORA_KEEP_IQ','1');monkeypatch.setenv('LORA_DECODE_WORKER_RSS_MB','0')
    monkeypatch.setenv('LORA_DECODE_BUDGET_S','10')
    with monkeypatch.context() as startup:
        startup.setattr(BackgroundDecoder,'_manager',lambda self:None)
        dec=BackgroundDecoder()
        for thread in dec._threads:thread.join()
    dec._threads=[];dec._worker_affinity=None;dec._pressure_n=0
    module=tmp_path/'decoder.py'
    module.write_text('''import os,time
from pathlib import Path
def process_file(path,**kwargs):
 p=Path(path)
 p.with_suffix('.started').write_text(str(os.getpid()))
 if 'crash' in p.name:os._exit(7)
 if 'block' in p.name:
  while not p.with_suffix('.release').exists():time.sleep(.005)
 print('fixture completed pid='+str(os.getpid()),flush=True)
''')
    dec._decoder_script=str(module)
    yield dec
    assert dec.abort(timeout=2)
    assert all(p.poll() is not None for p in dec._workers)


def submit(dec,tmp_path,name):
    path=tmp_path/f'SF7_125k_915.0000MHz_1000ksps_{name}.cf32';path.write_bytes(b'original samples')
    dec.submit(str(path),path.name,role='early');return path


def launch_one(dec,name):
    # One manager processes one queued job then a normal shutdown sentinel.
    dec._queue.put((float('inf'),next(dec._qseq),None))
    thread=threading.Thread(target=dec._manager,name=name);dec._threads.append(thread);thread.start()
    return thread


def test_sequential_jobs_reuse_actual_child_across_different_managers(instance,tmp_path):
    first=submit(instance,tmp_path,'first');a=launch_one(instance,'manager-A');a.join(3);assert not a.is_alive()
    pid=int(first.with_suffix('.started').read_text())
    second=submit(instance,tmp_path,'second');b=launch_one(instance,'manager-B');b.join(3);assert not b.is_alive()
    assert int(second.with_suffix('.started').read_text())==pid
    assert len(instance._workers)==1 and instance.pending()==0
    assert instance._queue.unfinished_tasks==0


def test_simultaneous_jobs_never_share_active_child(instance,tmp_path):
    first=submit(instance,tmp_path,'block-a');second=submit(instance,tmp_path,'block-b')
    a=threading.Thread(target=instance._manager);b=threading.Thread(target=instance._manager)
    instance._threads.extend((a,b));a.start();b.start()
    wait_until(lambda:first.with_suffix('.started').exists() and second.with_suffix('.started').exists())
    assert first.with_suffix('.started').read_text()!=second.with_suffix('.started').read_text()
    assert len(instance._workers)==2
    first.with_suffix('.release').touch();second.with_suffix('.release').touch()
    wait_until(lambda:instance.pending()==0)
    assert len(instance._idle_workers)==2


def test_dead_cached_worker_is_reaped_not_reused(instance,tmp_path):
    first=submit(instance,tmp_path,'first');thread=launch_one(instance,'A');thread.join(3)
    old=instance._workers[0];old.kill();old.wait(timeout=1)
    second=submit(instance,tmp_path,'second');thread=launch_one(instance,'B');thread.join(3)
    assert int(second.with_suffix('.started').read_text())!=old.pid
    assert old not in instance._idle_workers


def test_worker_crash_is_not_returned_to_idle_cache(instance,tmp_path):
    path=submit(instance,tmp_path,'crash');thread=launch_one(instance,'A');thread.join(3)
    assert not thread.is_alive();wait_until(lambda:instance._workers[0].poll() is not None)
    assert not instance._idle_workers
    assert instance.pending()==0 and instance._refs=={}


def test_rss_recycle_does_not_return_child(instance,tmp_path,monkeypatch):
    monkeypatch.setenv('LORA_DECODE_WORKER_RSS_MB','.0001')
    path=submit(instance,tmp_path,'first');thread=launch_one(instance,'A');thread.join(3)
    assert not thread.is_alive();assert instance._workers[0].poll() is not None
    assert not instance._idle_workers


def test_abort_with_active_and_idle_children_preserves_pending_iq(instance,tmp_path):
    first=submit(instance,tmp_path,'first');thread=launch_one(instance,'A');thread.join(3)
    busy=submit(instance,tmp_path,'block-busy');thread=threading.Thread(target=instance._manager);instance._threads.append(thread);thread.start()
    wait_until(lambda:busy.with_suffix('.started').exists())
    pending=submit(instance,tmp_path,'pending')
    assert instance.abort(timeout=1.5)
    assert busy.read_bytes()==pending.read_bytes()==b'original samples'
    assert not instance._idle_workers
    assert all(p.poll() is not None for p in instance._workers)


def test_idle_cache_is_lifo_and_double_return_does_not_duplicate(instance,tmp_path):
    a=instance._start_worker();b=instance._start_worker()
    instance._return_worker(a);instance._return_worker(b);instance._return_worker(b)
    assert instance._lease_worker() is b
    assert instance._lease_worker() is a
    assert not instance._idle_workers
    instance._return_worker(a);instance._return_worker(b)


def test_late_return_after_abort_cannot_resurrect_child(instance,tmp_path):
    worker=instance._start_worker()
    assert instance.abort(timeout=1)
    instance._return_worker(worker)
    assert not instance._idle_workers and worker.poll() is not None
    assert instance._lease_worker() is None


def test_normal_eof_drains_and_reaps_idle_children(instance,tmp_path):
    for name in ('one','two'):
        submit(instance,tmp_path,name);thread=launch_one(instance,name);thread.join(3)
    assert len(instance._workers)==1
    instance.drain(timeout=1)
    assert all(p.poll() is not None for p in instance._workers)
    assert not instance._idle_workers


def test_worker_ready_log_is_after_actual_child_import(instance,tmp_path):
    module=Path(instance._decoder_script)
    module.write_text('import time\ntime.sleep(.03)\ndef process_file(*args,**kw):pass\n')
    path=submit(instance,tmp_path,'ready');thread=launch_one(instance,'ready');thread.join(3)
    rows=(tmp_path/'raw.log').read_text().splitlines()
    start=next(r for r in rows if ' START ' in r);ready=next(r for r in rows if ' WORKER-READY ' in r)
    assert float(ready.split()[1])-float(start.split()[1])>=.025
    assert path.name in ready and 'pid=' in ready


def test_protocol_write_failure_reaps_child_and_never_caches(instance,tmp_path):
    worker=instance._start_worker();instance._return_worker(worker)
    original=worker.stdin
    class BrokenWrite:
        def __getattr__(self,name):return getattr(original,name)
        def write(self,*args):raise BrokenPipeError('injected protocol failure')
    worker.stdin=BrokenWrite()
    path=submit(instance,tmp_path,'broken');thread=launch_one(instance,'broken');thread.join(3)
    assert not thread.is_alive() and worker.poll() is not None
    assert not instance._idle_workers and instance.pending()==0
    assert path.read_bytes()==b'original samples'
    assert 'JOB-ERROR' in (tmp_path/'raw.log').read_text()


def test_real_silent_child_timeout_reaped_not_cached(instance,tmp_path,monkeypatch):
    worker=instance._start_worker();instance._return_worker(worker)
    clock=[1000.]
    def select_stub(*args):clock[0]+=args[-1];return [],[],[]
    import select
    monkeypatch.setattr(select,'select',select_stub)
    monkeypatch.setattr(detector.time,'time',lambda:clock[0])
    path=submit(instance,tmp_path,'block-timeout');thread=launch_one(instance,'timeout');thread.join(3)
    assert not thread.is_alive() and worker.poll() is not None
    assert not instance._idle_workers and instance.pending()==0
    assert 'DECODE TIMEOUT (90s wall)' in (tmp_path/'raw.log').read_text()


def test_abort_during_completed_job_return_never_recaches(instance,tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=instance._return_worker
    def delayed(worker):entered.set();assert release.wait(2);original(worker)
    monkeypatch.setattr(instance,'_return_worker',delayed)
    submit(instance,tmp_path,'completed');manager=launch_one(instance,'returning')
    assert entered.wait(2)
    results=[];aborter=threading.Thread(target=lambda:results.append(instance.abort(timeout=1.5)))
    aborter.start();assert instance._abort_event.wait(1);release.set();aborter.join(2);manager.join(2)
    assert results==[True] and not instance._idle_workers
    assert all(p.poll() is not None for p in instance._workers)


def test_abort_racing_lazy_lease_spawn_reaps_new_child(instance,tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=detector.subprocess.Popen
    def delayed(*args,**kwargs):entered.set();assert release.wait(2);return original(*args,**kwargs)
    monkeypatch.setattr(detector.subprocess,'Popen',delayed)
    leased=[];aborted=[]
    thread=threading.Thread(target=lambda:leased.append(instance._lease_worker()));thread.start();assert entered.wait(2)
    aborter=threading.Thread(target=lambda:aborted.append(instance.abort(timeout=1.5)));aborter.start()
    assert instance._abort_event.wait(1);release.set();thread.join(2);aborter.join(2)
    assert leased==[None] and aborted==[True]
    assert len(instance._workers)==1 and instance._workers[0].poll() is not None
    assert not instance._idle_workers
