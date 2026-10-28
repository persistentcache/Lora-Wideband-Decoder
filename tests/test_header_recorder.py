"""Header opt-in, save-worker bypass and fallback keep-N isolation."""
import ast
from pathlib import Path
import threading
import pytest

from test_shared_recorder import rec
from test_header_capture import fixture as capture_fixture, advance, prepare


def worker(rec):
    import detector
    cls=next(n for n in ast.parse(Path(detector.__file__).read_text()).body
             if isinstance(n,ast.ClassDef) and n.name=='SignalRecorder')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_save_worker')
    env=dict(vars(detector));env['_nb_plateau_offset']=lambda *args,**kw:None
    exec(compile(ast.Module([method],type_ignores=[]),detector.__file__,'exec'),env)
    rec._save_queue.put(None)
    thread=threading.Thread(target=env['_save_worker'],args=(rec,))
    thread.start();thread.join(3)
    assert not thread.is_alive()


def test_header_flag_requires_shared_path(rec,monkeypatch):
    from test_capture_timeline import reader
    monkeypatch.setenv('LORA_HEADER_CHECKPOINT','1')
    monkeypatch.delenv('LORA_SHARED_CAPTURE')
    with pytest.raises(RuntimeError,match='SHARED_CAPTURE'):rec.set_history_reader(reader())
    assert rec._shared is None


def test_header_workspace_requires_explicit_aggregate_budget_before_start(rec,monkeypatch):
    from test_capture_timeline import reader
    monkeypatch.setenv('LORA_HEADER_CHECKPOINT','1')
    monkeypatch.setenv('LORA_SHARED_CAPTURE_MB','2048')
    with pytest.raises(RuntimeError,match='memory|budget'):rec.set_history_reader(reader())
    assert rec._shared is None


@pytest.mark.parametrize('startup_fails',[False,True])
def test_header_workspace_stays_reserved_until_service_stops(rec,monkeypatch,startup_fails):
    import header_checkpoint_service
    from test_capture_timeline import reader
    monkeypatch.setenv('LORA_HEADER_CHECKPOINT','1')
    monkeypatch.setenv('LORA_SHARED_CAPTURE_MB','3072')
    held=[]
    class Service:
        def __init__(self,*,workspace_lease,**kwargs):
            held.append(workspace_lease)
            if startup_fails:raise RuntimeError('launch failed')
        def poll(self):return None
        def close(self,**kwargs):held[0].release();return True
    monkeypatch.setattr(header_checkpoint_service,'HeaderCheckpointService',Service)
    if startup_fails:
        with pytest.raises(RuntimeError,match='launch failed'):rec.set_history_reader(reader())
    else:
        rec.set_history_reader(reader())
        assert rec._shared.stats()['budget']['by_kind']['header_worker_workspace']==640*1024**2
        rec._shared.close(wait=False)
    assert held and rec._shared.stats()['budget']['used_bytes']==0


def test_unexpected_prefix_writer_failure_keeps_full_capture(rec,capture_fixture,monkeypatch):
    f=capture_fixture;advance(f,71000);prefix=f.submitted.pop()
    rec._header=f.controller
    def failure(view):raise OSError('prefix cleanup failed')
    monkeypatch.setattr(f.controller,'write_prefix',failure)
    assert rec._queue_shared_save(prefix)
    worker(rec)
    assert not f.capture.failure
    advance(f,202000)
    full=f.submitted.pop();assert full.role=='shared_full'
    assert len(full.stream(0))==25000
    full.release()


def test_prefix_save_path_never_touches_packet_or_keep_registry(rec,capture_fixture):
    f=capture_fixture;advance(f,71000);prefix=f.submitted.pop()
    rec._header=f.controller
    assert rec._queue_shared_save(prefix)
    worker(rec)
    assert len(f.headers.requests)==1 and prefix._released
    assert rec._saved_abs_t==[] and rec._img_carriers==[]
    assert not list(Path(rec.export_dir).glob('SF*.cf32'))


def test_early_save_bypasses_keep_n_reads_and_writes(rec,capture_fixture):
    f=capture_fixture;identifier=prepare(f)
    f.headers.complete(identifier,endpoint=4000);f.capture.step()
    early=f.submitted.pop()
    rec._header=f.controller
    # Full fallback already owns both keep-N entries; optional early neither
    # consumes another nor gets discarded by these whole-span entries.
    entries=[(915e6,24.,7,125.,0.,None)]*2
    rec._saved_abs_t=list(entries)
    assert rec._queue_shared_save(early)
    worker(rec)
    files=list(Path(rec.export_dir).glob('SF*.cf32'))
    assert len(files)==1 and '_early_' in files[0].name
    assert rec._saved_abs_t==entries


def test_early_then_full_both_save_when_keep_one_is_configured(rec,capture_fixture):
    f=capture_fixture;identifier=prepare(f)
    f.headers.complete(identifier,endpoint=4000);f.capture.step()
    early=f.submitted.pop();rec._header=f.controller;rec._dedup_keep=1
    rec._queue_shared_save(early);worker(rec)
    assert rec._saved_abs_t==[]
    advance(f,202000);full=f.submitted.pop()
    rec._queue_shared_save(full);worker(rec)
    files=list(Path(rec.export_dir).glob('SF*.cf32'))
    assert len(files)==2 and any('_shared_full_' in p.name for p in files)
    assert len(rec._saved_abs_t)==1
