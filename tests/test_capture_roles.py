"""Shared/early decode roles cannot declare an entire RF window covered."""
import subprocess
import sys
import pytest
from detector import BackgroundDecoder


@pytest.fixture
def dec(monkeypatch):
    monkeypatch.setenv('LORA_DECODE_WORKERS','1')
    monkeypatch.setenv('LORA_DECODE_RAWLOG','')
    monkeypatch.setenv('LORA_PKT_LOG','')
    monkeypatch.delenv('LORA_KEEP_IQ',raising=False)
    monkeypatch.delenv('LORA_KEEP_CAPTURES_DIR',raising=False)
    with monkeypatch.context() as startup:
        startup.setattr(BackgroundDecoder,'_manager',lambda self:None)
        decoder=BackgroundDecoder()
        for thread in decoder._threads: thread.join()
    return decoder


def mesh(identifier):
    return ('=== RESULT ===\nCRC: OK\n--- Meshtastic Packet ---\n'
            f'PacketID: {identifier} Flags: 0x00 (hop_limit=0 hop_start=0 hops_taken=0)\n\n')


@pytest.mark.parametrize('role',['early','shared_full','shared_partial','continuation'])
def test_role_never_registers_waits_or_marks_whole_window(dec,tmp_path,role):
    bucket=(915e6,100.,12,62500)
    legacy=tmp_path/'legacy.cf32'; legacy.write_bytes(b'iq')
    dec.submit(str(legacy),legacy.name,bucket_key=bucket)
    dec._mark_bucket_inflight(str(legacy))
    dec._mark_bucket_decoded(legacy.name)
    path=tmp_path/f'{role}.cf32'; path.write_bytes(b'iq')
    dec.submit(str(path),path.name,bucket_key=bucket,role=role)
    assert not dec._bucket_already_decoded(str(path))
    assert not dec._is_bucket_inflight(str(path))
    before=list(dec._decoded_buckets)
    # Explicit role also protects compact output from basename alias matches.
    dec._compact(mesh('0x12345678'),legacy.name,role=role)
    assert dec._decoded_buckets==before
    dec._clear_bucket_inflight(str(path))
    assert dec._is_bucket_inflight(str(legacy))


def test_default_bucket_admission_remains_unchanged(dec,tmp_path):
    path=tmp_path/'default.cf32'; path.write_bytes(b'iq')
    dec.submit(str(path),path.name,bucket_key=(915e6,100.,12,62500))
    dec._compact(mesh('0x12345678'),path.name)
    assert dec._bucket_already_decoded(str(path))


def test_unknown_or_conflicting_role_refused_before_refcount(dec,tmp_path):
    path=tmp_path/'capture.cf32'; path.write_bytes(b'iq')
    with pytest.raises(ValueError): dec.submit(str(path),path.name,role='typo')
    assert not dec._refs and dec.pending()==0
    dec.submit(str(path),path.name,role='early')
    with pytest.raises(ValueError): dec.submit(str(path),path.name,role='default')
    assert dec._refs[str(path)]==1 and dec.pending()==1


class StopManager(BaseException):pass


@pytest.mark.parametrize('early',['fail','success','budget'])
def test_real_manager_failed_or_successful_early_cannot_suppress_full_second_frame(
        dec,tmp_path,monkeypatch,early):
    # Real wire, manager publication, compact admission, retry/refcount/cleanup.
    code='''
import sys,json
for line in sys.stdin:
    path,mode,budget=line.rstrip('\\n').split('\\t')
    first='early.cf32' in path
    behavior=sys.argv[1]
    if first and behavior=='budget' and not budget:
        print('[BUDGET] exhausted',flush=True)
    elif not first or behavior=='success':
        identifier='0x11111111' if first else '0x22222222'
        print('[PKT] '+json.dumps({'proto':'meshtastic','pkt_id':identifier,'confidence':'verified'}),flush=True)
        print('=== RESULT ===\\nCRC: OK\\n--- Meshtastic Packet ---',flush=True)
        print('PacketID: '+identifier+' Flags: 0x00 (hop_limit=0 hop_start=0 hops_taken=0)\\n',flush=True)
    print('__END__',flush=True)
'''
    processes=[]
    def start():
        p=subprocess.Popen([sys.executable,'-u','-c',code,early],stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,text=True)
        processes.append(p);return p
    monkeypatch.setattr(dec,'_start_worker',start)
    dec._pressure_n=0
    raw=[]
    monkeypatch.setattr(dec,'_rawlog',lambda *args:raw.append(args))
    bucket=(915e6,100.,12,62500)
    paths=[]
    for role in ('early','shared_full','shared_partial','continuation'):
        path=tmp_path/f'{role}.cf32';path.write_bytes(b'iq');paths.append(path)
        dec.submit(str(path),path.name,bucket_key=bucket,role=role)
    original_get=dec._queue.get
    def get(*args,**kwargs):
        if dec._queue.empty():raise StopManager()
        return original_get(*args,**kwargs)
    monkeypatch.setattr(dec._queue,'get',get)
    try:
        with pytest.raises(StopManager):dec._manager()
        assert ('meshtastic','0x22222222') in dec._decoded_keys
        assert dec.decoded_counts()[1]==(2 if early=='success' else 1)
        assert not dec._decoded_buckets and not dec._fname_bucket
        assert not any(row[0]=='SKIP-BUCKET' for row in raw)
        assert dec.pending()==0 and not dec._refs
        assert not dec._capture_roles
        assert all(not path.exists() for path in paths)
        if early=='budget':assert any(row[0]=='REQUEUE-SLOW' for row in raw)
    finally:
        for process in processes:
            if process.poll() is None:process.stdin.close()
            process.wait(timeout=3)
            process.stdout.close()
