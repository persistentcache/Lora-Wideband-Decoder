"""Bounded header worker ownership and real process lifecycle."""
import importlib.util


def test_header_checkpoint_service_is_available():
    assert importlib.util.find_spec("header_checkpoint_service") is not None, (
        "metadata-only bounded header worker service is missing")

import os
import time
from pathlib import Path
import numpy as np
import pytest


class Lease:
    def __init__(self):
        self.releases = 0
    def release(self):
        self.releases += 1




@pytest.fixture
def api():
    import header_checkpoint_service as api
    return api


@pytest.fixture
def request_factory(api, tmp_path):
    path = tmp_path / 'prefix.cf32'
    np.ones(4096, np.complex64).tofile(path)
    def make(number=1, **kwargs):
        values = dict(request_id=number, path=str(path), offset_samples=0,
                      sample_count=4096, file_identity=api.file_identity(path.stat()),
                      sf=12, bw=62500., nb_fs=500000., mix_center_hz=915e6,
                      identity_token=('stream', 3, 4, 100000, 20),
                      cpu_budget_s=.2, wall_timeout_s=2, name='normal')
        values.update(kwargs)
        return api.HeaderRequest(**values)
    return make


@pytest.fixture
def service(api, monkeypatch):
    def command(fd):
        import sys
        code = """
import sys,os,time
sys.path.insert(0,sys.argv[2])
from multiprocessing.connection import Connection
conn=Connection(int(sys.argv[1]))
while True:
    request=conn.recv()
    if request is None:break
    if request.name=='stall':time.sleep(10)
    if request.name=='crash':os._exit(17)
    conn.send((request.request_id,'ok',{'identity_token':request.identity_token,
               'pid':os.getpid()},None,.01))
"""
        return [sys.executable,'-u','-c',code,str(fd),str(Path(api.__file__).parent)]
    monkeypatch.setattr(api, '_worker_command', command)
    services = []
    def make(**kwargs):
        kwargs.setdefault('workspace_lease', Lease())
        item = api.HeaderCheckpointService(**kwargs)
        services.append(item)
        return item
    yield make
    for item in services:
        assert item.close(timeout_s=2)


def wait_result(service, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = service.poll()
        if result is not None:
            return result
        time.sleep(.005)
    pytest.fail('header result was not delivered within bounded test time')


def test_persistent_worker_and_lease_transfer(service, request_factory):
    s = service(max_pending=2)
    a, b = Lease(), Lease()
    assert s.submit(request_factory(1), a)
    first = wait_result(s)
    assert first.status == 'ok' and first.request.identity_token == ('stream', 3, 4, 100000, 20)
    assert a.releases == 1
    assert s.submit(request_factory(2), b)
    second = wait_result(s)
    assert first.metadata['pid'] == second.metadata['pid']
    assert b.releases == 1 and s.pending == 0


def test_uncollected_results_hold_descriptor_capacity_but_not_iq_lease(service, request_factory):
    s = service(max_pending=1)
    a, refused = Lease(), Lease()
    assert s.submit(request_factory(1), a)
    deadline = time.monotonic() + 2
    while not a.releases and time.monotonic() < deadline:
        time.sleep(.005)
    assert a.releases == 1
    assert not s.submit(request_factory(2), refused)
    assert refused.releases == 0 and s.pending == 1
    assert wait_result(s).status == 'ok'
    assert s.submit(request_factory(2), refused)
    assert wait_result(s).status == 'ok' and refused.releases == 1


@pytest.mark.parametrize('mode,status', [('stall', 'timeout'), ('crash', 'error')])
def test_timeout_or_crash_releases_once_and_restarts(service, request_factory, mode, status):
    workspace = Lease()
    s = service(max_pending=2, workspace_lease=workspace)
    a, b = Lease(), Lease()
    assert s.submit(request_factory(1, name=mode, wall_timeout_s=.08 if mode == 'stall' else 2), a)
    assert s.submit(request_factory(2), b)
    failed, good = wait_result(s), wait_result(s)
    assert failed.status == status and good.status == 'ok'
    assert a.releases == b.releases == 1 and s.pending == 0
    assert workspace.releases == 0
    assert s.close(timeout_s=1) and workspace.releases == 1


def test_cancel_active_and_queued_never_releases_live_mapping_early(service, request_factory):
    s = service(max_pending=2)
    a, b = Lease(), Lease()
    assert s.submit(request_factory(1, name='stall', wall_timeout_s=1), a)
    assert s.submit(request_factory(2), b)
    deadline = time.monotonic() + 2
    while s.active_request_id != 1 and time.monotonic() < deadline:
        time.sleep(.005)
    assert s.active_request_id == 1 and a.releases == 0
    assert s.cancel(1) and s.cancel(2)
    results = [wait_result(s), wait_result(s)]
    assert {r.status for r in results} == {'cancelled'}
    assert a.releases == b.releases == 1
    assert not s.cancel(1)


def test_close_is_bounded_releases_all_and_rejects_new_work(service, request_factory):
    s = service(max_pending=2)
    a, b, c = Lease(), Lease(), Lease()
    assert s.submit(request_factory(1, name='stall', wall_timeout_s=20), a)
    assert s.submit(request_factory(2), b)
    started = time.monotonic()
    assert s.close(timeout_s=1)
    assert time.monotonic() - started < 1.2
    assert a.releases == b.releases == 1
    assert not s.submit(request_factory(3), c) and c.releases == 0
    assert s.close(timeout_s=.1)
    assert a.releases == b.releases == 1


@pytest.mark.parametrize('change', [dict(sample_count=1000000000), dict(offset_samples=-1),
    dict(cpu_budget_s=float('nan')), dict(wall_timeout_s=0), dict(sf=6),
    dict(nb_fs=0), dict(mix_center_hz=float('nan')), dict(identity_token=['mutable'])])
def test_invalid_or_oversized_metadata_refused_without_lease_transfer(service, request_factory, change):
    s = service(max_prefix_bytes=40000)
    lease = Lease()
    assert not s.submit(request_factory(**change), lease)
    assert lease.releases == 0 and s.pending == 0


def test_worker_uses_readonly_actual_sample_slice_and_child_cpu_duration(api, request_factory, monkeypatch):
    import decoder
    path = Path(request_factory().path)
    iq = (np.arange(4096) + 2j*np.arange(4096)).astype(np.complex64)
    iq.tofile(path)
    original = path.read_bytes()
    calls = []
    def probe(prefix, sf, bw, fs, name, **kwargs):
        calls.append((prefix.copy(), prefix.flags.writeable, sf, bw, fs, kwargs))
        return {'ready_end_bw': 12345, 'bw': bw, 'identity_token': kwargs['identity_token']}
    monkeypatch.setattr(decoder, 'probe_frame_header', probe)
    req = request_factory(offset_samples=7, sample_count=1000)
    status, result, error = api._probe_request(req)
    assert status == 'ok' and error is None and result['ready_end_bw'] == 12345
    prefix, writable, sf, bw, fs, kwargs = calls[0]
    np.testing.assert_array_equal(prefix, iq[7:1007])
    assert not writable and (sf,bw,fs) == (12,62500,500000)
    assert 0 < kwargs['budget_s'] <= req.cpu_budget_s
    assert kwargs['identity_token'] == req.identity_token
    assert path.read_bytes() == original


def test_changed_identity_or_file_length_abstains_before_decoder(api, request_factory, monkeypatch):
    import decoder
    req = request_factory()
    monkeypatch.setattr(decoder, 'probe_frame_header', lambda *a, **k: pytest.fail('stale input'))
    Path(req.path).write_bytes(b'changed')
    status, result, error = api._probe_request(req)
    assert status == 'stale' and result is None


def test_probe_exception_and_mutation_during_attempt_never_return_readiness(api, request_factory, monkeypatch):
    import decoder
    req = request_factory()
    def change(*args, **kwargs):
        os.utime(req.path, ns=(1, 1))
        return {'ready_end_bw': 1}
    monkeypatch.setattr(decoder, 'probe_frame_header', change)
    assert api._probe_request(req)[0:2] == ('stale', None)
    req = request_factory()
    def fail(*args, **kwargs):
        raise RuntimeError('probe failed')
    monkeypatch.setattr(decoder, 'probe_frame_header', fail)
    assert api._probe_request(req)[0:2] == ('error', None)


def test_production_spawn_worker_abstains_on_short_noise_without_parent_state_effect(api, request_factory):
    import decoder
    before = (list(decoder._LAST_EMIT_VERIFIED), list(decoder._HARNESS_ATTEMPT))
    s = api.HeaderCheckpointService(workspace_lease=Lease(), max_pending=1)
    lease = Lease()
    try:
        assert s.submit(request_factory(wall_timeout_s=5), lease)
        result = wait_result(s, timeout=6)
        assert result.status == 'abstained' and result.metadata is None
        assert lease.releases == 1
        assert (decoder._LAST_EMIT_VERIFIED, decoder._HARNESS_ATTEMPT) == before
    finally:
        assert s.close(timeout_s=2)


def test_persistent_workspace_stays_charged_until_worker_termination(service, request_factory):
    workspace = Lease()
    s = service(workspace_lease=workspace)
    prefix = Lease()
    assert s.submit(request_factory(), prefix)
    assert wait_result(s).status == 'ok'
    assert prefix.releases == 1 and workspace.releases == 0
    assert s.close(timeout_s=1)
    assert workspace.releases == 1
    assert s.close(timeout_s=1) and workspace.releases == 1


def test_startup_failure_keeps_workspace_reserved_until_close(service, request_factory, monkeypatch):
    workspace = Lease()
    s = service(workspace_lease=workspace)
    def fail():
        raise OSError('spawn failed')
    monkeypatch.setattr(s, '_start_worker', fail)
    prefix = Lease()
    assert s.submit(request_factory(), prefix)
    assert wait_result(s).status == 'error'
    assert prefix.releases == 1 and workspace.releases == 0
    assert s.close(timeout_s=1) and workspace.releases == 1


def test_nested_metadata_is_bounded_before_pipe_submission(service, request_factory):
    token = tuple(tuple('v' * 100 for _ in range(32)) for _ in range(32))
    s = service()
    lease = Lease()
    assert not s.submit(request_factory(identity_token=token), lease)
    assert lease.releases == 0


def test_worker_result_reports_cpu_and_wall_cost(service, request_factory):
    s = service()
    assert s.submit(request_factory(), Lease())
    result = wait_result(s)
    assert result.cpu_seconds == .01
    assert result.wall_seconds >= 0


def test_worker_workspace_release_failure_is_visible_at_close(api):
    class BrokenLease:
        def release(self): raise OSError('ledger failure')
    s = api.HeaderCheckpointService(workspace_lease=BrokenLease())
    assert not s.close(timeout_s=1)
    assert 'ledger failure' in s.failure


def test_descriptor_cannot_request_unbounded_frontend_padding(service, request_factory):
    s = service()
    for change in (dict(nb_fs=62500*2048), dict(request_id=1 << 1024)):
        lease = Lease()
        assert not s.submit(request_factory(**change), lease)
        assert lease.releases == 0


def test_failed_manager_thread_start_releases_workspace(api, monkeypatch):
    workspace = Lease()
    def fail(self): raise RuntimeError('thread start failed')
    monkeypatch.setattr(api.threading.Thread, 'start', fail)
    with pytest.raises(RuntimeError, match='thread start failed'):
        api.HeaderCheckpointService(workspace_lease=workspace)
    assert workspace.releases == 1


def test_ipc_descriptor_has_small_bounded_wire_size(service, request_factory):
    s = service()
    lease = Lease()
    # Individually valid fields can otherwise sum to a blocking pipe write.
    token = tuple(('x' * 250) + str(i) for i in range(15))
    req = request_factory(path='p' * 4000, identity_token=token)
    assert not s.submit(req, lease)
    assert lease.releases == 0


def test_standalone_child_pins_actual_numeric_pools_without_mutating_parent(api, request_factory, monkeypatch):
    # Launch through the production IPC/env path, changing only the child probe
    # to report its actual loaded pools. This inspects native state, not env alone.
    pytest.importorskip('threadpoolctl')  # inspection only; no runtime dependency
    import sys
    source = str(Path(api.__file__).parent)
    code = '''
import sys
sys.path.insert(0,sys.argv[2])
from multiprocessing.connection import Connection
import header_checkpoint_service as service

def probe(request):
    import os,numpy
    from threadpoolctl import threadpool_info
    return 'ok', {'env':os.environ['OPENBLAS_NUM_THREADS'],'pools':threadpool_info()},None
service._probe_request=probe
service._worker_main(Connection(int(sys.argv[1])))
'''
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','4')
    monkeypatch.setenv('OMP_NUM_THREADS','4')
    monkeypatch.setattr(api, '_worker_command', lambda fd:[sys.executable,'-u','-c',code,str(fd),source])
    workspace, prefix = Lease(), Lease()
    s = api.HeaderCheckpointService(workspace_lease=workspace)
    try:
        assert s.submit(request_factory(wall_timeout_s=5),prefix)
        result=wait_result(s,6)
        assert result.status=='ok' and result.metadata['env']=='1'
        pools=result.metadata['pools']
        assert pools and all(pool['num_threads']==1 for pool in pools)
        assert os.environ['OPENBLAS_NUM_THREADS']==os.environ['OMP_NUM_THREADS']=='4'
        assert prefix.releases==1 and workspace.releases==0
    finally:
        assert s.close(timeout_s=2)
    assert workspace.releases==1


def test_failed_popen_closes_both_ipc_ends_and_prefix_lease(api, request_factory, monkeypatch):
    pipes=[]
    real_pipe=api.multiprocessing.Pipe
    def pipe():
        pair=real_pipe();pipes.extend(pair);return pair
    def fail(*args,**kwargs):raise OSError('exec failed')
    monkeypatch.setattr(api.multiprocessing,'Pipe',pipe)
    monkeypatch.setattr(api.subprocess,'Popen',fail)
    workspace,prefix=Lease(),Lease()
    s=api.HeaderCheckpointService(workspace_lease=workspace)
    try:
        assert s.submit(request_factory(),prefix)
        result=wait_result(s)
        assert result.status=='error' and 'exec failed' in result.error
        assert len(pipes)==2 and all(endpoint.closed for endpoint in pipes)
        assert prefix.releases==1 and workspace.releases==0
    finally:assert s.close(timeout_s=1)
    assert workspace.releases==1


def test_unsupported_fd_launch_environment_abstains_without_lease_leak(api,request_factory,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(api,'os',SimpleNamespace(name='unsupported'))
    workspace,prefix=Lease(),Lease()
    s=api.HeaderCheckpointService(workspace_lease=workspace)
    try:
        assert s.submit(request_factory(),prefix)
        result=wait_result(s)
        assert result.status=='error' and 'POSIX' in result.error
        assert prefix.releases==1 and workspace.releases==0
    finally:assert s.close(timeout_s=1)
    assert workspace.releases==1
