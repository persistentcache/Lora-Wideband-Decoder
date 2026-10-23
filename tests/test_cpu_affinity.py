import pytest
import cpu_affinity as affinity


def topology(tmp_path, groups):
    for group in groups:
        for cpu in group:
            folder = tmp_path / f'cpu{cpu}' / 'topology'
            folder.mkdir(parents=True)
            (folder / 'thread_siblings_list').write_text(','.join(map(str, group)))
    return tmp_path


def test_partition_keeps_smt_siblings_together(tmp_path):
    root = topology(tmp_path, [(0, 4), (1, 5), (2, 6), (3, 7)])
    capture, process = affinity.capture_partition(set(range(8)), root)
    assert capture == {1, 5}
    assert process == {0, 2, 3, 4, 6, 7}


def test_partition_never_expands_existing_affinity(tmp_path):
    root = topology(tmp_path, [(2, 10), (3, 11), (6, 14), (7, 15)])
    allowed = {2, 3, 6, 7, 10, 11, 14}
    capture, process = affinity.capture_partition(allowed, root)
    assert capture.isdisjoint(process)
    assert capture | process == allowed
    assert 15 not in capture | process


@pytest.mark.parametrize('groups', [[(0,)], [(0, 2), (1, 3)], [(0,), (1,), (2,)]])
def test_small_hosts_keep_their_existing_cpu_assignment(tmp_path, groups):
    root = topology(tmp_path, groups)
    allowed = {cpu for group in groups for cpu in group}
    assert affinity.capture_partition(allowed, root) is None


def test_missing_topology_does_not_guess_physical_cores(tmp_path):
    assert affinity.capture_partition(set(range(8)), tmp_path) is None


def test_launcher_isolates_only_uhd(monkeypatch):
    monkeypatch.setattr(affinity, 'capture_partition', lambda: ({1, 5}, {0, 2, 3, 4, 6, 7}))
    monkeypatch.setattr(affinity.shutil, 'which', lambda name: '/usr/bin/taskset')
    cap, proc = affinity.isolate_capture_commands('capture', 'detect', 'uhd')
    assert cap == '/usr/bin/taskset -c 1,5 capture'
    assert proc == '/usr/bin/taskset -c 0,2,3,4,6,7 detect'
    assert affinity.isolate_capture_commands('capture', 'detect', 'hackrf') == ('capture', 'detect')
    monkeypatch.setenv('LORA_CAPTURE_ISOLATION', '0')
    assert affinity.isolate_capture_commands('capture', 'detect', 'uhd') == ('capture', 'detect')


def test_missing_taskset_keeps_commands_usable(monkeypatch):
    monkeypatch.setattr(affinity.shutil, 'which', lambda name: None)
    assert affinity.isolate_capture_commands('capture', 'detect', 'uhd') == ('capture', 'detect')


def test_reader_stays_inside_restricted_cpu_set(monkeypatch):
    calls = []
    monkeypatch.setattr(affinity.os, 'sched_getaffinity', lambda pid: {8, 10})
    monkeypatch.setattr(affinity.os, 'sched_setaffinity', lambda pid, cpus: calls.append((pid, cpus)))
    affinity.pin_reader()
    assert calls == [(0, {8})]


def test_affinity_permission_failure_is_nonfatal(monkeypatch):
    def denied(*args):
        raise PermissionError('restricted host')
    monkeypatch.setattr(affinity.os, 'sched_getaffinity', denied)
    assert affinity.capture_partition() is None
    affinity.pin_reader()
