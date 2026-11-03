import subprocess
import sys
import threading

import pytest

from detector import BackgroundDecoder


@pytest.fixture
def decoder(monkeypatch):
    monkeypatch.setenv('LORA_DECODE_WORKERS', '1')
    monkeypatch.setenv('LORA_DECODE_RAWLOG', '')
    monkeypatch.setenv('LORA_PKT_LOG', '')
    monkeypatch.delenv('LORA_KEEP_IQ', raising=False)
    monkeypatch.delenv('LORA_KEEP_CAPTURES_DIR', raising=False)
    # Construct the real scheduler without launching a decoder process. Tests
    # drive its real queues synchronously, or explicitly run the real manager.
    with monkeypatch.context() as startup:
        startup.setattr(BackgroundDecoder, '_manager', lambda self: None)
        instance = BackgroundDecoder()
        for thread in instance._threads:
            thread.join()
    return instance


def submit(decoder, name, **kwargs):
    decoder.submit(name, name, **kwargs)


def take(decoder):
    _, _, job = decoder._queue.get_nowait()
    return job, decoder._slow_take(job)


def finish(decoder, job, token):
    decoder._slow_retire(token)
    decoder._ref_dec_and_maybe_unlink(job[0])
    decoder._queue.task_done()


def test_deferred_job_gets_a_turn_with_continuous_fast_arrivals(decoder):
    submit(decoder, 'deferred', slow=True)
    submit(decoder, 'fast-0', prio=10)
    seen = []
    for number in range(20):
        submit(decoder, f'fast-{number + 1}', prio=20 + number)
        decoder.maybe_release_straggler()
        job, token = take(decoder)
        seen.append(job[1])
        finish(decoder, job, token)
        if job[1] == 'deferred':
            break
    assert seen[0].startswith('fast-')
    assert 'deferred' in seen, 'slow work starved while fast work stayed queued'
    assert len(seen) <= 9
    assert not decoder._queue.empty()


def test_released_job_cannot_be_overtaken_by_new_fast_jobs(decoder):
    submit(decoder, 'deferred', slow=True)
    decoder.maybe_release_straggler()
    submit(decoder, 'fresh-high-quality', prio=100)
    job, token = take(decoder)
    assert job[1] == 'deferred'
    assert token is job
    finish(decoder, job, token)


@pytest.mark.parametrize('release', ['maybe_release_straggler', 'maybe_release_pressure'])
@pytest.mark.parametrize('pressure,stress', [(0.21, False), (0.0, True)])
def test_live_pressure_and_gate_stress_block_slow_admission(
        decoder, release, pressure, stress):
    submit(decoder, 'deferred', slow=True)
    decoder.set_pressure(pressure)
    if stress:
        decoder._gate_stress.set()
    getattr(decoder, release)()
    assert decoder._queue.empty()
    assert decoder.pending() == 1
    decoder.set_pressure(0)
    decoder._gate_stress.clear()
    getattr(decoder, release)()
    job, token = take(decoder)
    assert job[1] == 'deferred'
    finish(decoder, job, token)


def test_same_path_fast_job_and_stale_token_cannot_free_slow_reservation(decoder, tmp_path):
    path = str(tmp_path / 'capture.cf32')
    (tmp_path / 'capture.cf32').write_bytes(b'iq')
    submit(decoder, path, slow=True, relay_after_syms=1)
    submit(decoder, 'second-slow', slow=True)
    decoder.maybe_release_straggler()
    slow, token = take(decoder)
    submit(decoder, path)
    fast, fast_token = take(decoder)
    assert fast_token is None
    finish(decoder, fast, fast_token)
    assert (tmp_path / 'capture.cf32').exists()
    decoder._slow_retire(fast)
    decoder.maybe_release_straggler()
    assert decoder._queue.empty()
    finish(decoder, slow, token)
    assert not (tmp_path / 'capture.cf32').exists()
    decoder.maybe_release_straggler()
    decoder._slow_retire(token)
    second, second_token = take(decoder)
    assert second_token is second
    finish(decoder, second, second_token)
    assert decoder.pending() == 0


def test_concurrent_release_paths_reserve_only_one_job(decoder):
    for number in range(10):
        submit(decoder, f'slow-{number}', slow=True)
    threads = [threading.Thread(target=decoder.maybe_release_straggler)
               for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert decoder._queue.qsize() == 1
    assert decoder._slow_queue.qsize() == 9
    job, token = take(decoder)
    decoder.maybe_release_pressure()
    assert decoder._queue.empty()
    finish(decoder, job, token)
    decoder.maybe_release_pressure()
    next_job, next_token = take(decoder)
    assert next_job[1] == 'slow-1'
    finish(decoder, next_job, next_token)


def test_shutdown_fold_does_not_duplicate_a_reserved_job(decoder):
    submit(decoder, 'first', slow=True)
    submit(decoder, 'second', slow=True)
    decoder.maybe_release_straggler()
    decoder.drain_slow_into_fast()
    decoder.drain_slow_into_fast()
    names = []
    while not decoder._queue.empty():
        job, token = take(decoder)
        names.append(job[1])
        finish(decoder, job, token)
    assert names == ['first', 'second']
    assert decoder.pending() == 0
    assert decoder._refs == {}


def test_failed_release_preserves_capture_and_future_slow_turn(decoder, monkeypatch, tmp_path):
    path = tmp_path / 'capture.cf32'
    path.write_bytes(b'iq')
    submit(decoder, str(path), slow=True)
    submit(decoder, 'second', slow=True)
    original_put = decoder._queue.put

    def cancelled(*args, **kwargs):
        raise InterruptedError('dispatch cancelled')

    monkeypatch.setattr(decoder._queue, 'put', cancelled)
    with pytest.raises(InterruptedError):
        decoder.maybe_release_straggler()
    assert decoder.pending() == 2
    assert path.exists()
    monkeypatch.setattr(decoder._queue, 'put', original_put)
    decoder.maybe_release_straggler()
    job, token = take(decoder)
    assert job[1] == str(path)
    assert token is job
    finish(decoder, job, token)
    assert not path.exists()


class StopManager(BaseException):
    """End a synchronous manager run at the test's dispatch boundary."""


def stop_when_empty(decoder, monkeypatch):
    original_get = decoder._queue.get

    def get(*args, **kwargs):
        if decoder._queue.empty():
            raise StopManager()
        return original_get(*args, **kwargs)

    monkeypatch.setattr(decoder._queue, 'get', get)


@pytest.mark.parametrize('worker_output', ['normal', 'budget', 'death'])
def test_manager_retires_reserved_jobs_and_retries_without_losing_files(
        decoder, monkeypatch, tmp_path, worker_output):
    # A lightweight real subprocess implements the decoder wire protocol. All
    # admission, retry, reservation, queue accounting and file cleanup stay real.
    worker_code = '''
import sys
mode = sys.argv[1]
for line in sys.stdin:
    budget = line.rstrip('\\n').split('\\t')[2]
    if mode == 'death':
        break
    if mode == 'budget' and not budget:
        print('[BUDGET] exhausted', flush=True)
    print('__END__', flush=True)
'''
    processes = []

    def start():
        worker = subprocess.Popen([sys.executable, '-u', '-c', worker_code, worker_output],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        processes.append(worker)
        return worker

    monkeypatch.setattr(decoder, '_start_worker', start)
    decoder._pressure_n = 0
    paths = [tmp_path / f'capture-{index}.cf32' for index in range(12)]
    for index, path in enumerate(paths):
        path.write_bytes(b'iq')
        submit(decoder, str(path), slow=index == 0)
    stop_when_empty(decoder, monkeypatch)
    try:
        with pytest.raises(StopManager):
            decoder._manager()
        assert decoder.pending() == 0
        assert decoder._refs == {}
        assert decoder._queue.unfinished_tasks == 0
        assert decoder._slow_queue.unfinished_tasks == 0
        assert all(not path.exists() for path in paths)
        # Completion, budget retries and worker EOF must all permit the next
        # slow job to be admitted; a leaked reservation would leave it stuck.
        submit(decoder, 'next-slow', slow=True)
        decoder.maybe_release_pressure()
        job, token = take(decoder)
        assert job[1] == 'next-slow'
        assert token is job
        finish(decoder, job, token)
    finally:
        for worker in processes:
            if worker.poll() is None:
                worker.stdin.close()
            worker.wait(timeout=5)
            worker.stdout.close()


@pytest.fixture(scope='module')
def production_retry():
    # Keep the actual manager eligibility and refcount/queue transfer together;
    # no decoder, device or long watchdog wait is needed for these transitions.
    import ast
    from pathlib import Path
    import detector
    tree = ast.parse(Path(detector.__file__).read_text())
    manager = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                   and c.name == 'BackgroundDecoder' for n in c.body
                   if isinstance(n, ast.FunctionDef) and n.name == '_manager')
    retry = next(n for n in ast.walk(manager) if isinstance(n, ast.If)
                 and '_stress_shrunk' in ast.unparse(n.test)
                 and '_slow_queue.put' in ast.unparse(n))
    return compile(ast.Module(body=[retry], type_ignores=[]), detector.__file__, 'exec')


VALID_HEADER = '  Header: PL=57 CR=4/8 CRC=yes HDR_CHK=OK  nibs=[3, 9]'
# Exact final header/termination excerpt from Retest007's live decode log.
# Its checksum-valid PL57 header appeared only in host replay. The live failed
# parse proves search progress, not a verified packet or a noise-free capture.
RETEST_007_WALL_OUTPUT = '''
  Header: PL=85 CR=4/5 CRC=no HDR_CHK=FAIL  nibs=[5, 5, 2, 2, 7, 14, 0, 0, 3, 0]
  *** HEADER CHECKSUM FAILED ***

  --- Pass2 Retry #1 ---
ERROR: DECODE TIMEOUT (115s wall)
'''


def retry_env(decoder, path, output, **overrides):
    import re
    state = dict(self=decoder, fpath=str(path), fname=path.name, output=output,
                 job_budget=None, _stress_shrunk=False, timed_out=True,
                 worker_died=False, relay_after_syms=None, relay_before_syms=None,
                 _re_t=re)
    state.update(overrides)
    return state


@pytest.mark.parametrize('output', [
    VALID_HEADER, RETEST_007_WALL_OUTPUT,
    'Header: PL=4 CR=4/5 CRC=no HDR_CHK=FAIL',
    'Header: PL=237 CR=4/8 CRC=yes HDR_CHK=FAIL',
])
@pytest.mark.parametrize('budget,stressed', [(None, False), (54.6, True)])
def test_parsed_header_wall_timeout_gets_one_slow_turn_without_losing_capture(
        decoder, monkeypatch, tmp_path, production_retry, budget, stressed, output):
    path = tmp_path / 'SF12_62.50k_test.cf32'
    path.write_bytes(b'iq')
    submit(decoder, str(path))
    fast, fast_token = take(decoder)
    state = retry_env(decoder, path, output,
                      job_budget=budget, _stress_shrunk=stressed)
    exec(production_retry, state)
    assert decoder._slow_queue.qsize() == 1
    finish(decoder, fast, fast_token)
    assert path.exists() and decoder._refs[str(path)] == 1
    decoder.maybe_release_straggler()
    slow, slow_token = take(decoder)
    assert slow[4] == decoder._slow_budget
    # The explicit slow budget makes a second timeout terminal, even if its
    # header parses too. A single capture cannot retry indefinitely.
    state.update(job_budget=slow[4], _stress_shrunk=False)
    exec(production_retry, state)
    assert decoder._slow_queue.empty()
    finish(decoder, slow, slow_token)
    assert decoder.pending() == 0 and decoder._refs == {}
    assert not path.exists()
    assert decoder._queue.unfinished_tasks == decoder._slow_queue.unfinished_tasks == 0


@pytest.mark.parametrize('output,overrides', [
    ('', {}),
    ('no preamble', {}),
    ('Header: PL=3 CR=4/8 CRC=yes HDR_CHK=OK', {}),
    ('Header: PL=238 CR=4/8 CRC=yes HDR_CHK=FAIL', {}),
    ('Header: PL=255 CR=4/8 CRC=yes HDR_CHK=OK', {}),
    ('Header: PL=-1 CR=4/8 CRC=yes HDR_CHK=OK', {}),
    ('Header: PL=57 CR=4/4 CRC=yes HDR_CHK=OK', {}),
    ('Header: PL=57 CR=4/9 CRC=yes HDR_CHK=FAIL', {}),
    ('Header: PL=57 CR=4/8 CRC=unknown HDR_CHK=FAIL', {}),
    ('Header: PL=57 CR=4/8 CRC=yes HDR_CHK=unknown', {}),
    ('diagnostic mentions HDR_CHK=OK without a parsed header', {}),
    (VALID_HEADER, {'worker_died': True}),
    (VALID_HEADER, {'timed_out': False}),
    (VALID_HEADER, {'job_budget': 45.0}),
    (VALID_HEADER, {'job_budget': 2.0}),
    (VALID_HEADER, {'relay_after_syms': 100}),
    (VALID_HEADER, {'relay_before_syms': 200}),
])
def test_wall_retry_excludes_no_parse_invalid_header_death_overrides_and_isolation(
        decoder, tmp_path, production_retry, output, overrides):
    path = tmp_path / 'SF12_62.50k_test.cf32'
    path.write_bytes(b'iq')
    submit(decoder, str(path))
    job, token = take(decoder)
    exec(production_retry, retry_env(decoder, path, output, **overrides))
    assert decoder._slow_queue.empty()
    finish(decoder, job, token)
    assert decoder.pending() == 0 and not path.exists()


def test_existing_cpu_budget_fallback_still_needs_no_header(
        decoder, tmp_path, production_retry):
    path = tmp_path / 'SF10_125k_test.cf32'
    path.write_bytes(b'iq')
    submit(decoder, str(path))
    job, token = take(decoder)
    exec(production_retry, retry_env(decoder, path, '[BUDGET] exhausted', timed_out=False))
    assert decoder._slow_queue.qsize() == 1
    finish(decoder, job, token)
    decoder.maybe_release_straggler()
    job, token = take(decoder)
    finish(decoder, job, token)
    assert not path.exists()
