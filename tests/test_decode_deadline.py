"""The supervisor must honor the CPU budget actually sent to a decoder job."""
import ast
import io
import os
import select
from pathlib import Path
from types import SimpleNamespace

import pytest
import detector


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


@pytest.fixture(scope='module')
def production_dispatch():
    # Execute the real protocol write, deadline, raw-fd reader and kill branch.
    # Only the clock/pipe are simulated; no duplicate watchdog policy is used.
    tree = ast.parse(Path(detector.__file__).read_text())
    manager = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                   and c.name == 'BackgroundDecoder' for n in c.body
                   if isinstance(n, ast.FunctionDef) and n.name == '_manager')
    body = next(n.body for n in ast.walk(manager) if isinstance(n, ast.Try)
                and any(assigned(child, '_bud_tok') for child in n.body))
    start = next(i for i, n in enumerate(body) if assigned(n, '_bud_tok'))
    end = next(i for i, n in enumerate(body) if assigned(n, 'output'))
    # One real manager iteration gives the new fatal-abort continue its
    # production loop context while preserving the actual deadline body.
    once = ast.For(target=ast.Name(id='_once', ctx=ast.Store()),
                   iter=ast.Tuple(elts=[ast.Constant(None)], ctx=ast.Load()),
                   body=body[start:end], orelse=[])
    return compile(ast.fix_missing_locations(ast.Module(body=[once], type_ignores=[])),
                   detector.__file__, 'exec')


class Worker:
    pid = 321
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = SimpleNamespace(fileno=lambda: 123)
        self.killed = False
    def kill(self):
        self.killed = True


def dispatch(code, monkeypatch, *, sf=10, override=None, default='10', finish=None):
    monkeypatch.setenv('LORA_DECODE_BUDGET_S', default)
    now = [0.0]
    worker = Worker()
    logs = []
    def wait(read, write, error, timeout):
        now[0] = min(now[0] + timeout, finish) if finish is not None else now[0] + timeout
        return ([123], [], []) if finish is not None and now[0] >= finish else ([], [], [])
    monkeypatch.setattr(select, 'select', wait)
    env = dict(job_budget=override, fname=f'SF{sf}_125k_test.cf32', fpath='/test/capture.cf32',
               relay_field='', worker=worker,
               os=SimpleNamespace(environ=os.environ, read=lambda fd, size: b'__END__\n'),
               time=SimpleNamespace(time=lambda: now[0]),
               self=SimpleNamespace(_rawlog=lambda *args: logs.append(args), _aborting=lambda:False))
    exec(code, env)
    return env, worker, now[0], logs


def test_explicit_slow_budget_can_finish_after_old_default_watchdog(production_dispatch, monkeypatch):
    env, worker, elapsed, logs = dispatch(production_dispatch, monkeypatch, override=45, finish=110)
    assert worker.stdin.getvalue().endswith('\t45.0\n')
    assert not env['timed_out'] and not worker.killed
    assert elapsed == 110
    assert env['_wall'] == pytest.approx(220.91883092036784)


@pytest.mark.parametrize('sf,override,default,wall', [
    (7, None, '10', 90),
    (11, None, '10', 90),
    (12, None, '10', 114.8528137423857),
    (11, None, '25', 180),
    (12, 45, '10', 411.8376618407357),
    (11, 54.5899, '10', 357.6),  # observed pressure floor, wire rounds54.6
    (11, 0, '10', 120),
    (11, 0.0, '10', 120),
    (11, .04, '10', 120),  # wire0.0 means unlimited CPU, still bounded wall
    (11, .06, '10', 90),
    (11, -1, '10', 120),
    (11, None, '0', 120),
])
def test_silent_worker_is_killed_at_effective_bounded_deadline(
        production_dispatch, monkeypatch, sf, override, default, wall):
    env, worker, elapsed, logs = dispatch(production_dispatch, monkeypatch,
                                         sf=sf, override=override, default=default)
    assert env['timed_out'] and worker.killed
    assert elapsed == pytest.approx(wall)
    assert env['_wall'] == pytest.approx(wall)
    assert env['output_lines'][-1] == f'ERROR: DECODE TIMEOUT ({wall:.0f}s wall)'


def test_start_log_identifies_the_effective_budget_and_worker(production_dispatch, monkeypatch):
    env, worker, elapsed, logs = dispatch(production_dispatch, monkeypatch,
                                         sf=11, override=54.5899, finish=1)
    start = next(message for kind, message in logs if kind == 'START')
    assert 'pid=321' in start
    assert 'base_budget=54.6' in start
    assert 'cpu_budget=218.4' in start
    assert 'wall_budget=357.6' in start
    assert 'SF11_125k_test.cf32' in start
