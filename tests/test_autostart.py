"""Boot autostart (run/web.py --autostart / the installed service): the
pipeline starts with the saved settings, keeps retrying until healthy, and a
deliberate Stop from the UI ends it."""
import os
import subprocess

import pytest
from web import app as web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Proc:
    def __init__(self, alive):
        self._alive = alive

    def poll(self):
        return None if self._alive else 1


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(web, 'AUTOSTART', {'pending': True, 'attempts': 0, 'cancelled': False})
    monkeypatch.setattr(web, 'PIPELINE', dict(web.PIPELINE, proc=None, running=False,
                                              want_running=False, restart_attempts=4,
                                              last_healthy_at=0.0))
    return web


def test_tick_starts_the_pipeline_and_resets_the_supervisor_budget(armed):
    calls = []
    assert armed._autostart_tick(start=lambda: calls.append(1) or {'ok': True}) == 'started'
    assert calls == [1]
    assert armed.PIPELINE['restart_attempts'] == 0
    assert armed.AUTOSTART == {'pending': True, 'attempts': 1, 'cancelled': False}


def test_tick_keeps_retrying_while_the_sdr_is_absent(armed):
    results = [armed._autostart_tick(start=lambda: {'ok': False, 'error': 'no SDR'})
               for _ in range(3)]
    assert results == ['failed'] * 3 and armed.AUTOSTART['attempts'] == 3
    assert armed.AUTOSTART['pending']


def test_tick_survives_an_exception_from_start(armed):
    def boom():
        raise RuntimeError('usb reset')
    assert armed._autostart_tick(start=boom) == 'failed'
    assert armed.AUTOSTART['pending']


def test_tick_waits_while_a_launch_is_alive_but_not_yet_healthy(armed):
    armed.PIPELINE.update(proc=_Proc(True), running=True)
    assert armed._autostart_tick(start=lambda: pytest.fail('must not relaunch')) == 'running'


def test_tick_leaves_a_crash_to_the_supervisor_while_it_still_has_budget(armed):
    armed.PIPELINE.update(proc=_Proc(False), running=True, want_running=True,
                          restart_attempts=web._SUP_MAX_ATTEMPTS - 1)
    assert armed._autostart_tick(start=lambda: pytest.fail('must not relaunch')) == 'supervised'
    assert armed.PIPELINE['restart_attempts'] == web._SUP_MAX_ATTEMPTS - 1


def test_tick_relaunches_once_the_supervisor_has_given_up(armed):
    armed.PIPELINE.update(proc=_Proc(False), running=True, want_running=True,
                          restart_attempts=web._SUP_MAX_ATTEMPTS)
    assert armed._autostart_tick(start=lambda: {'ok': True}) == 'started'
    assert armed.PIPELINE['restart_attempts'] == 0


def test_healthy_telemetry_hands_over_to_the_supervisor(armed):
    armed.PIPELINE['last_healthy_at'] = 123.0
    assert armed._autostart_tick(start=lambda: pytest.fail('must not relaunch')) == 'healthy'
    assert not armed.AUTOSTART['pending']
    assert armed._autostart_tick() == 'off'


def test_stop_from_the_ui_cancels_autostart(armed, monkeypatch):
    monkeypatch.setattr(web, 'stop_pipeline', lambda: {'ok': True})
    assert web.app.test_client().post('/api/pipeline/stop').status_code == 200
    assert armed.AUTOSTART['cancelled']
    assert armed._autostart_tick(start=lambda: pytest.fail('must not relaunch')) == 'cancelled'
    assert not armed.AUTOSTART['pending']


def test_stop_without_autostart_does_not_mark_anything(monkeypatch):
    monkeypatch.setattr(web, 'AUTOSTART', {'pending': False, 'attempts': 0, 'cancelled': False})
    monkeypatch.setattr(web, 'stop_pipeline', lambda: {'ok': True})
    web.app.test_client().post('/api/pipeline/stop')
    assert not web.AUTOSTART['cancelled']


def test_loop_sleeps_then_ticks_until_done(armed, monkeypatch):
    slept, ticks = [], iter(['failed', 'started', 'running', 'healthy'])

    def tick():
        r = next(ticks)
        if r == 'healthy':
            armed.AUTOSTART['pending'] = False
        return r
    monkeypatch.setattr(web, '_autostart_tick', tick)
    armed._autostart_loop(sleep=slept.append)
    assert slept[0] == web._AUTOSTART_FIRST_DELAY_S
    assert slept[1:] == [web._AUTOSTART_RETRY_S] * 4


def _installer(*args, check=True):
    return subprocess.run(['bash', os.path.join(ROOT, 'install.sh'), *args],
                          capture_output=True, text=True, check=check)


def test_installer_renders_a_unit_that_autostarts_as_the_user():
    out = _installer('--print-service').stdout
    assert 'ExecStart=/usr/bin/python3 "%s/run/web.py" --autostart' % ROOT in out
    assert 'Environment=LORA_AUTOSTART=1' in out
    assert 'WorkingDirectory=%s' % ROOT in out
    assert 'User=%s' % (os.environ.get('SUDO_USER') or os.environ.get('USER')) in out
    assert 'Restart=always' in out and 'WantedBy=multi-user.target' in out
    assert 'KillMode=control-group' in out            # stopping the unit releases the SDR


def test_installer_rejects_unknown_options():
    r = _installer('--bogus', check=False)
    assert r.returncode == 2 and 'unknown option' in r.stderr


def test_installer_help_lists_the_service_flags():
    out = _installer('--help').stdout
    assert '--install-service' in out and '--installautorun' in out and '--uninstall-service' in out
