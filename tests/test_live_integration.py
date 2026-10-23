import io
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import cpu_affinity
from detector import StreamBuffer
from web import app as web


def test_reader_progress_counts_only_unique_consumed_samples():
    reader = StreamBuffer(io.BytesIO(np.zeros(2000, dtype=np.int16).tobytes()),
                          rate=1000, buf_seconds=2)
    try:
        reader._thread.join(timeout=2)
        assert not reader._thread.is_alive()
        raw, skipped = reader.read_raw(20)
        assert len(raw) == 40 and skipped == 0
        assert reader.progress() == (1000, 20, 980, 0)
        assert reader.skip_to_latest(10) == 970
        assert reader.progress() == (1000, 20, 10, 970)
        reader.read_raw(10)
        assert reader.progress() == (1000, 30, 0, 970)
    finally:
        reader._cvt_pool.shutdown(wait=True)


@pytest.mark.parametrize('sdr,driver,isolated', [
    ('usrp', 'uhd', True), ('soapy', 'uhd', True), ('hackrf', 'hackrf', False),
])
def test_web_launcher_assigns_disjoint_cpus(monkeypatch, tmp_path, sdr, driver, isolated):
    monkeypatch.setattr(web, 'SETTINGS', dict(sdr=sdr, channelize_enabled=False, auto_gain=False))
    monkeypatch.setattr(web, '_radio_cfg', lambda: dict(rate_hz=32000000,
        bandwidth_hz=32000000, center_mhz=915, gain=40, soapy_driver=driver))
    monkeypatch.setattr(web, '_effective_capture_dir', lambda: str(tmp_path / 'capture folder'))
    monkeypatch.setattr(web.sdr_profiles, 'soapy_python', lambda: sys.executable)
    monkeypatch.setattr(cpu_affinity, 'capture_partition', lambda: ({1, 5}, {0, 2, 3, 4, 6, 7}))
    monkeypatch.setattr(cpu_affinity.shutil, 'which', lambda name: '/usr/bin/' + name)
    cap, det = [shlex.split(p) for p in web._build_pipeline_cmd().split(' | ')]
    assert cap[cap.index('--driver') + 1] == driver
    assert det[det.index('--export-iq') + 1] == str(tmp_path / 'capture folder')
    if isolated:
        assert cap[:3] == ['/usr/bin/taskset', '-c', '1,5']
        assert det[:3] == ['/usr/bin/taskset', '-c', '0,2,3,4,6,7']
    else:
        assert cap[0] == sys.executable
        assert det[0] == sys.executable


@pytest.mark.parametrize('warning', [
    '[INPUT WARNING] Receiving 2.6 Msps of 3 Msps requested.',
    '[22s] CATCHUP skip 30M samples (10s)',
])
def test_web_preserves_input_warning_and_clears_after_recovery(monkeypatch, tmp_path, warning):
    log = tmp_path / 'pipeline.log'
    log.write_text(warning + '\n[STAT] 30s | 3.2Msps | win=60 det=0 '
                   'save_q=0 dec_q=0 pipe=2ms input=2.6Msps backlog=0.1s\n')
    health = dict(web.HEALTH)
    monkeypatch.setattr(web, 'HEALTH', health)
    monkeypatch.setattr(web, 'PIPELINE_LOG', str(log))
    monkeypatch.setattr(web, 'PIPELINE', dict(running=True, started_at=1))
    monkeypatch.setattr(web, '_radio_cfg', lambda: dict(rate_hz=3000000))
    snapshots = []
    monkeypatch.setattr(web, '_broadcast', lambda event: snapshots.append(dict(event['data'])))

    class Finished(BaseException):
        pass

    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            with log.open('a') as f:
                f.write('[KEEP-UP RECOVERED] Input and processing rates recovered.\n')
        else:
            raise Finished

    monkeypatch.setattr(web, 'time', SimpleNamespace(time=lambda: 100 + len(sleeps), sleep=sleep))
    with pytest.raises(Finished):
        web._tail_health()
    assert snapshots[0]['warn']
    assert snapshots[0]['input_msps'] == 2.6
    assert snapshots[0]['backlog_s'] == 0.1
    assert snapshots[-1]['warn'] is None
