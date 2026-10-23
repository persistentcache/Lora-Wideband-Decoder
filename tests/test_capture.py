import sys
from types import SimpleNamespace

import pytest
import soapy_rx


@pytest.mark.parametrize('driver,extra,expected', [
    ('uhd', [], 'driver=uhd,num_recv_frames=1024'),
    ('uhd', ['--uhd-recv-frames', '512'], 'driver=uhd,num_recv_frames=512'),
    ('uhd', ['--uhd-recv-frames', '0'], 'driver=uhd'),
    ('uhd,num_recv_frames=128', [], 'driver=uhd,num_recv_frames=128'),
    ('hackrf', [], 'driver=hackrf'),
    ('rtlsdr', [], 'driver=rtlsdr'),
])
def test_device_buffer_arguments_reach_device_constructor(monkeypatch, driver, extra, expected):
    calls = []

    def device(args):
        calls.append(args)
        raise RuntimeError('stop before hardware access')

    monkeypatch.setitem(sys.modules, 'SoapySDR', SimpleNamespace(
        Device=device, SOAPY_SDR_RX=0, SOAPY_SDR_CS16='CS16', SOAPY_SDR_OVERFLOW=-4))
    monkeypatch.setattr(sys, 'argv', ['soapy_rx.py', '--driver', driver,
                                    '-f', '915e6', '-s', '32e6', *extra])
    assert soapy_rx.main() == 3
    assert calls == [expected]


def test_negative_buffer_count_is_rejected_before_device_access(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['soapy_rx.py', '--driver', 'uhd',
                                    '-f', '915e6', '-s', '32e6', '--uhd-recv-frames', '-1'])
    with pytest.raises(SystemExit) as exc:
        soapy_rx.main()
    assert exc.value.code == 2
