"""Frame-length-exact capture is the default wherever the host can carry its
memory budget; plain capture otherwise or on request; the explicit diagnostic
form keeps its old meaning."""
import pytest

import detector
from detector import shared_capture_plan, SHARED_CAPTURE_PLAN_BYTES, _MIB
from test_shared_recorder import rec            # noqa: F401 (fixture)
from test_capture_timeline import reader

GIB = 1024 * _MIB
HEADER_EXTRA = (8 * 64 * _MIB + 128 * _MIB) + 2 * 64 * _MIB + 65536


def test_auto_enables_exact_capture_with_headroom():
    plan = shared_capture_plan({}, 16 * GIB)
    assert plan['enabled'] and plan['header'] and not plan['explicit']
    assert plan['budget_bytes'] == SHARED_CAPTURE_PLAN_BYTES + HEADER_EXTRA
    assert 'frame-exact capture' in plan['reason']


def test_auto_falls_back_to_plain_on_a_small_host():
    need = 2 * (SHARED_CAPTURE_PLAN_BYTES + HEADER_EXTRA) + 2 * GIB
    plan = shared_capture_plan({}, need - 1)
    assert not plan['enabled'] and 'plain capture' in plan['reason']
    assert shared_capture_plan({}, need)['enabled']
    assert not shared_capture_plan({}, None)['enabled']    # unknown memory: stay plain


def test_auto_without_header_checkpoints_needs_less():
    plan = shared_capture_plan({'LORA_HEADER_CHECKPOINT': '0'}, 7 * GIB)
    assert plan['enabled'] and not plan['header']
    assert plan['budget_bytes'] == SHARED_CAPTURE_PLAN_BYTES


def test_auto_with_header_requested_but_no_room_is_an_error():
    with pytest.raises(RuntimeError, match='does not fit'):
        shared_capture_plan({'LORA_HEADER_CHECKPOINT': '1'}, 4 * GIB)


def test_plain_on_request_and_its_contradiction():
    plan = shared_capture_plan({'LORA_SHARED_CAPTURE': '0'}, 64 * GIB)
    assert not plan['enabled'] and plan['explicit']
    with pytest.raises(RuntimeError, match='requires LORA_SHARED_CAPTURE=1'):
        shared_capture_plan({'LORA_SHARED_CAPTURE': '0', 'LORA_HEADER_CHECKPOINT': '1'}, 64 * GIB)


def test_explicit_form_keeps_its_old_meaning():
    plan = shared_capture_plan({'LORA_SHARED_CAPTURE': '1', 'LORA_SHARED_CAPTURE_MB': '8'}, 1 * GIB)
    assert plan['enabled'] and plan['explicit'] and not plan['header']
    assert plan['budget_bytes'] == 8 * _MIB                  # memory is not consulted
    with pytest.raises(RuntimeError, match='memory|budget'):
        shared_capture_plan({'LORA_SHARED_CAPTURE': '1', 'LORA_HEADER_CHECKPOINT': '1',
                             'LORA_SHARED_CAPTURE_MB': '2048'}, 64 * GIB)
    plan = shared_capture_plan({'LORA_SHARED_CAPTURE': '1', 'LORA_HEADER_CHECKPOINT': '1',
                                'LORA_SHARED_CAPTURE_MB': '3072'}, 64 * GIB)
    assert plan['enabled'] and plan['header'] and plan['budget_bytes'] == 3072 * _MIB


def test_recorder_takes_exact_capture_by_default_on_a_big_host(rec, monkeypatch, capsys):
    monkeypatch.delenv('LORA_SHARED_CAPTURE')
    monkeypatch.delenv('LORA_SHARED_CAPTURE_MB')
    monkeypatch.setattr(detector, '_mem_available_bytes', lambda: 16 * GIB)
    rec.set_history_reader(reader())
    assert rec._shared is not None and rec._header is not None
    assert rec._shared.budget.total_bytes == SHARED_CAPTURE_PLAN_BYTES + HEADER_EXTRA
    assert '[CAPTURE] frame-exact capture' in capsys.readouterr().err
    rec._shared.close(wait=False)


def test_recorder_stays_plain_on_a_small_host_and_says_so(rec, monkeypatch, capsys):
    monkeypatch.delenv('LORA_SHARED_CAPTURE')
    monkeypatch.delenv('LORA_SHARED_CAPTURE_MB')
    monkeypatch.setattr(detector, '_mem_available_bytes', lambda: 4 * GIB)
    rec.set_history_reader(reader())
    assert rec._shared is None
    assert '[CAPTURE] plain capture' in capsys.readouterr().err


def test_auto_mode_degrades_to_plain_without_ring_history(rec, monkeypatch, capsys):
    monkeypatch.delenv('LORA_SHARED_CAPTURE')
    monkeypatch.delenv('LORA_SHARED_CAPTURE_MB')
    monkeypatch.setattr(detector, '_mem_available_bytes', lambda: 16 * GIB)
    rec.set_history_reader(object())                       # a source without validated history
    assert rec._shared is None
    assert 'no validated ring history' in capsys.readouterr().err
    monkeypatch.setenv('LORA_SHARED_CAPTURE', '1')
    with pytest.raises(RuntimeError, match='ring history'):
        rec.set_history_reader(object())                   # the explicit form still refuses


def test_plan_scales_with_available_memory_between_floor_and_cap():
    from detector import shared_capture_plan_bytes, SHARED_CAPTURE_PLAN_MAX_BYTES
    assert shared_capture_plan_bytes(None) == SHARED_CAPTURE_PLAN_BYTES
    assert shared_capture_plan_bytes(14 * GIB) == SHARED_CAPTURE_PLAN_BYTES      # 16 GB Pi 5: the floor
    assert shared_capture_plan_bytes(16 * GIB) == SHARED_CAPTURE_PLAN_BYTES
    assert shared_capture_plan_bytes(57 * GIB) == 57 * GIB // 8                  # this laptop: ~7.1 GiB
    assert shared_capture_plan_bytes(57 * GIB) % _MIB == 0
    assert shared_capture_plan_bytes(200 * GIB) == SHARED_CAPTURE_PLAN_MAX_BYTES
    plan = shared_capture_plan({}, 57 * GIB)
    assert plan['enabled'] and plan['budget_bytes'] == 57 * GIB // 8 + HEADER_EXTRA
    assert f'budget {plan["budget_bytes"] // _MIB} MiB' in plan['reason']
    # The explicit form never consults memory.
    assert shared_capture_plan({'LORA_SHARED_CAPTURE': '1'}, 57 * GIB)['budget_bytes'] == 2048 * _MIB
