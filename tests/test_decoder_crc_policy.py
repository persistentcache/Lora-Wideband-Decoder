"""Research CRC coincidences cannot authorize operational packet publication."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest

import decoder


# Exact received bytes from the preserved extended-run capture, including
# the two received CRC bytes. The alternate match is real, not a mock.
FALSE_RECEIVED = bytes.fromhex(
    'f24ff6ff80956a9294c574e5213108e0615f6726c0f66e841178ccc220602d01'
    'cffaeaaa0b54c6b92652bfedc1621877f1fd2098e18e98c3ae0826400a')
VALID_RECEIVED = bytes.fromhex(
    'ffffffffe0956aba94c57400213100e061536a0cc0cf8e6e4175814a818c2d11'
    'd9f6a7caac74d4d4e3d08e0c8162237cd4cc4778e1ed93417fad2c983d')
STAGES = ('primary', 'drift', 'ldro', 'frac', 'newton')


def native_received(payload):
    """Independent bitwise fixture CRC, no production CRC helper/table."""
    crc = 0
    for byte in payload[:-2]:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xffff
    crc ^= (payload[-2] << 8) | payload[-1]
    return bytes(payload) + crc.to_bytes(2, 'little')


@pytest.fixture(scope='module')
def production_blocks():
    tree = ast.parse(Path(decoder.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_attempt')
    start = next(i for i, n in enumerate(fn.body)
                 if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == 'soft_decode_payload')
    end = next(i for i, n in enumerate(fn.body[start:], start)
               if isinstance(n, ast.If) and ast.unparse(n.test) == '_HARNESS_OUT')
    # Supply RF measurements/demod bytes; keep actual production branching,
    # retry adoption, CRC and dispatch. The numerical CRC is never stubbed.
    primary = [copy.deepcopy(n) for n in fn.body[start:end]
               if not (isinstance(n, ast.FunctionDef)
                       and n.name == '_measure_payload_frac')]
    early_start = next(i for i, n in enumerate(fn.body)
                       if isinstance(n, ast.Assign) and any(
                           isinstance(t, ast.Name) and t.id == '_early_handled'
                           for t in n.targets))
    early_end = next(i for i, n in enumerate(fn.body[early_start:], early_start)
                     if isinstance(n, ast.If)
                     and ast.unparse(n.test) == '_early_handled')
    final_start = next(i for i, n in enumerate(fn.body)
                       if isinstance(n, ast.If)
                       and '[CRC GATE]' in ast.unparse(n))
    final_end = next(i for i, n in enumerate(fn.body[final_start:], final_start)
                     if isinstance(n, ast.If) and 'not _handled' in ast.unparse(n.test)) + 1

    def compile_body(body):
        return compile(ast.fix_missing_locations(ast.Module(
            body=body, type_ignores=[])), decoder.__file__, 'exec')

    final = ast.FunctionDef(name='final_dispatch', args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=copy.deepcopy(fn.body[final_start:final_end]), decorator_list=[])
    return (compile_body(primary), compile_body(copy.deepcopy(
        fn.body[early_start:early_end])), compile_body([final]))


def run_primary(blocks, monkeypatch, raw, stage='primary', *, crc_present=True,
                later_raw=None):
    # Keep later trials available after a rejected alternative so the test
    # also catches a false success prematurely stopping recovery.
    monkeypatch.setenv('LORA_LDRO_FALLBACK', '1' if stage == 'ldro' else '0')
    monkeypatch.delenv('LORA_DRIFT_LEGACY', raising=False)
    target_call = {'primary': 1, 'drift': 2, 'ldro': 2, 'frac': 2, 'newton': 4}[stage]
    damaged = raw[:-2] + b'\x5a\xa5'
    assert not decoder.check_crc(damaged, len(raw) - 2)[0]
    calls = []

    def demod(*args, **kwargs):
        calls.append(kwargs)
        value = raw if len(calls) == target_call else damaged
        if later_raw is not None and len(calls) == target_call + 1:
            value = later_raw
        return list(value), list(value[:-2]), [], [0] * (len(value) * 2)

    env = dict(vars(decoder), soft_decode_payload=demod,
               _measure_payload_frac=lambda tau: (
                   (.395, 64) if stage in ('frac', 'newton') else (0., 64)),
               crc_present=crc_present, payload_len=len(raw) - 2,
               need_bytes=len(raw), tau_frac=0., diag_out={},
               _drift_applied=stage == 'drift', _drift_phase=np.zeros(16),
               iq1=np.ones(16, np.complex64), sf=7, N=128,
               ppm_pay=7, pay_levels=128, pay_bin_group=1,
               print=lambda *args, **kwargs: None)
    exec(blocks[0], env)
    return env, calls


def route(blocks, env, monkeypatch, *, protocols, unknown=False, sync=0x2b):
    emitted = []
    monkeypatch.setattr(decoder, '_PROTO_ENABLED', set(protocols))
    monkeypatch.setattr(decoder, '_PROTO_UNKNOWN', unknown)
    monkeypatch.setattr(decoder, '_emit_pkt', lambda rec: emitted.append(dict(rec)))
    monkeypatch.setattr(decoder, '_load_keys', lambda: None)
    monkeypatch.setattr(decoder, '_decrypt_mt_pki_dm', lambda *args: None)
    monkeypatch.setattr(decoder, '_KNOWN_NODES', set())
    monkeypatch.setattr(decoder, '_DECODED_SET', set())
    monkeypatch.setattr(decoder, '_GATE_HDR_TUPLES', set())
    monkeypatch.delenv('LORA_UNKNOWN_REPORT', raising=False)
    env.update(_PROTO_ENABLED=set(protocols), _PROTO_UNKNOWN=unknown,
               _GATE_HDR_TUPLES=set(),
               # On the US915 125 kHz uplink grid (channel 63 = 914.9 MHz):
               # off-grid LoRaWAN-shaped frames are no longer candidates, and
               # these tests exercise dispatch policy, not the channel grid.
               name='SF7_125k_914.9004MHz_1000ksps_pwr52_fixture.cf32',
               _hw_fp=None, bw=125000, cr=4, best_pre_len=16,
               _sync_word=sync, preamble_start=0, preamble_bin=0)
    exec(blocks[1], env)
    if not env['_early_handled']:
        env['_is_meshtastic'] = ('meshtastic' in protocols and sync == 0x2b)
        exec(blocks[2], env)
        result = env['final_dispatch']()
    else:
        result = 'early'
    return emitted, result


def test_explicit_research_retains_alternate_crc_identification():
    assert decoder.check_crc(FALSE_RECEIVED, 59, strict=False) == (True, 'std-IBM-BE')
    assert decoder.check_crc(FALSE_RECEIVED, 59, strict=True) == (False, 'FAIL')
    assert native_received(VALID_RECEIVED[:-2]) == VALID_RECEIVED
    assert decoder.check_crc(VALID_RECEIVED, 59, strict=True) == (True, 'xor-CCITT-LE')


@pytest.mark.parametrize('stage', STAGES)
def test_alternate_crc_cannot_authorize_any_primary_or_rescue_route(
        production_blocks, monkeypatch, stage):
    env, calls = run_primary(production_blocks, monkeypatch, FALSE_RECEIVED, stage)
    emitted, result = route(production_blocks, env, monkeypatch,
                            protocols={'meshtastic', 'meshcore', 'lorawan',
                                       'radiohead', 'raw_lora', 'lora_aprs'},
                            unknown=True)
    assert emitted == []
    assert result[0] == 'FAIL'
    assert not env['crc_ok'] and not env['_clean_crc']
    if stage in ('frac', 'newton'):
        assert len(calls) == 4  # alternative match cannot stop the remaining trial


def test_fractional_rescue_continues_past_collision_to_native_received_frame(
        production_blocks, monkeypatch):
    env, calls = run_primary(production_blocks, monkeypatch, FALSE_RECEIVED,
                             'frac', later_raw=VALID_RECEIVED)
    emitted, _ = route(production_blocks, env, monkeypatch,
                       protocols={'meshtastic'}, unknown=True)
    assert len(calls) == 3
    assert env['crc_method'] == 'xor-CCITT-LE'
    assert env['raw_bytes'] == list(VALID_RECEIVED)
    assert env['tau_frac'] == pytest.approx(-.605)
    assert [record['raw_hex'] for record in emitted] == [VALID_RECEIVED[:-2].hex()]


@pytest.mark.parametrize('stage', STAGES)
def test_native_crc_preserves_exact_verified_frame_through_each_route(
        production_blocks, monkeypatch, stage):
    env, _ = run_primary(production_blocks, monkeypatch, VALID_RECEIVED, stage)
    emitted, result = route(production_blocks, env, monkeypatch,
                            protocols={'meshtastic'})
    assert env['crc_ok'] and env['_clean_crc'] and result == 'early'
    assert len(emitted) == 1
    assert emitted[0]['raw_hex'] == VALID_RECEIVED[:-2].hex()
    assert emitted[0]['pktid'] == '0x0074c594'
    assert emitted[0]['confidence'] == 'verified'
    assert emitted[0]['text'] == 'LWDFIX28-E36B 0009 CUSTOM_SF7_125_CR8'


def test_native_crc_unknown_key_unicast_remains_a_candidate(
        production_blocks, monkeypatch):
    # Use the same undecryptable bytes but a valid native received CRC:
    # eligibility depends on PHY integrity, never on candidate confidence.
    raw = native_received(FALSE_RECEIVED[:-2])
    env, _ = run_primary(production_blocks, monkeypatch, raw)
    emitted, _ = route(production_blocks, env, monkeypatch,
                       protocols={'meshtastic'})
    assert len(emitted) == 1
    assert emitted[0]['confidence'] == 'candidate'
    assert not emitted[0]['decrypted'] and emitted[0]['enc_hex']
    assert emitted[0]['raw_hex'] == raw[:-2].hex()


@pytest.mark.parametrize('protocols,payload,expected', [
    ({'lora_aprs'}, b'\x3c\xff\x01N0CALL>APRS:>fixture status', 'lora_aprs'),
    (set(), b'\x13\x92\xae\x7d\x88\x04\x9d', 'unknown'),
])
def test_native_crc_preserves_other_protocol_and_unknown_evidence(
        production_blocks, monkeypatch, protocols, payload, expected):
    env, _ = run_primary(production_blocks, monkeypatch, native_received(payload))
    emitted, result = route(production_blocks, env, monkeypatch,
                            protocols=protocols, unknown=True, sync=0x12)
    assert result == 'early' and len(emitted) == 1
    assert emitted[0]['proto'] == expected
    assert emitted[0]['raw_hex'] == payload.hex()
    if expected == 'unknown':
        assert emitted[0]['from'] is None and emitted[0]['to'] is None


def test_crc_absent_lorawan_dispatch_remains_available(production_blocks, monkeypatch):
    payload = b'\x20' + bytes(range(16))  # encrypted Join Accept, 17-byte form
    env, _ = run_primary(production_blocks, monkeypatch,
                         native_received(payload), crc_present=False)
    emitted, _ = route(production_blocks, env, monkeypatch,
                       protocols={'lorawan'}, sync=0x34)
    assert not env['crc_ok'] and not env['_clean_crc']
    assert len(emitted) == 1
    assert emitted[0]['hint'] == 'lorawan'
    assert emitted[0]['raw_hex'] == payload.hex()
