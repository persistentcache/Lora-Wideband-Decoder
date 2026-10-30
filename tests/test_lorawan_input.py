"""Demodulator byte lists retain LoRaWAN structural hints without promotion."""
import pytest

import decoder


UNKNOWN_RECEIVED = bytes.fromhex(
    '4024821112803c0301a13a3019eb59dd9b80aa97c6c7893c5d9267bafeb4651ef14'
    'c90298f126faa72437fd50aa57300cb184ca1b648d9154c57571c084acb5e520fb038')
DEV_EUI = bytes.fromhex('1122334455667788')
JOIN_REQUEST = (b'\x00' + bytes.fromhex('0102030405060708')
                + DEV_EUI + b'\x34\x12' + b'\xaa\xbb\xcc\xdd')


def data_frame(mhdr, fctrl=0x80, options=b'', port=b'\x01'):
    return (bytes([mhdr]) + bytes.fromhex('24821112') + bytes([fctrl])
            + b'\x3c\x03' + options + port + b'\xa5' + b'\x11\x22\x33\x44')


VALID_FRAMES = [
    pytest.param(JOIN_REQUEST, {'msg_type': 'Join Request',
                              'deveui': '8877665544332211'}, id='join-request'),
    pytest.param(b'\x20' + bytes(range(16)), {'msg_type': 'Join Accept'},
                 id='join-accept'),
    pytest.param(b'\x20' + bytes(range(32)), {'msg_type': 'Join Accept'},
                 id='join-accept-cflist'),
    *[pytest.param(data_frame(mhdr), {'msg_type': name, 'devaddr': '12118224',
                                     'fcnt': 828, 'fport': 1}, id=name)
      for mhdr, name in [(0x40, 'Unconfirmed Data Up'),
                         (0x60, 'Unconfirmed Data Down'),
                         (0x80, 'Confirmed Data Up'),
                         (0xa0, 'Confirmed Data Down')]],
    *[pytest.param(b'\xc0' + bytes([kind])
                   + bytes(range(8 if kind == 1 else 3)) + DEV_EUI
                   + b'\x34\x12' + b'\xaa\xbb\xcc\xdd',
                   {'msg_type': 'Rejoin Request', 'deveui': '8877665544332211'},
                   id=f'rejoin-{kind}') for kind in (0, 1, 2)],
]


@pytest.mark.parametrize('payload,fields', VALID_FRAMES)
@pytest.mark.parametrize('container', [bytes, bytearray, list])
def test_supported_containers_preserve_fields_and_candidate_status(
        payload, fields, container, capsys):
    expected = decoder.parse_lorawan_packet(payload)
    expected_output = capsys.readouterr().out
    actual = decoder.parse_lorawan_packet(container(payload))
    assert actual == expected
    assert capsys.readouterr().out == expected_output
    assert {k: actual[k] for k in fields} == fields
    assert actual['proto'] == 'unknown'
    assert actual['hint'] == 'lorawan'
    assert actual['confidence'] == 'candidate'
    assert actual['decrypted'] is False


MALFORMED = [
    pytest.param(b'', id='empty'),
    pytest.param(bytes([JOIN_REQUEST[0] | 1]) + JOIN_REQUEST[1:], id='major'),
    pytest.param(bytes([JOIN_REQUEST[0] | 4]) + JOIN_REQUEST[1:], id='rfu'),
    pytest.param(JOIN_REQUEST[:-1], id='short-join'),
    pytest.param(JOIN_REQUEST + b'\x00', id='long-join'),
    pytest.param(b'\x20' + bytes(17), id='wrong-join-accept-length'),
    pytest.param(b'\x40' + bytes(10), id='short-data'),
    pytest.param(data_frame(0x40, fctrl=15), id='fopts-overrun'),
    pytest.param(data_frame(0x40, fctrl=1, options=b'\x02', port=b'\x00'),
                 id='fopts-and-port-zero'),
    pytest.param(b'\xc0', id='missing-rejoin-type'),
    pytest.param(b'\xc0\x00' + bytes(16), id='short-rejoin-zero'),
    pytest.param(b'\xc0\x01' + bytes(17), id='wrong-rejoin-one-length'),
    pytest.param(b'\xc0\x02' + bytes(18), id='long-rejoin-two'),
    pytest.param(b'\xc0\x03' + bytes(17), id='invalid-rejoin-type'),
    pytest.param(b'\xe0' + bytes(22), id='proprietary'),
]


@pytest.mark.parametrize('payload', MALFORMED)
@pytest.mark.parametrize('container', [bytes, bytearray, list])
def test_malformed_structure_is_rejected(payload, container):
    assert decoder.parse_lorawan_packet(container(payload)) is None


@pytest.mark.parametrize('payload', [None, 23, 33, True, 23.0, 'not byte data'])
def test_scalar_or_text_input_cannot_become_a_synthetic_frame(payload):
    with pytest.raises(TypeError):
        decoder.parse_lorawan_packet(payload)


@pytest.mark.parametrize('bad', [-1, 256, 'x', None])
def test_invalid_list_elements_cannot_become_a_candidate(bad):
    payload = list(JOIN_REQUEST)
    payload[8] = bad
    with pytest.raises((TypeError, ValueError)):
        decoder.parse_lorawan_packet(payload)


def test_native_crc_capture_dispatch_preserves_raw_phy_and_structural_hint(
        monkeypatch, capsys):
    payload = UNKNOWN_RECEIVED[:-2]
    assert len(payload) == 66
    assert decoder.check_crc(UNKNOWN_RECEIVED, 66, strict=True) == (
        True, 'xor-CCITT-LE')
    emitted = []
    rf = {'freq_mhz': 912.8997, 'rssi': 15, 'sf': 7, 'bw': 125000}
    monkeypatch.setattr(decoder, '_PROTO_ENABLED', {'lorawan', 'raw_lora'})
    monkeypatch.setattr(decoder, '_LORA_REGION', 'US915')
    monkeypatch.setattr(decoder, '_emit_pkt', lambda rec: emitted.append(dict(rec)))

    def no_meshtastic(*args):
        pytest.fail('LoRaWAN structural candidate reached Meshtastic handler')

    handled, winner, via = decoder._dispatch_nonmt_or_meshtastic(
        list(payload), {'rf': rf, 'crc_present': True}, 0x34, rf, 7, 125000, 1,
        no_meshtastic, False, True)
    assert '[DISPATCH EXC]' not in capsys.readouterr().out
    assert handled and via == 'nonmt_candidate'
    assert emitted == [winner]
    assert winner == {
        'proto': 'unknown', 'hint': 'lorawan', 'confidence': 'candidate',
        'msg_type': 'Unconfirmed Data Up', 'decrypted': False, 'on_grid': True,
        'devaddr': '12118224', 'fcnt': 828, 'fport': 1,
        'summary': 'DevAddr 12118224 · FCnt 828 · FPort 1 · ADR',
        'raw_hex': payload.hex(), 'sf': 7, 'bw': 125000, 'cr': 1,
        'sync': '0x34', 'freq_mhz': 912.8997, 'rssi': 15,
    }
