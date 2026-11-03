"""A decrypted Meshtastic Data message is verified only when it is one a node
could have sent: whole buffer parsed, known fields, a text portnum carries
text.  AES-CTR has no integrity check, so this parse is the integrity check."""
import pytest

import decoder
from decoder import data_message_valid, parse_protobuf_strict, aes_ctr_decrypt, MESH_AES_KEY


def test_well_formed_messages_pass():
    assert data_message_valid(b'\x08\x01\x12\x05hello')[1]                 # TEXT_MESSAGE_APP "hello"
    assert data_message_valid(b'\x08\x04\x12\x03abc')[1]                   # NODEINFO with bytes
    assert data_message_valid(b'\x08\x04')[1]                              # payload-less non-text port
    fixed = b'\x08\x01\x12\x02hi' + b'\x35' + (1234).to_bytes(4, 'little')  # request_id as fixed32
    assert data_message_valid(fixed)[1]
    assert data_message_valid(b'\x08\x01\x12\x02hi\x48\x01')[1]            # bitfield (9) varint


def test_damaged_or_foreign_messages_fail():
    assert not data_message_valid(b'')[1]
    assert not data_message_valid(b'\x08\x01\x12\x4b' + b'x' * 10)[1]      # payload length past the end
    assert not data_message_valid(b'\x08\x01')[1]                          # text port without text
    assert not data_message_valid(b'\x08\x01\x12\x00')[1]                  # text port with empty text
    assert not data_message_valid(b'\x08\x01\x12\x02\xff\xfe')[1]          # not UTF-8
    assert not data_message_valid(b'\x08\x7f\x12\x02hi')[1]                # unknown portnum
    assert not data_message_valid(b'\x08\x01\x12\x02hi\x58\x01')[1]        # unknown field 11
    assert not data_message_valid(b'\x08\x01\x12\x02hi\x13\x01')[1]        # wire type 3 (group)
    assert not data_message_valid(b'\x0a\x02hi')[1]                        # portnum with wire type 2
    assert not parse_protobuf_strict(b'\x08\x81')[1]                       # varint runs off the end


def _packet(plaintext, sender=b'\xe0\x95\x6a\xba', pkt_id=b'\x70\xfc\x18\xcd'):
    header = b'\xff\xff\xff\xff' + sender + pkt_id + b'\x21\x31\x00\x00'
    nonce = bytearray(16); nonce[0:4] = pkt_id; nonce[8:12] = sender
    return header + aes_ctr_decrypt(MESH_AES_KEY, bytes(nonce), plaintext)


def test_bit_error_variant_of_a_real_text_message_is_not_verified(capsys, monkeypatch):
    monkeypatch.setattr(decoder, '_load_keys', lambda: None)
    text = b'LWDFIX28-B995 0007 CUSTOM_SF12_62P5 LONG-0123456789012345678901234567890123'
    good = _packet(b'\x08\x01\x12' + bytes([len(text)]) + text)
    decoder.parse_meshtastic_packet(good, _clean_crc=True)
    out = capsys.readouterr().out
    assert 'PacketID: 0xcd18fc70' in out and 'Message: "LWDFIX28-B995 0007' in out
    assert '"confidence":"verified"' in out
    # The live case: a CRC-16 coincidence with the payload-length byte damaged
    # (ciphertext offset 19 = plaintext byte 3) decrypts to portnum 1 and no
    # payload.  Three flipped bits, CRC passed, the old rule said verified.
    bad = bytearray(good); bad[16 + 3] ^= 0x40
    decoder.parse_meshtastic_packet(bytes(bad), _clean_crc=True)
    out = capsys.readouterr().out
    assert 'decrypt failed' in out and 'PacketID:' not in out
    assert '"confidence":"verified"' not in out
    assert '"confidence":"candidate"' in out            # header-only, as any undecryptable clean frame
