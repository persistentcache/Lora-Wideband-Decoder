"""Verified worker records must reach the UI while later-frame search continues."""
import io
import json
from pathlib import Path
import select
from types import SimpleNamespace

import pytest

import detector
from detector import BackgroundDecoder


class StopManager(BaseException):
    pass


class Worker:
    pid = 99999999  # no real process: /proc RSS lookup safely fails

    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = SimpleNamespace(fileno=lambda: 123)
        self.killed = False
        self.dead = False

    def poll(self):
        return 0 if self.dead else (-9 if self.killed else None)

    def kill(self):
        self.killed = True


def packet(identifier='first', proto='meshtastic'):
    return {'proto': proto, 'confidence': 'verified', 'pktid': identifier,
            'from': '!ba6a95e0', 'raw_hex': identifier.encode().hex(),
            'freq_mhz': 915.0, 'sf': 12, 'bw': 62500}


def line(record):
    return '[PKT] ' + json.dumps(record, separators=(',', ':')) + '\n'


def read_packets(path):
    return [json.loads(value) for value in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def decoder(monkeypatch, tmp_path):
    monkeypatch.setenv('LORA_DECODE_WORKERS', '1')
    monkeypatch.setenv('LORA_DECODE_RAWLOG', str(tmp_path / 'raw.log'))
    monkeypatch.setenv('LORA_PKT_LOG', str(tmp_path / 'packets.jsonl'))
    monkeypatch.setenv('LORA_KEEP_CAPTURES_DIR', str(tmp_path / 'recordings'))
    monkeypatch.setenv('LORA_DECODE_WORKER_RSS_MB', '0')
    monkeypatch.setenv('LORA_DECODE_BUDGET_S', '10')
    monkeypatch.delenv('LORA_KEEP_IQ', raising=False)
    with monkeypatch.context() as startup:
        startup.setattr(BackgroundDecoder, '_manager', lambda self: None)
        instance = BackgroundDecoder()
        for thread in instance._threads:
            thread.join()
    instance._pressure_n = 0
    return instance


def run_manager(decoder, monkeypatch, tmp_path, chunks):
    """Run real dispatch/read/publication/cleanup, replacing only clock and pipe."""
    now = [1000.0]
    pending = [(1000.0 + when, data.encode() if isinstance(data, str) else data)
               for when, data in chunks]
    snapshots = []
    worker = Worker()
    path = tmp_path / 'SF7_500k_test.cf32'
    path.write_bytes(b'unchanged IQ')
    decoder.submit(str(path), path.name,
                   bucket_key=(915e6, 100.0, 60000.0, 0.1, 12, 62500))
    original_get = decoder._queue.get

    def get(*args, **kwargs):
        if decoder._queue.empty():
            raise StopManager()
        return original_get(*args, **kwargs)

    def wait(read, write, error, timeout):
        snapshots.append({'time': now[0],
                          'packets': read_packets(Path(decoder._pkt_log_path))
                          if decoder._pkt_log_path else [],
                          'refs': dict(decoder._refs),
                          'count': decoder.decoded_counts()[0],
                          'bucket_marks': decoder._diag_marks})
        limit = now[0] + timeout
        if pending and pending[0][0] <= limit:
            now[0] = max(now[0], pending[0][0])
            return [123], [], []
        now[0] = limit
        return [], [], []

    def read(fd, size):
        _, data = pending.pop(0)
        if not data:
            worker.dead = True
        return data

    monkeypatch.setattr(decoder._queue, 'get', get)
    monkeypatch.setattr(decoder, '_start_worker', lambda: worker)
    monkeypatch.setattr(select, 'select', wait)
    monkeypatch.setattr(detector.os, 'read', read)
    monkeypatch.setattr(detector.time, 'time', lambda: now[0])
    with pytest.raises(StopManager):
        decoder._manager()
    return worker, snapshots, path


def test_packet_published_while_search_continues_then_second_frame_once(
        decoder, monkeypatch, tmp_path):
    first, second = packet(), packet('second', 'meshcore')
    worker, snapshots, path = run_manager(decoder, monkeypatch, tmp_path, [
        (1, '__BEGIN__ test\n' + line(first)),
        (10, 'still searching later frames\n' + line(second) + '__END__\n'),
    ])
    early = next(s for s in snapshots if s['time'] == 1001)
    assert [p['pktid'] for p in early['packets']] == ['first']
    assert early['packets'][0]['ts'] == 1001
    assert early['count'] == 1 and early['refs'] == {str(path): 1}
    assert early['bucket_marks'] == 0
    published = read_packets(tmp_path / 'packets.jsonl')
    assert [p['pktid'] for p in published] == ['first', 'second']
    assert [p['ts'] for p in published] == [1001, 1010]
    assert decoder.decoded_counts() == (2, 2)
    raw = (tmp_path / 'raw.log').read_text()
    raw_packets = [json.loads(s[6:]) for s in raw.splitlines() if s.startswith('[PKT] ')]
    assert raw_packets == published
    assert 'still searching later frames' in raw and raw.count(' JOB ') == 1
    sidecar = json.loads((tmp_path / 'recordings' / path.with_suffix('.json').name).read_text())
    assert [p['pktid'] for p in sidecar['packets']] == ['first', 'second']
    assert decoder._refs == {} and decoder.pending() == 0
    assert decoder._queue.unfinished_tasks == 0 and not worker.killed


@pytest.mark.parametrize('ending', ['timeout', 'eof'])
def test_packet_survives_later_timeout_or_eof_without_duplicate(
        decoder, monkeypatch, tmp_path, ending):
    chunks = [(1, line(packet()))]
    if ending == 'eof':
        chunks.append((3, b''))
    worker, snapshots, path = run_manager(decoder, monkeypatch, tmp_path, chunks)
    early = next(s for s in snapshots if s['time'] == 1001)
    assert len(early['packets']) == 1 and early['refs'] == {str(path): 1}
    published = read_packets(tmp_path / 'packets.jsonl')
    assert len(published) == 1 and published[0]['ts'] == 1001
    assert decoder.decoded_counts() == (1, 1)
    raw = (tmp_path / 'raw.log').read_text()
    assert ('DECODE TIMEOUT' if ending == 'timeout' else 'WORKER DIED') in raw
    assert worker.killed == (ending == 'timeout')
    assert decoder._refs == {} and decoder.pending() == 0


def test_partial_line_has_no_effect_until_newline(decoder, monkeypatch, tmp_path):
    data = line(packet())
    cut = len(data) // 2
    _, snapshots, _ = run_manager(decoder, monkeypatch, tmp_path, [
        (1, data[:cut]), (2, data[cut:-1]), (3, '\n'), (4, '__END__\n'),
    ])
    for time in (1001, 1002):
        state = next(s for s in snapshots if s['time'] == time)
        assert state['packets'] == [] and state['count'] == 0
    complete = next(s for s in snapshots if s['time'] == 1003)
    assert complete['packets'][0]['ts'] == 1003
    assert decoder.decoded_counts() == (1, 1)


@pytest.mark.parametrize('bad', [
    '{broken', '[]', 'null', '4', '"text"',
    '{"proto":[],"raw_hex":"aa"}', '{"text":["invalid identity"]}',
])
def test_malformed_packet_has_no_effect_and_does_not_lose_next_packet(
        decoder, monkeypatch, tmp_path, bad):
    _, snapshots, path = run_manager(decoder, monkeypatch, tmp_path, [
        (1, '[PKT] ' + bad + '\n'), (2, line(packet())), (3, '__END__\n'),
    ])
    state = next(s for s in snapshots if s['time'] == 1001)
    assert state['packets'] == [] and state['count'] == 0
    assert [p['pktid'] for p in read_packets(tmp_path / 'packets.jsonl')] == ['first']
    assert decoder.decoded_counts() == (1, 1)
    sidecar = json.loads((tmp_path / 'recordings' / path.with_suffix('.json').name).read_text())
    assert len(sidecar['packets']) == 1
    assert 'JOB-ERROR' not in (tmp_path / 'raw.log').read_text()


@pytest.mark.parametrize('proto,confidence', [
    ('meshtastic', 'verified'), ('meshcore', 'verified'), ('lorawan', 'verified'),
    ('unknown', 'verified'), ('meshtastic', 'candidate'),
])
def test_all_existing_record_categories_and_repeats_keep_routing(
        decoder, monkeypatch, tmp_path, proto, confidence):
    record = packet(proto=proto)
    record['confidence'] = confidence
    _, snapshots, _ = run_manager(decoder, monkeypatch, tmp_path, [
        (1, line(record)), (2, line(record)), (3, '__END__\n'),
    ])
    assert next(s for s in snapshots if s['time'] == 1001)['count'] == 1
    published = read_packets(tmp_path / 'packets.jsonl')
    assert len(published) == 2  # UI owns result dedup; no new routing policy here.
    assert [p['confidence'] for p in published] == [confidence, confidence]
    assert decoder.decoded_counts() == (2, 1)


def test_disabled_packet_log_still_tracks_counts_and_recording_summary(
        decoder, monkeypatch, tmp_path):
    decoder._pkt_log_path = ''
    _, snapshots, path = run_manager(decoder, monkeypatch, tmp_path, [
        (1, line(packet())), (2, '__END__\n'),
    ])
    assert next(s for s in snapshots if s['time'] == 1001)['count'] == 1
    assert not (tmp_path / 'packets.jsonl').exists()
    sidecar = json.loads((tmp_path / 'recordings' / path.with_suffix('.json').name).read_text())
    assert sidecar['packets'][0]['pktid'] == 'first'
    assert decoder.decoded_counts() == (1, 1)


def test_concurrent_publication_preserves_complete_records_and_summaries(decoder, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = str(tmp_path / 'capture.cf32')

    def publish(number):
        return decoder._publish_packet_line(
            line(packet(str(number))), path, received_at=1000 + number)

    with ThreadPoolExecutor(max_workers=4) as pool:
        returned = list(pool.map(publish, range(40)))
    expected = {str(number) for number in range(40)}
    records = read_packets(tmp_path / 'packets.jsonl')
    assert len(records) == 40 and {p['pktid'] for p in records} == expected
    assert all(p['ts'] == 1000 + int(p['pktid']) for p in records)
    assert {json.loads(value[6:])['pktid'] for value in returned} == expected
    assert decoder.decoded_counts() == (40, 40)
    assert {p['pktid'] for p in decoder._cap_recs[path]} == expected
    assert len(decoder._confirmed_freqs) == 40 and decoder._diag_marks == 0


def test_log_write_failure_preserves_record_for_final_raw_output(decoder, tmp_path):
    # Existing best-effort disk-write policy: a logging failure must not kill
    # the worker's continuing search or lose its record from final raw output.
    decoder._pkt_log_path = str(tmp_path)  # a directory cannot be append-opened
    value = packet()
    value['ts'] = 1  # receiver receipt time remains authoritative
    result = decoder._publish_packet_line(line(value), 'capture', received_at=1234)
    assert json.loads(result[6:])['ts'] == 1234
    assert decoder.decoded_counts() == (1, 1)
    assert decoder._cap_recs['capture'][0]['pktid'] == 'first'
