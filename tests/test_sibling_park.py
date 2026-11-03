"""A capture whose RF bucket is already decoding is parked off the worker until
that job clears the bucket: skipped if the bucket verified, released otherwise,
released by deadline when the in-flight job never reports."""
import pytest
from detector import BackgroundDecoder


@pytest.fixture
def dec(monkeypatch):
    monkeypatch.setenv('LORA_DECODE_WORKERS', '1')
    monkeypatch.setenv('LORA_DECODE_RAWLOG', '')
    monkeypatch.setenv('LORA_PKT_LOG', '')
    monkeypatch.delenv('LORA_KEEP_IQ', raising=False)
    monkeypatch.delenv('LORA_KEEP_CAPTURES_DIR', raising=False)
    with monkeypatch.context() as startup:
        startup.setattr(BackgroundDecoder, '_manager', lambda self: None)
        decoder = BackgroundDecoder()
        for thread in decoder._threads:
            thread.join()
    return decoder


def mesh(identifier):
    return ('=== RESULT ===\nCRC: OK\n--- Meshtastic Packet ---\n'
            f'PacketID: {identifier} Flags: 0x00 (hop_limit=0 hop_start=0 hops_taken=0)\n\n')


BUCKET = (915e6, 100.0, 60e3, 0.1, 10, 250e3)       # freq, t, ftol, ttol, sf, bw
SIBLING = (915.05e6, 100.02, 60e3, 0.1, 10, 250e3)  # same instant, 50 kHz off


def capture(tmp_path, name, bucket, dec, **kw):
    path = tmp_path / name
    path.write_bytes(b'iq')
    dec.submit(str(path), name, bucket_key=bucket, **kw)
    return path


def take(dec):
    prio, _, job = dec._queue.get_nowait()
    return prio, job


def test_sibling_is_parked_and_skipped_once_its_twin_verifies(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32', BUCKET, dec)
    copy = capture(tmp_path, 'copy.cf32', SIBLING, dec)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])                 # the first copy is decoding
    prio2, job2 = take(dec)
    assert dec._park_sibling(job2, prio2, None)
    assert dec._queue.empty() and dec.pending() == 1   # parked, still counted for drain
    assert copy.exists()
    dec._compact(mesh('0x12345678'), first.name)      # the first copy verifies ...
    dec._clear_bucket_inflight(job1[0])                # ... and leaves
    assert dec._queue.empty() and dec._slow_queue.empty()
    assert not copy.exists()                           # the parked copy was skipped
    assert not dec._sib_park and dec.pending() == 0


def test_sibling_is_released_when_its_twin_found_nothing(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32', BUCKET, dec)
    copy = capture(tmp_path, 'copy.cf32', SIBLING, dec, prio=7.0)
    entries = {job[1]: (prio, job) for prio, job in (take(dec), take(dec))}
    _, job1 = entries['first.cf32']
    prio2, job2 = entries['copy.cf32']
    dec._mark_bucket_inflight(job1[0])
    assert prio2 == -7.0
    assert dec._park_sibling(job2, prio2, None)
    dec._clear_bucket_inflight(job1[0])                # nothing verified
    prio, _, released = dec._queue.get_nowait()
    assert released is job2 and prio == -7.0           # back at its own priority
    assert copy.exists()
    assert not dec._park_sibling(released, prio, None)  # no sibling in flight now


def test_unrelated_capture_is_not_parked(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32', BUCKET, dec)
    other = capture(tmp_path, 'other.cf32', (915e6, 106.0, 60e3, 0.1, 10, 250e3), dec)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])
    prio2, job2 = take(dec)
    assert not dec._park_sibling(job2, prio2, None)
    assert not dec._sib_park


def test_slow_tier_sibling_returns_to_the_slow_tier(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32', BUCKET, dec)
    copy = capture(tmp_path, 'copy.cf32', SIBLING, dec)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])
    prio2, job2 = take(dec)
    with dec._lock:
        dec._slow_released_job = job2                  # it was the released slow job
    token = dec._slow_take(job2)
    assert token is job2
    assert dec._park_sibling(job2, prio2, token)
    assert dec._slow_inflight_job is None              # reservation retired while parked
    dec._clear_bucket_inflight(job1[0])
    assert dec._queue.empty()
    assert dec._slow_queue.get_nowait() is job2


def test_parked_sibling_past_its_deadline_runs_concurrently(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32', BUCKET, dec)
    copy = capture(tmp_path, 'copy.cf32', SIBLING, dec)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])
    prio2, job2 = take(dec)
    assert dec._park_sibling(job2, prio2, None)
    dec._flush_stale_parks()                           # not stale yet
    assert dec._queue.empty()
    for key in dec._sib_park:                          # age the entry past its deadline
        dec._sib_park[key] = [(j, s, p, 0.0) for j, s, p, _d in dec._sib_park[key]]
    dec._flush_stale_parks()
    prio, _, released = dec._queue.get_nowait()
    assert released is job2 and not dec._sib_park
    # Its twin is still in flight, but a job that waited out its deadline runs anyway.
    assert not dec._park_sibling(released, prio, None)


def test_drain_releases_every_parked_sibling(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32', BUCKET, dec)
    copy = capture(tmp_path, 'copy.cf32', SIBLING, dec)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])
    prio2, job2 = take(dec)
    assert dec._park_sibling(job2, prio2, None)
    dec._flush_stale_parks(force=True)
    assert dec._queue.get_nowait()[2] is job2
    assert not dec._sib_park


def test_early_crop_sibling_parks_behind_its_twin_and_skips_once_it_verifies(dec, tmp_path):
    """Two early crops of ONE frame (the gate labelled the burst at carriers a
    few hundred Hz apart; two header checkpoints followed) decode once."""
    first = capture(tmp_path, 'a_early_1.cf32', (915.0163e6, 100.0, 60e3, 0.1, 12, 62.5e3), dec,
                    role='early', frame_key=1)
    copy = capture(tmp_path, 'a_early_2.cf32', (915.0166e6, 100.01, 60e3, 0.1, 12, 62.5e3), dec,
                   role='early', frame_key=2)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])
    prio2, job2 = take(dec)
    assert dec._park_sibling(job2, prio2, None)
    assert dec._queue.empty() and dec.pending() == 1 and copy.exists()
    dec._compact(mesh('0x12345678'), first.name, role='early')
    dec._clear_bucket_inflight(job1[0])
    assert dec._queue.empty() and dec._slow_queue.empty()
    assert not copy.exists() and dec.pending() == 0


def test_early_crop_sibling_is_released_when_its_twin_fails(dec, tmp_path):
    first = capture(tmp_path, 'b_early_1.cf32', (915.0163e6, 100.0, 60e3, 0.1, 12, 62.5e3), dec,
                    role='early', frame_key=1)
    copy = capture(tmp_path, 'b_early_2.cf32', (915.0166e6, 100.01, 60e3, 0.1, 12, 62.5e3), dec,
                   role='early', frame_key=2)
    _, job1 = take(dec)
    dec._mark_bucket_inflight(job1[0])
    prio2, job2 = take(dec)
    assert dec._park_sibling(job2, prio2, None)
    dec._clear_bucket_inflight(job1[0])                # nothing verified: the copy gets its turn
    _, _, released = dec._queue.get_nowait()
    assert released is job2 and copy.exists()

