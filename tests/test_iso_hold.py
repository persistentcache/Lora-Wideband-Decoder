"""The time-isolated twin of a fast-SF primary waits for the plain job on its
file: skipped once that job published a verified packet, released otherwise,
never stranded."""
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


def capture(tmp_path, name='SF7_125k_915.0MHz_1000ksps_pwr50_shared_full_1_x_001.cf32'):
    path = tmp_path / name
    path.write_bytes(b'iq')
    return path


def queued_names(dec):
    return sorted(job[1] for _, _, job in list(dec._queue.queue))


def test_twin_waits_for_the_plain_job_and_is_skipped_once_it_verified(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, role='shared_full')
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full')
    assert queued_names(dec) == [path.name]            # only the plain job runs now
    assert dec.pending() == 2                          # the hold still counts for drain
    assert dec._refs[str(path)] == 2
    _, _, plain = dec._queue.get_nowait()
    # The plain job publishes a packet (any capture role records it) ...
    dec._compact(mesh('0x12345678'), path.name, role='shared_full')
    dec._plain_done(plain[0])
    dec._ref_dec_and_maybe_unlink(plain[0])
    # ... so the twin never runs and the file is released.
    assert dec._queue.empty() and dec._slow_queue.empty()
    assert dec.pending() == 0
    assert str(path) not in dec._refs
    assert not path.exists()


def test_twin_is_released_when_the_plain_job_found_nothing(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, role='shared_full')
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full', prio=-50.0)
    _, _, plain = dec._queue.get_nowait()
    dec._plain_done(plain[0])
    dec._ref_dec_and_maybe_unlink(plain[0])
    _, _, twin = dec._queue.get_nowait()
    assert twin[2:4] == (4, 170) and twin[0] == str(path)
    assert path.exists()                               # the twin still owns a reference
    dec._ref_dec_and_maybe_unlink(twin[0])
    assert not path.exists()


def test_twin_released_to_the_slow_tier_keeps_its_tier(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, role='shared_full')
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full', slow=True)
    _, _, plain = dec._queue.get_nowait()
    dec._plain_done(plain[0])
    assert dec._queue.empty()
    assert dec._slow_queue.get_nowait()[2:4] == (4, 170)


def test_twin_submitted_after_a_verified_plain_job_is_skipped_at_once(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, role='shared_full')
    _, _, plain = dec._queue.get_nowait()
    dec._compact(mesh('0x12345678'), path.name, role='shared_full')
    dec._plain_done(plain[0])
    dec._ref_dec_and_maybe_unlink(plain[0])
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full')
    assert dec._queue.empty() and dec.pending() == 0
    assert not path.exists()


def test_twin_without_a_plain_job_runs_as_before(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full')
    assert queued_names(dec) == [path.name]
    # One-sided relay searches are not twins and are never held.
    other = capture(tmp_path, 'other.cf32')
    dec.submit(str(other), other.name, role='shared_full')
    dec.submit(str(other), other.name, relay_after_syms=4, role='shared_full')
    assert queued_names(dec).count(other.name) == 2


def test_requeued_plain_job_keeps_the_twin_held(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, role='shared_full')
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full')
    _, _, plain = dec._queue.get_nowait()
    dec._plain_enqueued(plain[0])                      # REQUEUE-SLOW of the plain job
    dec._plain_done(plain[0])                          # the first plain run ends
    assert dec._queue.empty()                          # still held for the retry
    dec._plain_done(plain[0])                          # the retry ends without a packet
    assert dec._queue.get_nowait()[2][2:4] == (4, 170)


def test_stale_hold_valve_and_drain_never_strand_a_twin(dec, tmp_path):
    path = capture(tmp_path)
    dec.submit(str(path), path.name, role='shared_full')
    dec.submit(str(path), path.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full')
    dec._queue.get_nowait()                            # plain job vanishes without reporting
    dec._flush_stale_iso_holds()                       # not stale yet
    assert dec._queue.empty() and dec.pending() == 1
    dec._flush_stale_iso_holds(max_age=0.0)
    assert dec._queue.get_nowait()[2][2:4] == (4, 170)
    assert not dec._iso_hold and not dec._plain_open


def test_a_duplicate_of_an_already_published_frame_still_counts_as_verified(dec, tmp_path):
    first = capture(tmp_path, 'first.cf32')
    second = capture(tmp_path)
    dec.submit(str(second), second.name, role='shared_full')
    dec.submit(str(second), second.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full')
    dec._compact(mesh('0x12345678'), first.name, role='shared_full')
    out = dec._compact(mesh('0x12345678'), second.name, role='shared_full')
    assert '[DUP]' in out                              # publication dedup suppressed it ...
    _, _, plain = dec._queue.get_nowait()
    dec._plain_done(plain[0])                          # ... but the twin is still skipped
    assert dec._queue.empty() and dec._slow_queue.empty()


# ------------------------------------------------------------ full-crop hold
def early_full(tmp_path, key=39):
    early = capture(tmp_path, f'SF12_62.50k_915.0MHz_500ksps_pwr20_early_41_x_002.cf32')
    full = capture(tmp_path, f'SF12_62.50k_915.0MHz_500ksps_pwr20_shared_full_{key}_x_001.cf32')
    return early, full


def test_full_crop_waits_for_its_early_sibling_and_is_skipped_once_it_verified(dec, tmp_path):
    early, full = early_full(tmp_path)
    dec.submit(str(early), early.name, role='early', frame_key=39, prio=float('inf'))
    dec.submit(str(full), full.name, role='shared_full', frame_key=39)
    dec.submit(str(full), full.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full', frame_key=39)
    assert queued_names(dec) == [early.name]           # full and its twin both wait
    assert dec.pending() == 3
    _, _, job = dec._queue.get_nowait()
    dec._compact(mesh('0x12345678'), early.name, role='early')
    dec._plain_done(job[0])
    dec._ref_dec_and_maybe_unlink(job[0])
    assert dec._queue.empty() and dec._slow_queue.empty() and dec.pending() == 0
    assert not full.exists() and not early.exists()
    assert not dec._full_hold and not dec._iso_hold and not dec._early_open


def test_full_crop_is_released_when_its_early_sibling_found_nothing(dec, tmp_path):
    early, full = early_full(tmp_path)
    dec.submit(str(early), early.name, role='early', frame_key=39)
    dec.submit(str(full), full.name, role='shared_full', frame_key=39, prio=-5.0)
    _, _, job = dec._queue.get_nowait()
    dec._plain_done(job[0])
    dec._ref_dec_and_maybe_unlink(job[0])
    _, _, released = dec._queue.get_nowait()
    assert released[0] == str(full) and released[2:4] == (None, None)
    assert full.exists()


def test_full_crop_without_a_known_early_sibling_runs_at_once(dec, tmp_path):
    early, full = early_full(tmp_path)
    dec.submit(str(full), full.name, role='shared_full', frame_key=39)
    assert queued_names(dec) == [full.name]
    # The early crop arriving later runs too (it is the latency path).
    dec.submit(str(early), early.name, role='early', frame_key=39)
    assert sorted(queued_names(dec)) == sorted([full.name, early.name])


def test_full_crop_submitted_after_a_verified_early_is_skipped_at_once(dec, tmp_path):
    early, full = early_full(tmp_path)
    dec.submit(str(early), early.name, role='early', frame_key=39)
    _, _, job = dec._queue.get_nowait()
    dec._compact(mesh('0x12345678'), early.name, role='early')
    dec._plain_done(job[0])
    dec._ref_dec_and_maybe_unlink(job[0])
    dec.submit(str(full), full.name, role='shared_full', frame_key=39)
    dec.submit(str(full), full.name, relay_after_syms=4, relay_before_syms=170,
               role='shared_full', frame_key=39)
    assert dec._queue.empty() and dec.pending() == 0 and not full.exists()


def test_other_frames_and_default_captures_are_not_held(dec, tmp_path):
    early, full = early_full(tmp_path)
    other = capture(tmp_path, 'SF12_62.50k_915.0MHz_500ksps_pwr20_shared_full_57_x_003.cf32')
    dec.submit(str(early), early.name, role='early', frame_key=39)
    dec.submit(str(other), other.name, role='shared_full', frame_key=57)   # another frame
    legacy = capture(tmp_path, 'legacy.cf32')
    dec.submit(str(legacy), legacy.name)                                   # default role
    assert sorted(queued_names(dec)) == sorted([early.name, other.name, legacy.name])


def test_stale_full_hold_is_released_by_the_valve(dec, tmp_path):
    early, full = early_full(tmp_path)
    dec.submit(str(early), early.name, role='early', frame_key=39)
    dec.submit(str(full), full.name, role='shared_full', frame_key=39)
    dec._queue.get_nowait()                            # early job vanishes unreported
    dec._flush_stale_iso_holds()
    assert dec._queue.empty()
    dec._flush_stale_iso_holds(max_age=0.0)
    assert dec._queue.get_nowait()[2][0] == str(full)
    assert not dec._full_hold and not dec._early_open
