import pytest
from throughput import ThroughputMeter


RATE = 3_000_000


def test_startup_time_is_excluded():
    meter = ThroughputMeter(RATE)
    assert meter.sample(0, 0, 0, 0) is None
    assert meter.sample(30, RATE, RATE, 0) is None
    stat = meter.sample(35, 6 * RATE, 6 * RATE, 0)
    assert stat.processed_msps == pytest.approx(3)
    assert stat.input_msps == pytest.approx(3)
    assert stat.warning is None


def test_slow_input_does_not_blame_processing():
    meter = ThroughputMeter(RATE)
    meter.sample(10, RATE, RATE, 0)
    first = meter.sample(15, 3 * RATE, 3 * RATE, 0)
    assert first.warning is None
    second = meter.sample(20, 5 * RATE, 5 * RATE, 0)
    assert '[INPUT WARNING]' in second.warning
    assert 'CPU' not in second.warning
    assert 'KEEP-UP WARNING' not in second.warning


def test_processing_deficit_requires_a_backlog():
    meter = ThroughputMeter(RATE)
    meter.sample(0, RATE, RATE, 0)
    meter.sample(5, 6 * RATE, 3 * RATE, 3 * RATE)
    stat = meter.sample(10, 11 * RATE, 5 * RATE, 6 * RATE)
    assert '[KEEP-UP WARNING]' in stat.warning
    assert stat.backlog_s == pytest.approx(6)


def test_recovery_clears_warning_and_can_warn_again():
    meter = ThroughputMeter(RATE)
    meter.sample(0, RATE, RATE, 0)
    meter.sample(5, 3 * RATE, 3 * RATE, 0)
    assert meter.sample(10, 5 * RATE, 5 * RATE, 0).warning
    recovered = meter.sample(15, 10 * RATE, 10 * RATE, 0)
    assert recovered.recovered
    assert recovered.warning is None
    meter.sample(20, 12 * RATE, 12 * RATE, 0)
    assert meter.sample(25, 14 * RATE, 14 * RATE, 0).warning


def test_skips_do_not_count_as_processed_samples_or_recovery():
    meter = ThroughputMeter(RATE)
    meter.sample(0, RATE, RATE, 0)
    stat = meter.sample(5, 6 * RATE, 3 * RATE, 0, skipped=3 * RATE)
    assert stat.processed_msps == pytest.approx(1.2)
    assert not stat.recovered
    assert meter.sample(10, 11 * RATE, 8 * RATE, 0, skipped=3 * RATE).recovered


def test_counter_reset_and_zero_time_do_not_produce_invalid_rates():
    meter = ThroughputMeter(RATE)
    meter.sample(5, 10, 10, 0)
    assert meter.sample(5, 20, 20, 0) is None
    assert meter.sample(6, 0, 0, 0) is None
    meter.sample(10, 10, 10, 0)
    stat = meter.sample(15, 5 * RATE + 10, 5 * RATE + 10, 0)
    assert stat.processed_msps == pytest.approx(3)


def test_loss_before_first_measurement_clears_after_recovery():
    meter = ThroughputMeter(RATE)
    # The detector reports this loss before the meter gets its first sample.
    assert meter.sample(30, 3 * RATE, RATE, 0, skipped=2 * RATE) is None
    stat = meter.sample(35, 8 * RATE, 6 * RATE, 0, skipped=2 * RATE)
    assert stat.recovered
    assert stat.warning is None
