"""Low-rate slow seeds preserve physical width and legacy wideband behavior."""
import numpy as np
import pytest
import detector


def plateau(width, *, peak=2300, floor=-30.0, power=25.0):
    psd = np.full(4096, floor, dtype=np.float32)
    left = peak - width // 2
    psd[left:left + width] = power
    return psd, peak


@pytest.mark.parametrize('rate', [20_000_000, 40_000_000])
def test_twenty_and_higher_msps_preserve_original_width_decisions(rate):
    for width in range(1, 257):
        psd, peak = plateau(width)
        assert detector._slow_seed_is_narrow(psd, peak, rate, -30.0) == (width <= 16)


@pytest.mark.parametrize('rate,last_allowed', [
    (5_000_000, 64), (8_000_000, 40), (10_000_000, 32),
    (12_500_000, 25), (16_000_000, 20), (20_000_000, 16),
    (30_720_000, 16), (40_000_000, 16), (61_440_000, 16),
])
def test_low_rate_boundary_scales_without_tightening_legacy_wideband_rates(rate, last_allowed):
    inside, peak = plateau(last_allowed)
    outside, _ = plateau(last_allowed + 1)
    assert detector._slow_seed_is_narrow(inside, peak, rate, -30.0)
    assert not detector._slow_seed_is_narrow(outside, peak, rate, -30.0)


@pytest.mark.parametrize('rate', [5_000_000, 10_000_000, 20_000_000])
def test_broad_125khz_plateau_does_not_gain_slow_scan_eligibility(rate):
    width = round(125000 / (rate / 4096.0))
    psd, peak = plateau(width)
    assert not detector._slow_seed_is_narrow(psd, peak, rate, -30.0)


def test_peak_relative_width_ignores_strong_signal_low_shoulders():
    psd, peak = plateau(100, power=10.0)
    left = peak - 26 // 2
    psd[left:left + 26] = 30.0
    before = psd.copy()
    assert detector._slow_seed_is_narrow(psd, peak, 10_000_000, -30.0)
    assert not detector._slow_seed_is_narrow(psd, peak, 20_000_000, -30.0)
    np.testing.assert_array_equal(psd, before)


def test_contour_equal_to_threshold_still_stops_the_edge_walk():
    psd, peak = plateau(100, power=13.0)
    left = peak - 16 // 2
    psd[left:left + 16] = 25.0
    assert detector._slow_seed_is_narrow(psd, peak, 20_000_000, -30.0)
    psd[left - 1] = 13.01
    assert not detector._slow_seed_is_narrow(psd, peak, 20_000_000, -30.0)


@pytest.mark.parametrize('rate', [10_000_000, 20_000_000, 40_000_000])
@pytest.mark.parametrize('offset', [36000, 400000])
def test_actual_welch_psd_of_sf12_62500_chirps_is_eligible(rate, offset):
    # Synthesize only the segments consumed by the real 1-second Welch path.
    # This avoids storing a 10/20-million-sample buffer or opening any device.
    bw = 62500.0
    symbol_s = 4096 / bw
    rng = np.random.default_rng(20261027)

    def segment(start, count):
        t = (start + np.arange(count, dtype=np.float64)) / rate
        within = np.remainder(t, symbol_s)
        phase = -bw / 2 * within + bw / (2 * symbol_s) * within**2 + offset * t
        signal = np.exp(2j * np.pi * phase)
        noise = 0.01 * (rng.standard_normal(count) + 1j * rng.standard_normal(count))
        return (signal + noise).astype(np.complex64)

    psd = detector._welch_psd_from(rate, segment, nfft=4096, n_avg=50)
    peaks = detector.find_peaks(psd, thresh_db=5, fres_hz=rate / 4096.0)
    peak = min(peaks, key=lambda p: abs((p[0] - 2048) * rate / 4096 - offset))
    assert abs((peak[0] - 2048) * rate / 4096 - offset) < bw / 2
    assert detector._slow_seed_is_narrow(psd, int(peak[0]), rate, float(np.median(psd)))
