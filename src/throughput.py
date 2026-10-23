"""Recent input and processing rates for live receiver health."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Throughput:
    input_msps: float
    processed_msps: float
    backlog_s: float
    warning: str | None = None
    recovered: bool = False


class ThroughputMeter:
    """Measure sample deltas after input starts, with two-interval warnings."""

    def __init__(self, rate):
        self.rate = float(rate)
        self._previous = None
        self._issue = None
        self._consecutive = 0
        self._warning_active = False

    @property
    def started(self):
        return self._previous is not None

    def sample(self, now, received, processed, backlog, skipped=0):
        current = (now, received, processed, skipped)
        previous = self._previous
        if received <= 0 or (previous is not None and (
                received < previous[1] or processed < previous[2]
                or skipped < previous[3])):
            self._previous = None
            self._issue = None
            self._consecutive = 0
            self._warning_active = False
            return None
        if previous is None:
            self._previous = current
            # The first read can already have emitted a buffer-loss warning.
            self._warning_active = skipped > 0
            return None
        duration = now - previous[0]
        if duration <= 0:
            return None
        self._previous = current
        incoming = (received - previous[1]) / duration
        processing = (processed - previous[2]) / duration
        backlog_s = max(0, backlog) / self.rate
        issue = None
        if skipped > previous[3]:
            issue = 'loss'
        elif backlog_s >= 1 and processing < incoming * 0.95:
            issue = 'processing'
        elif incoming < self.rate * 0.95 and backlog_s < 1:
            issue = 'input'
        self._consecutive = self._consecutive + 1 if issue == self._issue else 1
        self._issue = issue
        warning = None
        recovered = issue is None and self._warning_active and backlog_s < 1
        if recovered:
            self._warning_active = False
        if issue == 'loss':
            # The detector already reports the exact skip at its source.
            self._warning_active = True
        elif issue and self._consecutive >= 2:
            self._warning_active = True
            if issue == 'input':
                warning = (f'[INPUT WARNING] Receiving {incoming / 1e6:.1f} Msps '
                           f'of {self.rate / 1e6:g} Msps requested. '
                           'Inspect the radio and USB capture logs.')
            else:
                warning = (f'[KEEP-UP WARNING] Processing {processing / 1e6:.1f} Msps '
                           f'of {incoming / 1e6:.1f} Msps received. '
                           f'backlog {backlog_s:.1f}s. '
                           'Reduce the sample rate or processing load.')
        return Throughput(incoming / 1e6, processing / 1e6,
                          backlog_s, warning, recovered)
