#!/usr/bin/env python3
"""sidekiq_rx.py — stream IQ from an Epiq Solutions SideKiq to stdout in sc16 format.

Wraps the SDK's rx_samples binary and corrects the I/Q byte order so the output
is compatible with detector.py (and the rest of the pipeline).

rx_samples packs each IQ pair as a 32-bit little-endian word:
    bits 31:16 = I  (upper)
    bits 15:0  = Q  (lower)
Reading as consecutive int16 LE gives: [Q, I, Q, I, ...]
detector expects:                   [I, Q, I, Q, ...] (s[0::2]=I, s[1::2]=Q)
So every pair must be swapped.

Usage (mirrors soapy_rx.py interface so detector can call it the same way):
    python3.10 sidekiq_rx.py -f 915000000 -s 28000000 -b 28000000 [-g 50] [-c 1]
"""

import sys
import os
import argparse
import subprocess
import signal
import numpy as np

PREBUILT = '/ro/home/sidekiq/sidekiq_sdk_current/prebuilt_apps/x86_64.gcc'
RX_SAMPLES = os.path.join(PREBUILT, 'rx_samples')

# ~2 seconds of samples per rx_samples invocation at 28 Msps.
# Looping gives continuous streaming with only a brief reinit gap between blocks.
WORDS_PER_RUN = 56_000_000

_proc = None


def _cleanup(sig=None, frame=None):
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        _proc.wait()
    sys.exit(0)


signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)


def main():
    global _proc

    p = argparse.ArgumentParser(description='SideKiq IQ capture → sc16 stdout')
    p.add_argument('-f', '--freq', type=int, required=True,
                   help='Center frequency in Hz')
    p.add_argument('-s', '--rate', type=int, default=28_000_000,
                   help='Sample rate in Hz')
    p.add_argument('-b', '--bw', type=int, default=None,
                   help='Bandwidth in Hz (defaults to sample rate)')
    p.add_argument('-g', '--gain', type=str, default=None,
                   help='RX gain in dB, or omit for AGC')
    p.add_argument('-c', '--card', type=int, default=1,
                   help='SideKiq card index (0 or 1, default 1)')
    args = p.parse_args()

    bw = args.bw if args.bw else args.rate

    cmd = [
        RX_SAMPLES,
        '-c', str(args.card),
        '-f', str(args.freq),
        '-r', str(args.rate),
        '-b', str(bw),
        '-d', '/dev/stdout',
        '-w', str(WORDS_PER_RUN),
    ]
    if args.gain and args.gain.strip().lower() not in ('', 'auto', 'agc'):
        try:
            cmd += ['-g', str(int(float(args.gain)))]
        except ValueError:
            pass

    out = sys.stdout.buffer
    CHUNK_WORDS = 32_768          # 32 K IQ pairs per read = 128 KB
    CHUNK_BYTES = CHUNK_WORDS * 4  # 4 bytes per IQ pair

    try:
        while True:
            _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
            while True:
                raw = _proc.stdout.read(CHUNK_BYTES)
                if not raw:
                    break
                # Swap Q/I → I/Q: reshape to (N,2) int16, reverse columns, flatten
                iq = np.frombuffer(raw, dtype='<i2').reshape(-1, 2)
                iq = np.ascontiguousarray(iq[:, ::-1])
                out.write(iq.tobytes())
                out.flush()
            _proc.wait()
    except BrokenPipeError:
        pass
    finally:
        _cleanup()


if __name__ == '__main__':
    main()
