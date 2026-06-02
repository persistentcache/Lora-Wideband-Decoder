#!/usr/bin/env python3
"""Run the detector pipeline headlessly (no web UI).

Forwards all arguments to the detector.  Examples:

    # Live decode from bladeRF at 28 Msps:
    bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000; \\
        set bandwidth rx 28000000; set agc rx on; \\
        rx config file=/dev/stdout format=bin n=0; rx start; rx wait" \\
    | python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode

    # Offline replay of a recorded capture:
    python3 run/headless.py -f recording.sc16 \\
        -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode

For per-flag detail, run with --help.
"""
import os, sys, runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

runpy.run_path(os.path.join(_SRC, 'detector.py'), run_name='__main__')
