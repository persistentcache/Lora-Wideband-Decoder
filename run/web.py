#!/usr/bin/env python3
"""Start the lora_ml web UI.

Reads lora.toml, launches the SDR → detector pipeline, and serves the web UI on
the host:port configured in [web].

Usage:
    python3 run/web.py
    python3 run/web.py --config /path/to/lora.toml --port 5000
"""
import os, sys, runpy

# Locate the repo root and the backend package, then hand off to src/web/app.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, 'src')
for _p in (_SRC, os.path.join(_SRC, 'web')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

runpy.run_path(os.path.join(_SRC, 'web', 'app.py'), run_name='__main__')
