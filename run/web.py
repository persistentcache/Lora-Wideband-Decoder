#!/usr/bin/env python3
"""Start the LORA Wideband Decoder web UI.

Reads lora.toml, launches the SDR → detector pipeline, and serves the web UI on
the host:port configured in [web].

Usage:
    python3 run/web.py
    python3 run/web.py --config /path/to/lora.toml --port 5000
"""
import os, sys, runpy

# Modules the whole pipeline needs in ONE interpreter.  SoapySDR is the catch:
# its bindings install via apt (`python3-soapysdr`) as a SWIG C-extension bound to
# the SYSTEM python and are NOT pip-installable — a Homebrew / pyenv / conda
# `python3` cannot import them.
_NEED = ('SoapySDR', 'flask', 'numpy', 'scipy')


def _missing(py=None):
    """Which of _NEED can't be imported.  py=None → check THIS interpreter
    in-process; otherwise probe another interpreter by path via a subprocess."""
    if py is None:
        out = []
        for m in _NEED:
            try:
                __import__(m)
            except Exception:
                out.append(m)
        return out
    import subprocess
    chk = ('import sys\n'
           'bad=[]\n'
           'for m in %r:\n'
           '    try: __import__(m)\n'
           '    except Exception: bad.append(m)\n'
           'sys.exit(1 if bad else 0)\n' % (_NEED,))
    try:
        ok = subprocess.run([py, '-c', chk], timeout=20,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        ok = False
    return [] if ok else list(_NEED)


def _ensure_full_stack_interpreter():
    """Guarantee the server runs under a Python that has the COMPLETE stack, so the
    SDR capture subprocess can `import SoapySDR` no matter which `python3` the user
    typed.  Without this, Start "immediately stops": the server boots (flask is in
    the user's python) but the capture child dies on `No module named 'SoapySDR'`.

    If the current interpreter is missing anything, hand the whole process over
    (os.execv) to one that has all of it — preferring the distro python the
    bindings are built against.  If none qualifies, print the apt command and exit
    rather than serving a UI whose Start button silently fails."""
    if not _missing():          # current interpreter is complete → done
        return                  # capture/detector reuse sys.executable, so they inherit it
    if os.environ.get('LORA_REEXECED'):     # already switched once — don't loop
        _fail()
    import shutil
    seen = {os.path.realpath(sys.executable)}
    for cand in ('/usr/bin/python3', shutil.which('python3'),
                 '/usr/bin/python3.13', '/usr/bin/python3.12', '/usr/bin/python3.11'):
        if not cand:
            continue
        rp = os.path.realpath(cand)
        if rp in seen or not os.path.exists(cand):
            continue
        seen.add(rp)
        if not _missing(cand):
            os.environ['LORA_REEXECED'] = '1'   # mark so the child can't re-loop
            sys.stderr.write('lora_web: switching to %s (has SoapySDR + full stack)\n' % cand)
            os.execv(cand, [cand] + sys.argv)   # replace this process in place
    _fail()


def _fail():
    sys.stderr.write(
        '\nLORA Wideband Decoder: no Python interpreter found with the full runtime\n'
        'stack (need: %s).\n\n'
        "SoapySDR's bindings install via apt and bind to the SYSTEM python only —\n"
        'they are not pip-installable:\n\n'
        '    sudo apt install python3-soapysdr soapysdr-module-all\n\n'
        'Then launch the server with that interpreter:\n\n'
        '    /usr/bin/python3 run/web.py\n\n' % ', '.join(_NEED))
    sys.exit(1)


_ensure_full_stack_interpreter()

# Locate the repo root and the backend package, then hand off to src/web/app.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, 'src')
for _p in (_SRC, os.path.join(_SRC, 'web')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

runpy.run_path(os.path.join(_SRC, 'web', 'app.py'), run_name='__main__')
