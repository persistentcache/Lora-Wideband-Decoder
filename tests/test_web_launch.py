"""Exercise the documented launcher, including its module search path."""
import os
from pathlib import Path
import subprocess
import sys


def test_web_launcher_can_load_app_and_show_help(tmp_path):
    # --help never accesses radio hardware. Supply only the optional native
    # driver import so this launcher regression also runs on development hosts.
    (tmp_path / 'SoapySDR.py').write_text('# No radio operations for --help.\n')
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / 'run' / 'web.py'), '--help'],
        cwd=repo, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert '--config' in result.stdout
