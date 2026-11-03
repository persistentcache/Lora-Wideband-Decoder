"""Carrier-local width estimation must not wrap a PSD search across Nyquist."""
import ast
from pathlib import Path

import numpy as np
import pytest
import detector


@pytest.fixture(scope='module')
def measure_width():
    # Exercise the production nested branch and its actual edge walker. This
    # isolates a deterministic PSD geometry from expensive RF peak discovery.
    tree = ast.parse(Path(detector.__file__).read_text())
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'detect_preamble')
    walk = next(n for n in ast.walk(entry) if isinstance(n, ast.FunctionDef)
                and n.name == '_walk_edge')
    branch = next(n for n in ast.walk(entry) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == '_psw is not None and abs(cfo_hz) > bw / 4.0')
    code = compile(ast.Module(body=[walk, branch], type_ignores=[]),
                   detector.__file__, 'exec')

    def run(psd, center_bin):
        scope = dict(np=np, _psw=psd, _rate1m=1000000, bw=500000,
                     cfo_hz=(center_bin-2048)*244.140625, _w_car=None)
        exec(code, scope)
        return scope['_w_car']
    return run


@pytest.mark.parametrize('center,begin,end,width', [
    (64, 48, 81, 8056.640625),
    (4030, 4014, 4047, 8056.640625),
    (1000, 980, 1021, 10009.765625),
    (41, 20, 61, 10009.765625),
    (4054, 4034, 4075, 10009.765625),
])
def test_carrier_search_measures_only_nearby_energy(measure_width, center, begin, end, width):
    psd = np.ones(4096)
    psd[begin:end] = 1000
    # Stronger energy on the opposite edge must not enter the local search.
    psd[4088 if center < 2048 else 8] = 1e9
    assert measure_width(psd, center) == width


@pytest.mark.parametrize('center,begin,end', [(64, 0, 81), (4030, 4014, 4096)])
def test_edge_touching_contour_keeps_width_abstention(measure_width, center, begin, end):
    psd = np.ones(4096)
    psd[begin:end] = 1000
    assert measure_width(psd, center) is None


@pytest.mark.parametrize('center', [40, 4055])
def test_carrier_outside_existing_guard_keeps_abstention(measure_width, center):
    psd = np.ones(4096)
    psd[max(0, center-10):center+11] = 1000
    assert measure_width(psd, center) is None
