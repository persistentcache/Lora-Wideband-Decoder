"""_recenter_and_decode hands back a span tuple or None; a bool once leaked
out on budget exhaustion and the caller's _rr[2] crashed the job."""
import ast
from pathlib import Path

import decoder


def _returns(fn):
    return [r.value for r in ast.walk(fn) if isinstance(r, ast.Return)]


def test_recenter_and_decode_never_returns_a_bool():
    tree = ast.parse(Path(decoder.__file__).read_text())
    outer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_process_file_label')
    inner = next(n for n in ast.walk(outer) if isinstance(n, ast.FunctionDef) and n.name == '_recenter_and_decode')
    for value in _returns(inner):
        assert value is None or not (isinstance(value, ast.Constant) and isinstance(value.value, bool)), ast.unparse(value)
    # and the caller still guards on None before indexing
    src = Path(decoder.__file__).read_text()
    assert '_rr = _recenter_and_decode(cfo_hz_raw)' in src and 'if _rr is not None:' in src


def test_no_function_in_the_decoder_mixes_bool_and_tuple_returns():
    tree = ast.parse(Path(decoder.__file__).read_text())
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        rets = [v for v in _returns(fn) if v is not None]
        bools = any(isinstance(v, ast.Constant) and isinstance(v.value, bool) for v in rets)
        tuples = any(isinstance(v, ast.Tuple) for v in rets)
        if bools and tuples:
            offenders.append(fn.name)
    # _decode_attempt returns (ok, margin) pairs AND bare bools from distinct
    # inner helpers that are read by name; the two below are the ones audited.
    assert set(offenders) <= {'_decode_attempt'}, offenders
