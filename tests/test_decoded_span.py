"""A successful frame mask must preserve a later packet in the same IQ file."""
import ast
from pathlib import Path

import numpy as np
import pytest

import decoder


@pytest.fixture(scope='module', params=['early_dispatch', 'full_path'])
def successful_span(request):
    # Execute each production success-return calculation. The expensive RF
    # demodulation is covered by retained-IQ replay; no mask policy is copied.
    tree = ast.parse(Path(decoder.__file__).read_text())
    attempt = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_decode_attempt')
    sites = {}
    for owner in ast.walk(attempt):
        body = getattr(owner, 'body', None)
        if not isinstance(body, list):
            continue
        for ret in body:
            if not (isinstance(ret, ast.Return)
                    and isinstance(ret.value, ast.Tuple)
                    and isinstance(ret.value.elts[0], ast.Constant)
                    and ret.value.elts[0].value == 'OK'):
                continue
            calculations = [n for n in body[:body.index(ret)]
                            if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id in
                                    {'n_pay_syms', 'total_syms', 'pkt_len_samples'}
                                    for t in n.targets)]
            name = 'full_path' if owner is attempt else 'early_dispatch'
            shell = ast.parse('def span(sf, N, ppm, ppm_pay, cr, need_bytes, '
                              'preamble_start, data_start, preamble_bin): pass')
            shell.body[0].body = calculations + [ret]
            ast.fix_missing_locations(shell)
            namespace = dict(vars(decoder))
            exec(compile(shell, decoder.__file__, 'exec'), namespace)
            sites[name] = namespace['span']
    assert set(sites) == {'early_dispatch', 'full_path'}
    return sites[request.param]


@pytest.mark.parametrize('sf,ppm_pay,cr,need_bytes,preamble_syms,body_syms', [
    # Header carries SF-7 payload nibbles. Each FEC block then supplies
    # ppm_pay nibbles in CR+4 symbols; these are hand-counted block fixtures.
    (12, 10, 4, 99, 16, 168),  # PL97+CRC: 5 header nibbles +20 blocks of10
    (12, 10, 4, 99, 32, 168),  # actual long preamble must extend the span
    (12, 12, 4, 99, 16, 144),  # accepted forced-LDRO-off:17 blocks of12
    (7, 7, 1, 39, 16, 68),     # PL37+CRC:12 blocks of7 at CR4/5
    (12, 10, 4, 7, 8, 16),     # 14 nibbles minus5 in header: one block
    (12, 10, 4, 9, 8, 24),     # same PL7 with CRC: two blocks
    (12, 10, 4, 239, 32, 392), # maximum PL237+CRC:48 blocks
])
def test_span_uses_actual_header_payload_blocks_and_preamble(
        successful_span, sf, ppm_pay, cr, need_bytes, preamble_syms, body_syms):
    n = 1 << sf
    preamble_start = 12345
    # Seven samples model the header grid's measured timing adjustment.
    data_start = preamble_start + int((preamble_syms + 4.25) * n) + 7
    result = successful_span(sf, n, sf - 2, ppm_pay, cr, need_bytes,
                             preamble_start, data_start, 17)
    expected_end = data_start + (body_syms + 1) * n  # existing one-symbol margin
    assert result == ('OK', preamble_start, expected_end - preamble_start, 17)


def test_success_mask_retains_complete_second_burst(successful_span):
    # Geometry measured in retained006 at the decoder's BW-rate time axis.
    # The former ~36-second mask erased both full12.34-second bursts.
    n = 4096
    first_start, first_end = 60000, 831250
    second_start, second_end = 853750, 1625000
    iq = np.zeros(2004112, dtype=np.float32)
    iq[first_start:first_end] = 1
    iq[second_start:second_end] = 2
    result = successful_span(12, n, 10, 10, 4, 99, first_start,
                             first_start + int(20.25 * n), 17)

    # Execute the actual caller's two mask statements, including its cap at EOF.
    tree = ast.parse(Path(decoder.__file__).read_text())
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_process_file_label')
    mask = next(n for n in ast.walk(process) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'mask_len > 0'
                and any(isinstance(s, ast.Assign)
                        and any(isinstance(t, ast.Subscript)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == 'iq1_work' for t in s.targets)
                        for s in n.body))
    body = [s for s in mask.body if isinstance(s, ast.Assign)]
    namespace = dict(iq1_work=iq, mask_start=result[1], mask_len=result[2])
    exec(compile(ast.Module(body=body, type_ignores=[]), decoder.__file__, 'exec'),
         namespace)
    assert not np.any(iq[first_start:first_end])
    assert np.all(iq[second_start:second_end] == 2)

