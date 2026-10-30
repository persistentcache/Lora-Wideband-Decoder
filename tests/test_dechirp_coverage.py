"""Light AST-only coverage checks: no detector import, numpy, DSP, or radio."""
import ast
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'src/detector.py'
TREE = ast.parse(SOURCE.read_text())


class DechirpCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = {'patience_trial_family', 'patience_detection_family', 'patience_cap_params',
                 '_capture_duration_s', '_rf_footprint_covered', '_covered_dechirp_channels',
                 '_report_dechirp_coverage'}
        functions = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
        missing = names-{node.name for node in functions}
        assert not missing, 'Missing production coverage behavior: '+repr(sorted(missing))
        trials = next(node.value for node in TREE.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'PATIENCE_TRIALS' for t in node.targets))
        cls.scope = dict(math=math, os=SimpleNamespace(environ={}),
                         sys=SimpleNamespace(stderr=io.StringIO()), PATIENCE_TRIALS=ast.literal_eval(trials))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), cls.scope)

    def split(self, channels, rate=10e6):
        return self.scope['_covered_dechirp_channels'](channels, 915e6, rate)

    def test_live_edge_trial_is_uncovered(self):
        chan = (919994165.3496335-915e6, 12, 62500., True)
        covered, rejected = self.split([chan])
        self.assertEqual(covered, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['source'], 'patience')
        self.assertAlmostEqual(rejected[0]['rf_high_hz'], 920025415.3496335)

    def test_both_edges_rates_and_exact_boundaries(self):
        for rate in (8e6,10e6):
            for sign in (-1,1):
                edge = sign*(rate/2-62500/2)
                for delta, accepted in ((-4e-6,True),(0,True),(.5e-6,True),(4e-6,False)):
                    with self.subTest(rate=rate, sign=sign, delta=delta):
                        covered, rejected = self.split([(edge+sign*delta,12,62500.,True)], rate)
                        self.assertEqual(bool(covered), accepted)
                        self.assertEqual(bool(rejected), not accepted)

    def test_configured_and_mixed_trial_rungs_preserve_order(self):
        channels = [(0.,7,250000.),(4.95e6,10,62500.,True),(4.95e6,12,125000.,True),
                    (-4.99e6,12,62500.),(-2e6,11,125000.,True)]
        covered, rejected = self.split(channels)
        self.assertEqual([tuple(c[:4]) for c in covered], [channels[0],channels[1],channels[4]])
        self.assertEqual([r['source'] for r in rejected], ['patience','configured'])

    def test_feasible_family_drives_bandwidth_duration_and_alias_labels(self):
        covered, rejected = self.split([(4.95e6,10,62500.,True)])
        self.assertFalse(rejected)
        self.assertEqual(covered[0][4], ((10,62500.),))
        detection = dict(sf=10,bw=62500,patience_trial=True,
                         _patience_covered_family=covered[0][4])
        self.assertEqual(self.scope['patience_detection_family'](detection), ((10,62500.),))
        self.assertEqual(self.scope['patience_cap_params'](detection), (62500.,1024/62500))
        self.assertEqual(self.scope['_capture_duration_s'](detection),
                         self.scope['_capture_duration_s'](dict(sf=10,bw=62500)))
        alias_loop = next(node for node in ast.walk(TREE) if isinstance(node, ast.For)
            and ast.unparse(node.target) == '(_fsf, _fbw)')
        labels = eval(compile(ast.Expression(alias_loop.iter),str(SOURCE),'eval'),
                      dict(self.scope,_src=detection))
        self.assertEqual(labels, ((10,62500.),))

    def test_full_coverage_and_existing_detections_keep_original_family(self):
        for sf,bw in self.scope['PATIENCE_TRIALS']:
            original = self.scope['patience_trial_family'](sf,bw)
            channel = (0.,sf,bw,True)
            covered,rejected = self.split([channel])
            self.assertEqual(covered,[channel])
            self.assertFalse(rejected)
            detection=dict(sf=sf,bw=bw,patience_trial=True)
            self.assertEqual(self.scope['patience_detection_family'](detection),original)
            self.assertEqual(self.scope['patience_cap_params'](detection),
                (max(b for _,b in original),max((1<<s)/b for s,b in original)))

    def test_matches_actual_capture_guard_and_accepts_export_crossing_edge(self):
        capture = ast.parse((SOURCE.parent/'capture_timeline.py').read_text())
        guard = next(node.test for node in ast.walk(capture) if isinstance(node,ast.If)
                     and 'math.isfinite(float(det[' in ast.unparse(node.test))
        code=compile(ast.Expression(guard),'<capture_guard>','eval')
        for rate in (8e6,10e6):
            for offset,bw in [(rate/2-40000,62500),(-rate/2+40000,62500),
                              (rate/2-10000,62500),(0,0),(math.nan,62500),(math.inf,62500)]:
                frequency=915e6+offset
                expected=eval(code,dict(math=math,det={'freq_hz':frequency},signal_half=bw/2,
                                      self=SimpleNamespace(center=915e6,rate=rate)))
                self.assertEqual(self.scope['_rf_footprint_covered'](frequency,bw,915e6,rate),expected)

    def test_direct_forced_filter_keeps_ordinary_peak_reference(self):
        detect=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='detect_preamble')
        assignment=next(node for node in detect.body if isinstance(node,ast.Assign)
            and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
            and node.value.func.id=='_covered_dechirp_channels')
        peak=object()
        scope=dict(self.scope,center_hz=915e6,wb_fs=10e6,cached_peaks=[peak],
                   dechirp_chans=[(4.994165e6,12,62500.,True)])
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(SOURCE),'exec'),scope)
        self.assertEqual(scope['dechirp_chans'],[])
        self.assertEqual(scope['cached_peaks'],[peak])

    def test_partial_family_metadata_is_attached_only_when_present(self):
        detect=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='detect_preamble')
        attach=next(node for node in ast.walk(detect) if isinstance(node,ast.If)
                    and ast.unparse(node.test)=='len(_dech) > 4')
        for channel, expected in [((4.95e6,10,62500.,True,((10,62500.),)),True),
                                  ((0.,10,62500.,True),False)]:
            scope=dict(_dech=channel,_ld=[{}])
            exec(compile(ast.Module(body=[attach],type_ignores=[]),str(SOURCE),'exec'),scope)
            self.assertEqual('_patience_covered_family' in scope['_ld'][-1],expected)

    def test_required_math_import_exists_in_production_module(self):
        self.assertTrue(any(isinstance(node,ast.Import) and any(alias.name=='math'
            and alias.asname is None for alias in node.names) for node in TREE.body))

    def test_empty_per_window_pool_channels_do_not_restore_startup_channels(self):
        dispatch=next(node for node in ast.walk(TREE) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and node.func.attr=='dispatch')
        channels=next(keyword.value for keyword in dispatch.keywords if keyword.arg=='dechirp_chans')
        task=eval(compile(ast.Expression(channels),str(SOURCE),'eval'),{'_win_chans':[]})
        pool=ast.parse((SOURCE.parent/'detect_pool.py').read_text())
        selection=next(node for node in ast.walk(pool) if isinstance(node,ast.IfExp)
            and ast.unparse(node.test)=='task_chans is not None')
        actual=eval(compile(ast.Expression(selection),'<pool>','eval'),
                    dict(task_chans=task,dechirp_chans=[(4.99e6,12,62500)]))
        self.assertEqual(actual,[])

    def test_main_coverage_diagnostics_are_aggregated_and_rate_limited(self):
        main=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        nodes=[node for node in ast.walk(main) if
            (isinstance(node,ast.For) and ast.unparse(node.target)=='_excluded') or
            (isinstance(node,ast.AugAssign) and ast.unparse(node.target)=='_coverage_total') or
            (isinstance(node,ast.If) and (ast.unparse(node.test)=='_coverage_rejected'
                or ast.unparse(node.test).startswith('_coverage_total != _coverage_reported')))]
        code=compile(ast.Module(body=sorted(nodes,key=lambda n:n.lineno),type_ignores=[]),str(SOURCE),'exec')
        clock=[100.]
        reported=[]
        scope=dict(_coverage_counts={'patience':0,'configured':0},_coverage_total=0,
            _coverage_reported=0,_coverage_report_at=0.,_coverage_representative=None,
            time=SimpleNamespace(monotonic=lambda:clock[0]),
            _report_dechirp_coverage=lambda item,total,counts,where:reported.append((total,dict(counts))))
        for tick in range(11):
            clock[0]=100.+tick*.5
            scope['_coverage_rejected']=[{'source':'patience'},{'source':'configured'}]
            exec(code,scope)
        self.assertEqual(reported,[(2,{'patience':1,'configured':1}),(22,{'patience':11,'configured':11})])

    def test_family_metadata_recomputes_when_input_width_changes(self):
        covered,_=self.split([(3.95e6,10,62500.,True)],8e6)
        self.assertEqual(covered[0][4],((10,62500.),))
        wider,rejected=self.split(covered,10e6)
        self.assertEqual(wider,[(3.95e6,10,62500.,True)])
        self.assertFalse(rejected)

    def test_uncovered_configured_channel_neither_adds_nor_coalesces_real_peaks(self):
        main=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        nodes=[node for node in ast.walk(main) if
            (isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in
                ('(_configured_chans, _coverage_rejected)','_covered_dechirp_src')) or
            (isinstance(node,ast.If) and ast.unparse(node.test)=='_covered_dechirp_src and (not _win_backlogged)')]
        self.assertEqual(len(nodes),3)
        peaks=[(4090,8,30.,0.),(4093,4,20.,0.)]
        scope=dict(self.scope,_dechirp_src=[(919994165.3496335,62500.,12)],
            _live_center_mhz=915.,a=SimpleNamespace(rate=10e6),_win_backlogged=False,
            _gate_peaks=list(peaks))
        exec(compile(ast.Module(body=sorted(nodes,key=lambda n:n.lineno),type_ignores=[]),str(SOURCE),'exec'),scope)
        self.assertEqual(scope['_gate_peaks'],peaks)
        self.assertEqual(scope['_configured_chans'],[])
        self.assertEqual(len(scope['_coverage_rejected']),1)


if __name__ == '__main__':
    unittest.main()
