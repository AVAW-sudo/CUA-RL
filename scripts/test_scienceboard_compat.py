import copy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scienceboard_compat import normalize_task, install_setup, patch_grass_source, GRASS_ANCHOR

ROOT = Path(__file__).resolve().parents[1]


class NormalizationTests(unittest.TestCase):
    def test_remove_only_empty_lean_cross_app_steps(self):
        base = ROOT / 'SCALE-CUA/osworld_eval/evaluation_examples/scienceboard'
        changed = 0
        removed = 0
        for path in base.glob('*/*.json'):
            task = json.loads(path.read_text())
            original = copy.deepcopy(task)
            fixed, changes = normalize_task(task)
            self.assertEqual(task, original)
            self.assertEqual(fixed['instruction'], original['instruction'])
            self.assertEqual(fixed['evaluator'], original['evaluator'])
            if changes:
                changed += 1
                self.assertEqual(task['type'], 'Lean')
                empty = [c for c in changes if c['kind'] == 'remove_empty_chimerax_open_from_lean']
                removed += len(empty)
                self.assertTrue(all(c['original']['parameters']['name'] == '' for c in empty))
                self.assertEqual(len(task['config']) - len(fixed['config']), len(empty))
                self.assertTrue(any(c['type'] == 'append_file' for c in fixed['config']))
            else:
                self.assertEqual(fixed, task)
        self.assertEqual(changed, 26)
        self.assertEqual(removed, 12)

    def test_kalgebra_wire_format_and_acknowledgement(self):
        controller = SimpleNamespace(vm_ip='127.0.0.1')
        install_setup(controller)
        response = SimpleNamespace(text='OK', raise_for_status=lambda: None)
        with patch('scienceboard_compat.requests.post', return_value=response) as post:
            controller._kalgebra_tab_setup(8000, 1)
            self.assertEqual(post.call_args.kwargs['json'], 1)
            controller._kalgebra_func_2d_setup(8000, 'x^2')
            self.assertEqual(post.call_args.kwargs['data'], 'x^2')
            controller._kalgebra_func_3d_setup(8000, 'x*y')
            self.assertEqual(post.call_args.kwargs['data'], 'x*y')
            response.text = 'ERROR'
            with self.assertRaises(RuntimeError):
                controller._kalgebra_tab_setup(8000, 1)

    def test_closed_display_is_not_successfully_hidden_toolbar(self):
        source = GRASS_ANCHOR + '\n    return {"toolbars": {"shown": frame.shown}}\n'
        tree = SimpleNamespace(Map=SimpleNamespace(layers=[]), _itemWithWindow=[])
        frame = SimpleNamespace(mainnotebook=SimpleNamespace(_tabs=SimpleNamespace(_pages=[object()])),
                                pg_panel=SimpleNamespace(maptree=tree),
                                goutput=SimpleNamespace(cmdOutput=SimpleNamespace(GetText=lambda: '')), shown=True)
        namespace = {'frame': frame}
        fixed = patch_grass_source(source)
        self.assertEqual(patch_grass_source(fixed), fixed)
        exec(fixed, namespace)
        dump = namespace['status_dump']
        self.assertFalse(dump()['toolbars']['shown'] == False)
        frame.shown = False
        self.assertTrue(dump()['toolbars']['shown'] == False)
        frame.mainnotebook._tabs._pages.clear()
        self.assertIsNone(dump()['toolbars']['shown'])
        self.assertFalse(dump()['toolbars']['shown'] == False)
        with self.assertRaises(RuntimeError):
            patch_grass_source('unknown version')


if __name__ == '__main__':
    unittest.main()
