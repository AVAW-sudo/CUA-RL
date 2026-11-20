import json
import re
import unittest
import yaml
from run_benchmark import ROOT

class PilotProtocolTests(unittest.TestCase):
    def test_pilot_preserves_original_algorithm_and_sampling_budget(self):
        full=yaml.safe_load((ROOT/'configs/qwen3vl-osworld-original-8gpu-25updates.yaml').read_text())
        pilot=yaml.safe_load((ROOT/'configs/qwen3vl-osworld-original-pilot-8gpu.yaml').read_text())
        for k in ['train','rollout','actor','loss','advantage','max_steps','task_adv_norm']:
            self.assertEqual(full[k],pilot[k],k)
        for k in ['images_to_keep','min_removal_threshold','max_total_len','max_turns','task_kwargs']:
            self.assertEqual(full['task'][k],pilot['task'][k],k)
        a=dict(full['dynamic_sampling']);b=dict(pilot['dynamic_sampling'])
        a.pop('initial_pass_rate_file');b.pop('initial_pass_rate_file')
        self.assertEqual(a,b)
        self.assertEqual(pilot['save_interval'],5)

    def test_selected_train_and_osworld_eval_seeds_are_disjoint(self):
        indices=json.loads((ROOT/'configs/original-pilot/train-indices.json').read_text())
        selected=json.loads((ROOT/'configs/original-pilot/osworld-eval.json').read_text())
        seeds=[re.search(r'[0-9a-f]{8}-[0-9a-f-]{27,}',i).group().rstrip('-') for i in indices]
        self.assertEqual(len(set(indices)),64)
        self.assertEqual(len(set(seeds)),64)
        self.assertFalse(set(seeds)&{s.split('/')[1] for s in selected})
        self.assertEqual(len(selected),30)
        self.assertEqual(len(json.loads((ROOT/'configs/original-pilot/scienceboard-eval.json').read_text())),20)

if __name__=='__main__':unittest.main()
