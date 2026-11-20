"""Compare exactly the frozen pilot evaluation IDs, preserving missing scores."""
import argparse
import hashlib
import json
from pathlib import Path
from run_benchmark import ROOT, write_json
from write_evaluation_report import generate


def report(final=False):
    config = json.loads((ROOT / 'configs/original-rl-pilot-pipeline.json').read_text())
    paths = []
    for original in config['sft_campaigns']:
        source = Path(original) / 'campaign.json'
        state = json.loads(source.read_text())
        benchmark = state['protocol']['benchmark']
        subset = Path(config['evaluation_subsets'][benchmark])
        selected = json.loads(subset.read_text())
        state['tasks'] = {key:state['tasks'][key] for key in selected}
        state['protocol'] = {**state['protocol'], 'scope':'Frozen pilot evaluation subset; extracted without resampling',
            'source_campaign':str(source), 'source_campaign_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'subset_file':str(subset), 'subset_sha256':hashlib.sha256(subset.read_bytes()).hexdigest()}
        path = ROOT / 'runs/pilot-report-inputs' / Path(original).name
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / 'campaign.json', state)
        paths.append(str(path))
    for name in ['reproduced-rl','osworld-rl-reference']:
        paths += [str(path) for path in (ROOT/'runs/benchmarks').glob(name+'__*-'+config['evaluation_suffix'])
                  if (path/'campaign.json').exists()]
    result = generate(paths,ROOT/'reports/original-grpo-pilot-comparison',final=final)
    result['pilot_manifest'] = json.loads((ROOT/config['pilot_manifest']).read_text())
    result['training_progress'] = json.loads((ROOT/'reports/original-grpo-progress.json').read_text()) if (ROOT/'reports/original-grpo-progress.json').exists() else {'status':'not_started'}
    write_json(ROOT/'reports/original-grpo-pilot-comparison.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--final',action='store_true')
    print(json.dumps({'status':report(p.parse_args().final)['status']}))
