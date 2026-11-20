"""Freeze an outcome-independent, seed-disjoint small original-GRPO experiment."""
import hashlib
import json
from pathlib import Path
import re
import yaml
from run_benchmark import ROOT, write_json

SEED = 'scalecua-original-pilot-20261120-v1'

def rank(value):
    return hashlib.sha256((SEED + ':' + value).encode()).hexdigest()

def domain(index):
    return re.split(r'_[0-9a-f]{8}-', index, maxsplit=1)[0]

def select(values, count, key):
    groups = {}
    for value in values:
        groups.setdefault(key(value), []).append(value)
    groups = {k: sorted(v, key=rank) for k,v in groups.items()}
    result=[]
    while len(result)<count:
        for group in sorted(groups):
            if groups[group] and len(result)<count:
                result.append(groups[group].pop(0))
        if not any(groups.values()) and len(result)<count:
            raise ValueError('Insufficient selection candidates')
    return sorted(result)

def main():
    if (ROOT/'runs/original-rl-pilot-pipeline/state.json').exists():
        raise RuntimeError('Pilot already launched; preserve its frozen selection and configuration')
    split=json.loads((ROOT/'configs/research-splits/manifest.json').read_text())
    # At most one trajectory variant per seed in this small pool.
    by_seed={}
    for index in split['splits']['train']['task_indices']:
        seed=re.search(r'[0-9a-f]{8}-[0-9a-f-]{27,}', index).group().rstrip('-')
        by_seed.setdefault(seed,[]).append(index)
    candidates=[min(v,key=rank) for v in by_seed.values()]
    indices=select(candidates,64,domain)
    examples={d:[i for i in indices if domain(i)==d] for d in sorted({domain(i) for i in indices})}
    osmeta=json.loads((ROOT/'SCALE-CUA/osworld_eval/evaluation_examples/test_nogdrive.json').read_text())
    dev=set(split['splits']['dev']['base_ids'])
    os_candidates=[d+'/'+i for d,ids in osmeta.items() for i in ids if i in dev]
    os_eval=select(os_candidates,min(30,len(os_candidates)),lambda x:x.split('/')[0])
    scimeta=json.loads((ROOT/'SCALE-CUA/osworld_eval/evaluation_examples/test_scienceboard.json').read_text())
    sci_eval=select([d+'/'+i for d,ids in scimeta.items() for i in ids],20,lambda x:x.split('/')[0])
    target=ROOT/'configs/original-pilot';target.mkdir(exist_ok=True)
    outputs={'train-indices.json':indices,'train-examples.json':examples,'osworld-eval.json':os_eval,'scienceboard-eval.json':sci_eval}
    for name,data in outputs.items():
        path=target/name
        if path.exists() and json.loads(path.read_text())!=data:raise RuntimeError('Frozen pilot selection changed')
        write_json(path,data)
    manifest={'selection_seed':SEED,'selection':'Deterministic domain round-robin over SHA256-ranked IDs; no outcomes consulted',
              'train_count':len(indices),'frontier_runs_per_task':4,'update_opportunities':25,
              'osworld_eval_count':len(os_eval),'scienceboard_eval_count':len(sci_eval),
              'seed_disjoint_from_pilot_train':True,'selection_hashes':{n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in outputs},
              'limitations':['Small-pool original-GRPO pilot, not a full 2049-task reproduction or full benchmark score.',
                             'Released SFT/RL checkpoints may have seen the benchmark seed tasks; no pretraining-disjoint claim.']}
    write_json(target/'manifest.json',manifest)
    for suffix in ['', '0']:
        source=ROOT/f'configs/osworld-original-worker{suffix}.yaml'
        cfg=yaml.safe_load(source.read_text()); params=cfg.pop('osworld-rl')['parameters']
        params.update(name='osworld-pilot',examples_path=str(target/'train-examples.json'),
                      cache_dir=str(ROOT/f'runs/pilot-env-cache-worker{suffix or "1"}'),
                      record_dir=str(ROOT/f'runs/pilot-records-worker{suffix or "1"}'))
        (ROOT/f'configs/osworld-pilot-worker{suffix}.yaml').write_text(yaml.safe_dump({'osworld-pilot':{'module':'local_osworld.LocalOSWorldTask','parameters':params}},sort_keys=False))
    source=yaml.safe_load((ROOT/'configs/qwen3vl-osworld-original-8gpu-25updates.yaml').read_text())
    source.update(run_name='original-grpo-pilot-worker1',group_name='qwen3vl-osworld-original-grpo-pilot',save_interval=5,
                  metrics_log_dir=str(ROOT/'runs/metrics/original-grpo-pilot-worker1'))
    source['task'].update(train_tasks=['osworld-pilot'],val_tasks=['osworld-pilot'],running_mode={'osworld-pilot':'custom'})
    source['dynamic_sampling']['initial_pass_rate_file']=str(target/'initial_pass_rates.json')
    training=ROOT/'configs/qwen3vl-osworld-original-pilot-8gpu.yaml';training.write_text(yaml.safe_dump(source,sort_keys=False))
    full=json.loads((ROOT/'configs/original-rl-pipeline.json').read_text())
    full.pop('maintenance_reason',None)
    full.update(enabled=False,state_directory='runs/original-rl-pilot-pipeline',purpose='SFT report then fixed 64-task original GRPO pilot (25 update opportunities)',
                stages=['frontier_initialization','original_grpo','reproduced_checkpoint_evaluation','released_rl_reference'],
                training_config=str(training.relative_to(ROOT)),worker_config='configs/osworld-pilot-worker.yaml',
                task_name='osworld-pilot',indices_file=str(target/'train-indices.json'),
                rates_file=str(target/'initial_pass_rates.json'),frontier_output='runs/frontier-pilot-initialization',
                frontier_tasks=64,run_name='original-grpo-pilot-worker1',
                evaluation_subsets={'osworld':str(target/'osworld-eval.json'),'scienceboard':str(target/'scienceboard-eval.json')},
                evaluation_suffix='original-pilot-v1',pilot_manifest='configs/original-pilot/manifest.json')
    full['secondary_environment_worker'].update(config='configs/osworld-pilot-worker0.yaml',rpc_smoke_report='reports/distributed-osworld-pilot-worker0-smoke.json')
    full['adaptations'].update(checkpoint_interval=5,task_pool_size=64,vm_backend='TCG cold reset (validated); snapshot promotion requires concurrent reset acceptance')
    full.pop('timing_projection',None)
    write_json(ROOT/'configs/original-rl-pilot-pipeline.json',full)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
