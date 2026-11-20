"""Validate equivalent SFT replicas and the narrowly scoped routing upgrade."""
import hashlib
import json
from pathlib import Path


def canonical_command(command):
    command = list(command)
    if '--port' not in command:
        raise ValueError('Inference command must explicitly pin its port')
    index = command.index('--port')
    del command[index:index + 2]
    return command


def validate_services(services, primary):
    if not 1 <= len(services) <= 8:
        raise ValueError('An inference pool must contain one to eight replicas')
    endpoints = set()
    identities = ['checkpoint', 'model_id', 'model_revision', 'model_dir', 'served_model_name', 'runtime_compatibility']
    for service in services:
        if any(service.get(key) != primary.get(key) for key in identities):
            raise ValueError('Inference pool mixes different checkpoint identities')
        if canonical_command(service['command']) != canonical_command(primary['command']):
            raise ValueError('Replica inference settings differ beyond the HTTP port')
        if service['hostname'] != primary['hostname']:
            raise ValueError('This pool must remain on the same authorized worker')
        if service['base_url'] in endpoints:
            raise ValueError('Duplicate inference endpoint')
        endpoints.add(service['base_url'])
    return services


def load_services(pool_path, primary):
    spec = json.loads(Path(pool_path).read_text())
    if spec['checkpoint'] != primary['checkpoint']:
        raise ValueError('Pool checkpoint does not match the selected evaluation arm')
    services = [json.loads(Path(path).read_text()) for path in spec['service_states']]
    return validate_services(services, primary)


ROUTING_EDITS = [
    ("    args = parser.parse_args()\n",
     "    parser.add_argument('--service-state', type=Path, help='Pinned inference replica state selected by the coordinator')\n"
     "    args = parser.parse_args()\n"),
    ("    state = json.loads(service_path.read_text())\n",
     "    service_path = args.service_state or service_path\n"
     "    state = json.loads(service_path.read_text())\n"),
    ("    manifest['adapter_sha256'] = {}\n",
     "    manifest['inference_service'] = {'state_file': str(service_path.resolve()),\n"
     "        'state_sha256': hashlib.sha256(service_path.read_bytes()).hexdigest(),\n"
     "        'hostname': state['hostname'], 'gpus': state.get('gpus'), 'base_url': state['base_url']}\n"
     "    manifest['adapter_sha256'] = {}\n"),
]


def routing_upgrade_source(source):
    for old, new in ROUTING_EDITS:
        if source.count(old) != 1:
            raise ValueError('Unknown evaluation launcher; routing-only upgrade refused')
        source = source.replace(old, new)
    return source


def validate_routing_upgrade(old_directory, new_directory):
    for name in ['eval_qemu_provider.py', 'qemu_runtime.py', 'scienceboard_compat.py']:
        if (old_directory / name).read_bytes() != (new_directory / name).read_bytes():
            raise ValueError(f'Routing upgrade must not alter the environment or verifier: {name}')
    old = (old_directory / 'evaluate_sft.py').read_text()
    new = (new_directory / 'evaluate_sft.py').read_text()
    if routing_upgrade_source(old) != new:
        raise ValueError('Evaluation changes exceed the explicit service-routing edits')
    return {'scope': 'Only service-state selection and manifest provenance added to evaluate_sft.py',
            'old_launcher_sha256': hashlib.sha256(old.encode()).hexdigest(),
            'new_launcher_sha256': hashlib.sha256(new.encode()).hexdigest(),
            'environment_and_verifier_adapters_unchanged': True}
