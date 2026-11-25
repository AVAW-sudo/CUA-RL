"""Auditable compatibility fixes for the released ScienceBoard task runtime.

Keep instructions, evaluator predicates and model actions unchanged. Every
normalization is recorded separately from the original task configuration.
"""
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import time
import requests


GRASS_PATH = '/usr/lib/grass84/gui/wxpython/wxgui.py'
GRASS_ANCHOR = '''def status_dump():
    global frame
    assert frame is not None
'''
GRASS_MISSING_DISPLAY = '''
    # CUA-RL: a closed map display is a valid GUI state, not an HTTP failure.
    # None deliberately differs from False: closing the whole display must
    # not satisfy the task asking to hide its toolbar.
    if not frame.mainnotebook._tabs._pages:
        return {
            "layers": [prejsonify(item) for item in frame.pg_panel.maptree.Map.layers],
            "windows": [{"name": item._data[0]["maplayer"].name, **prejsonify(item)}
                        for item in frame.pg_panel.maptree._itemWithWindow],
            "toolbars": {"keys": [], "shown": None, "layer": None},
            "statusbar": {"mode": None, "scale": None},
            "region": {},
            "output": frame.goutput.cmdOutput.GetText(),
            "map_display_present": False,
        }
'''


def patch_grass_source(source):
    if GRASS_ANCHOR + GRASS_MISSING_DISPLAY in source:
        return source
    if source.count(GRASS_ANCHOR) != 1:
        raise RuntimeError('Unexpected GRASS status_dump source; refusing to patch')
    return source.replace(GRASS_ANCHOR, GRASS_ANCHOR + GRASS_MISSING_DISPLAY)


def prepare_grass(controller):
    """Patch only the current VM overlay; return both hashes for the audit."""
    base = f'http://{controller.vm_ip}:{controller.server_port}'
    response = requests.get(base + '/read', params={'path': GRASS_PATH}, timeout=30)
    response.raise_for_status()
    original = response.text
    fixed = patch_grass_source(original)
    before = hashlib.sha256(original.encode()).hexdigest()
    after = hashlib.sha256(fixed.encode()).hexdigest()
    code = ('from pathlib import Path; import hashlib; '
            f'p=Path({GRASS_PATH!r}); '
            f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={before!r}; '
            f'p.write_text({fixed!r})')
    response = requests.post(base + '/execute', json={'command': ['python3', '-c', code]}, timeout=30)
    response.raise_for_status()
    if response.json().get('returncode') != 0:
        raise RuntimeError('Failed to install GRASS missing-display serializer in VM overlay')
    return {'kind': 'grass_closed_display_serialization', 'path': GRASS_PATH,
            'original_sha256': before, 'patched_sha256': after}


@lru_cache(maxsize=1)
def setup_repairs():
    return json.loads(Path(__file__).with_name('scienceboard_setup_repairs.json').read_text())['tasks']


def restore_initialization(task):
    """Restore only attested omissions in the released task, in their original order."""
    repair = setup_repairs().get(task['id'])
    if not repair:
        return []
    signature = hashlib.sha256(json.dumps(task['config'], sort_keys=True).encode()).hexdigest()
    if signature != repair['config_sha256']:
        raise ValueError(f"ScienceBoard task setup changed for {task['id']}")
    config = []
    changes = []
    lean_prefix = []
    for i, old in enumerate(task['config']):
        step = repair['steps'].get(str(i))
        if step is None:
            config.append(old)
            continue
        op = step['op']
        params = {k: v for k, v in step.items() if k not in ('op', 'wait')}
        if op in ('map', 'scale', 'layer'):
            fixed = {'type': 'grass_' + op, 'parameters': {'port': 8000, **params}}
        elif op in ('open', 'def'):
            text = 'open ' + ' '.join(params['libs']) if op == 'open' else params['expr']
            lean_prefix.append(text)
            fixed = {'type': 'append_file', 'parameters': {'path': '/home/user/sci/Sci/Basic.lean', 'content': text + '\n'}}
        elif op == 'download':
            fixed = {'type': 'download', 'parameters': {'files': [params]}}
        elif op in ('alphafold_match', 'color'):
            command = 'alphafold match ' + params['name'] if op == 'alphafold_match' else 'color ' + params['style']
            fixed = {'type': 'chimerax_command', 'parameters': {'command': command}}
        else:
            raise ValueError(f'Unsupported initialization operation {op}')
        config.append(fixed)
        if step.get('wait'):
            config.append({'type': 'sleep', 'parameters': {'seconds': step['wait']}})
        changes.append({'kind': 'restore_' + op, 'step_index': i, 'original': old, 'replacement': fixed})
    if lean_prefix:
        # Both the file and the release's header check must contain the same context.
        query = next(s['parameters']['expr'] for s in config if s['type'] == 'lean_query')
        header = task['evaluator']['lean_header']
        if header.count(query) != 1:
            raise ValueError(f"Unrecognized Lean header in {task['id']}")
        task['evaluator']['lean_header'] = header.replace(query, '\n'.join([*lean_prefix, query]))
    task['config'] = config
    return changes


def normalize_task(task, profile='released'):
    result = copy.deepcopy(task)
    if profile not in ('released', 'restored'):
        raise ValueError(f'Unknown ScienceBoard setup profile {profile}')
    changes = restore_initialization(result) if profile == 'restored' else []
    if result.get('evaluator', {}).get('task_type') == 'Lean':
        config = []
        for index, step in enumerate(result.get('config', [])):
            if step['type'] == 'chimerax_open' and not step.get('parameters', {}).get('name'):
                changes.append({'kind': 'remove_empty_chimerax_open_from_lean', 'step_index': index,
                                'original': step})
            elif step['type'] == 'lean_query':
                # The released helper's /write truncates the file. The task
                # controller intends to append the theorem after Mathlib imports.
                expr = step['parameters']['expr'].replace('by sorry', 'by').strip() + '\n'
                config.append({'type': 'append_file', 'parameters': {
                    'path': '/home/user/sci/Sci/Basic.lean', 'content': expr}})
                changes.append({'kind': 'append_lean_query_preserving_imports', 'step_index': index,
                                'original': step})
            else:
                config.append(step)
        result['config'] = config
    return result, changes


def install_setup(controller):
    """Match the original ScienceBoard KAlgebra manager's wire format.

    Reference: OS-Copilot/ScienceBoard, sci/KAlgebra/kalgebra.py:
    /tab takes a JSON integer; /add/{2,3}d take expression text.
    """
    def operation(port, route, **kwargs):
        response = requests.post(f'http://{controller.vm_ip}:{port}/{route}', timeout=60, **kwargs)
        response.raise_for_status()
        if response.text.strip() != 'OK':
            raise RuntimeError(f'ScienceBoard {route} did not acknowledge the setup operation')
    def tab(port, index):
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError('KAlgebra tab index must be an integer in [0, 3]')
        operation(port, 'tab', json=index)
    controller._kalgebra_tab_setup = tab
    controller._kalgebra_func_2d_setup = lambda port, expr: operation(port, 'add/2d', data=expr)
    controller._kalgebra_func_3d_setup = lambda port, expr: operation(port, 'add/3d', data=expr)

    def grass_map(port, grassdb, location, mapset):
        response = requests.post(f'http://{controller.vm_ip}:{port}/init/map',
                                 json={'grassdb': grassdb, 'location': location, 'mapset': mapset}, timeout=60)
        response.raise_for_status()
        # On TCG the GUI can complete the switch after the image's 10-second HTTP acknowledgement window.
        # Check the selected mapset in the running GRASS GUI process before sending any model input.
        if response.text.strip() not in {'OK', 'ERROR'}:
            raise RuntimeError('Unexpected GRASS map initialization response')
        expected = {'GISDBASE': grassdb, 'LOCATION_NAME': location, 'MAPSET': mapset}
        script = """from pathlib import Path
import json
states = []
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = p.read_bytes().split(bytes([0]))
        if not any(arg.endswith(b'/wxgui.py') for arg in argv):
            continue
        env = dict(item.split(b'=', 1) for item in (p.parent / 'environ').read_bytes().split(bytes([0])) if b'=' in item)
        rc = Path(env[b'GISRC'].decode()).read_text()
        values = dict(line.split(': ', 1) for line in rc.splitlines() if ': ' in line)
        states.append({key: values.get(key) for key in ('GISDBASE', 'LOCATION_NAME', 'MAPSET')})
    except (FileNotFoundError, PermissionError, KeyError, ProcessLookupError):
        continue
print(json.dumps(states))
"""
        deadline = time.monotonic() + 60
        while True:
            check = requests.post(f'http://{controller.vm_ip}:{controller.server_port}/execute',
                                  json={'command': ['python3', '-c', script]}, timeout=30)
            check.raise_for_status()
            result = check.json()
            if result.get('returncode') == 0 and json.loads(result['output']) == [expected]:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'GRASS GUI did not select requested mapset {expected}')
            time.sleep(2)

    def grass_layer(port, query):
        operation(port, 'init/layer', json={'query': query})
        deadline = time.monotonic() + 60
        while True:
            result = requests.get(f'http://{controller.vm_ip}:{port}/dump', timeout=20)
            result.raise_for_status()
            if any(layer['name'].split('@')[0] == query['name'] for layer in result.json()['layers']):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'GRASS GUI did not display requested layer {query}')
            time.sleep(2)

    controller._grass_map_setup = grass_map
    controller._grass_layer_setup = grass_layer
    controller._grass_scale_setup = lambda port, scale: operation(port, 'init/scale', json={'scale': scale})

    def chimerax_command(command):
        response = requests.post(f'http://{controller.vm_ip}:{controller.server_port}/chimerax/run',
                                 json={'command': command}, timeout=120)
        response.raise_for_status()
        body = response.json()
        if body.get('error') or body.get('log messages', {}).get('error'):
            raise RuntimeError(f'ChimeraX initialization failed for {command}')
    controller._chimerax_command_setup = chimerax_command
