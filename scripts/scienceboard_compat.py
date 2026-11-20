"""Auditable compatibility fixes for the released ScienceBoard task runtime.

Keep instructions, evaluator predicates and model actions unchanged. Every
normalization is recorded separately from the original task configuration.
"""
import copy
import hashlib
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


def normalize_task(task):
    result = copy.deepcopy(task)
    changes = []
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
            raise RuntimeError(f'KAlgebra {route} did not acknowledge the setup operation')
    def tab(port, index):
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError('KAlgebra tab index must be an integer in [0, 3]')
        operation(port, 'tab', json=index)
    controller._kalgebra_tab_setup = tab
    controller._kalgebra_func_2d_setup = lambda port, expr: operation(port, 'add/2d', data=expr)
    controller._kalgebra_func_3d_setup = lambda port, expr: operation(port, 'add/3d', data=expr)
