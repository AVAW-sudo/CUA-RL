"""Reuse PyAutoGUI inside one disposable guest for literal input events."""
import ast
import json
import logging
from pathlib import Path
import time

METHODS = frozenset('click doubleClick tripleClick rightClick moveTo moveRel dragTo scroll hscroll '
                    'press keyDown keyUp hotkey typewrite write sleep mouseDown mouseUp'.split())


def input_events(command):
    try:
        events = []
        for stmt in ast.parse(command).body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                return None
            call = stmt.value
            fn = call.func
            if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and fn.value.id == 'pyautogui' and fn.attr in METHODS):
                return None
            if any(k.arg is None for k in call.keywords):
                return None
            events.append([fn.attr, [ast.literal_eval(a) for a in call.args],
                           {k.arg: ast.literal_eval(k.value) for k in call.keywords}])
        # Exclude Python-only literals, such as bytes, which cannot round-trip through JSON.
        return events if events and json.loads(json.dumps(events, allow_nan=False)) == events else None
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def serve(socket_path):
    import os
    from http.server import BaseHTTPRequestHandler
    from socketserver import UnixStreamServer
    import pyautogui
    pyautogui.FAILSAFE = False

    class Handler(BaseHTTPRequestHandler):
        def address_string(self):
            # UnixStreamServer has no TCP peer address for the standard HTTP log.
            return 'local'

        def do_POST(self):
            try:
                size = int(self.headers['Content-Length'])
                if not 0 < size < 1024 * 1024:
                    raise ValueError('Invalid request size')
                for name, args, kwargs in json.loads(self.rfile.read(size)):
                    if name not in METHODS:
                        raise ValueError('Unsupported GUI input')
                    getattr(pyautogui, name)(*args, **kwargs)
                self.send_response(200)
                self.end_headers()
            except Exception as exc:
                self.send_error(500, str(exc))

    os.umask(0o077)
    with UnixStreamServer(socket_path, Handler) as server:
        server.serve_forever()


class GuestInput:
    def __init__(self, vm, controller):
        self.vm = vm
        self.controller = controller
        self.original = controller.execute_python_command
        self.socket_path = None

    def execute(self, command):
        events = input_events(command)
        if events is None:
            return self.original(command)
        if self.socket_path is None:
            code = ("import pathlib,sys,tempfile,subprocess,time; d=pathlib.Path(tempfile.mkdtemp()); "
                    "sock=str(d/'input.sock'); proc=subprocess.Popen([sys.executable,'-c',sys.argv[1],sock],"
                    "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); "
                    "\nfor _ in range(300):\n if (d/'input.sock').exists() or proc.poll() is not None: break\n time.sleep(.1)"
                    "\nassert (d/'input.sock').exists(), 'GUI input process did not start'; print(sock)")
            self.socket_path = self.vm.guest(code, Path(__file__).read_text())
        import requests
        start = time.monotonic()
        request = ['curl', '--silent', '--show-error', '--fail', '--noproxy', '*', '--max-time', '115', '--unix-socket',
                   self.socket_path, '--data-binary', json.dumps(events), 'http://localhost/input']
        r = requests.post(self.controller.http_server + '/execute', json={'command': request, 'shell': False}, timeout=125)
        r.raise_for_status()
        result = r.json()
        if result.get('returncode') != 0:
            raise RuntimeError(f'GUI input failed: {result}')
        logging.getLogger('desktopenv.pycontroller').info('Resident GUI input completed in %.3fs',
                                                         time.monotonic() - start)
        return result


def compare_input(vm, evidence):
    """Exercise identical keyboard and mouse input on a temporary guest widget."""
    from types import SimpleNamespace
    from ais_vm_check import save
    widget = """
import sys, tkinter as tk
from pathlib import Path
root = tk.Tk()
root.title('SFT keyboard and mouse check')
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
value = tk.StringVar()
entry = tk.Entry(root, textvariable=value)
entry.pack(fill=tk.BOTH, expand=True)
value.trace_add('write', lambda *_: Path(sys.argv[1]).write_text(value.get()))
root.update()
root.lift()
entry.focus_force()
entry.bind('<ButtonPress-1>', lambda *_: entry.focus_force(), add='+')
Path(sys.argv[1]).write_text('')
root.mainloop()
"""
    launch_widget = ("import json,sys,tempfile,pathlib,subprocess,time; p=pathlib.Path(tempfile.mkdtemp())/'typed'; "
                     "log=p.with_suffix('.log'); proc=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(p)],"
                     "stdout=log.open('w'),stderr=subprocess.STDOUT,start_new_session=True); "
                     "\nfor _ in range(300):\n if p.exists() or proc.poll() is not None: break\n time.sleep(.1)"
                     "\nassert p.exists(), 'GUI test widget failed to start: ' + log.read_text()[-800:]; "
                     "print(json.dumps([str(p),proc.pid]))")
    def legacy(command):
        vm.guest('import pyautogui; import time; pyautogui.FAILSAFE = False; ' + command)
    resident = GuestInput(vm, SimpleNamespace(execute_python_command=legacy, http_server=f'http://{vm.address}:{vm.port}'))
    results = []
    for mode, send in [('legacy', legacy), ('resident', resident.execute)]:
        path, pid = json.loads(vm.guest(launch_widget, widget))
        # Use the same neutral setup and exclude the resident's one-time import from action timing.
        send('pyautogui.moveTo(150, 130)')
        commands = ['pyautogui.click(150,130)', "pyautogui.hotkey('ctrl','a')", "pyautogui.press('backspace')",
                    "pyautogui.write('abc123')", "pyautogui.press('left')", "pyautogui.press('backspace')",
                    'pyautogui.moveTo(180,140)']
        durations = []
        for command in commands:
            start = time.monotonic()
            send(command)
            durations.append(time.monotonic() - start)
        measured = json.loads(vm.guest("import json,pyautogui,sys; from pathlib import Path; "
                                       "print(json.dumps([Path(sys.argv[1]).read_text(), list(pyautogui.position())]))", path))
        vm.guest("import os,signal,sys; os.kill(int(sys.argv[1]),signal.SIGTERM)", str(pid))
        results.append({'backend': mode, 'commands': commands, 'seconds': durations, 'final_state': measured})
        save(evidence / 'input-check.json', {'trials': results})
        if measured != ['abc13', [180, 140]]:
            raise RuntimeError(f'Unexpected {mode} GUI effect: {measured}')


if __name__ == '__main__':
    import sys
    serve(sys.argv[1])
