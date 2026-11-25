"""Fixed functional checks sent to the disposable OSWorld guest by ais_vm_check.py."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request


def desktop(token):
    import pyautogui as gui
    import tkinter as tk

    window = tk.Tk()
    try:
        window.title('CUA-RL input check')
        window.geometry('360x110+100+100')
        window.attributes('-topmost', True)
        field = tk.Entry(window)
        field.pack(expand=True, fill='both')
        clicks = []
        field.bind('<Button-1>', lambda event: clicks.append(event.x))
        initial_pointer = list(gui.position())
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            window.update()
            if field.winfo_viewable() and field.winfo_width() > 20 and field.winfo_height() > 20:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('X11 did not map the test entry field')
        window.lift()
        field.focus_force()
        window.update()
        x = field.winfo_rootx() + field.winfo_width() // 2
        y = field.winfo_rooty() + field.winfo_height() // 2
        if not gui.onScreen(x, y) or (x, y) in gui.FAILSAFE_POINTS:
            raise RuntimeError(f'Invalid mapped field center {(x, y)}')
        # Use an X11 pointer move to leave a fresh desktop's corner, with PyAutoGUI failsafe on.
        from Xlib import X, display
        from Xlib.ext import xtest
        connection = display.Display()
        try:
            xtest.fake_input(connection, X.MotionNotify, x=x, y=y)
            connection.sync()
        finally:
            connection.close()
        try:
            gui.click(x, y)
        except gui.FailSafeException as exc:
            raise RuntimeError(f'Pointer {gui.position()} at click, target {(x, y)}, start {initial_pointer}') from exc
        window.update()
        gui.write(token, interval=0.04)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and field.get() != token:
            window.update()
            time.sleep(0.05)
        if not clicks or field.get() != token:
            raise RuntimeError('Mouse click or typed text did not reach the test window')
        return {'typed_characters': len(token), 'clicks': len(clicks),
                'initial_pointer': initial_pointer, 'display': list(gui.size())}
    finally:
        window.destroy()


def network(url, token):
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with local.open(url, timeout=10) as response:
        if token not in response.read().decode():
            raise RuntimeError('Guest received the wrong HTTP fixture')
    return {'received_fixture': True}


def browser(url, token):
    cmd = next((shutil.which(n) for n in ('google-chrome', 'chromium', 'chromium-browser')
                if shutil.which(n)), None)
    if not cmd:
        raise FileNotFoundError('Chrome/Chromium')
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.run([cmd, '--headless', '--disable-gpu', '--disable-dev-shm-usage',
                               '--disable-background-networking', '--no-first-run', '--no-proxy-server',
                               '--user-data-dir=' + profile, '--dump-dom', url],
                              capture_output=True, text=True, timeout=95)
        if proc.returncode or f'<title>{token}</title>' not in proc.stdout:
            raise RuntimeError(f'Browser did not render the fixture: {proc.stderr[-400:]}')
    return {'executable': cmd, 'rendered_fixture': True}


def office(token):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        doc = root / 'input.txt'
        doc.write_text(token)
        cmd = shutil.which('libreoffice') or shutil.which('soffice')
        if not cmd:
            raise FileNotFoundError('LibreOffice')
        proc = subprocess.run([cmd, '-env:UserInstallation=' + (root / 'profile').as_uri(),
                               '--headless', '--convert-to', 'pdf', '--outdir', directory, str(doc)],
                              capture_output=True, text=True, timeout=95)
        pdf = root / 'input.pdf'
        if proc.returncode or not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF-'):
            raise RuntimeError(f'Writer PDF export failed: {proc.stdout[-300:]} {proc.stderr[-300:]}')
        return {'pdf_bytes': pdf.stat().st_size}


def science():
    grass = shutil.which('grass') or shutil.which('grass84')
    if not grass:
        raise FileNotFoundError('GRASS GIS')
    # GRASS creates and removes a temporary Cartesian project around this command.
    commands = '''
import json, subprocess
subprocess.run(['g.region', 'n=2', 's=0', 'e=2', 'w=0', 'res=1'], check=True)
subprocess.run(['r.mapcalc', 'expression=ais_check = 7'], check=True)
stats = subprocess.check_output(['r.univar', '-g', 'map=ais_check'], text=True)
values = dict(line.split('=', 1) for line in stats.splitlines() if '=' in line)
print(json.dumps({'cells': int(values['n']), 'mean': float(values['mean'])}))
'''
    proc = subprocess.run([grass, '--tmp-project', 'XY', '--exec', sys.executable, '-c', commands],
                          capture_output=True, text=True, timeout=95)
    if proc.returncode or not proc.stdout.strip():
        raise RuntimeError('GRASS GIS failed: ' + proc.stderr[-650:])
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if result != {'cells': 4, 'mean': 7}:
        raise RuntimeError(f'Unexpected raster statistics: {result}')
    return result


if __name__ == '__main__':
    action, token, url = sys.argv[1:]
    checks = {'desktop': lambda: desktop(token), 'network': lambda: network(url, token),
              'browser': lambda: browser(url, token), 'office': lambda: office(token), 'science': science}
    print(json.dumps(checks[action]()))
