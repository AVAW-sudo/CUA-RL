"""Guest verifier health checks and scoring error propagation for the private RL worker."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests


_score = ContextVar('ais_verifier_requests', default=None)
_download_errors = ContextVar('ais_verifier_download', default=None)


def read_guest_file(get_file, path):
    """Let one upstream get_file call finish its own retries before judging it."""
    scope = _score.get()
    if scope is None:
        return get_file(path)
    errors = []
    token = _download_errors.set(errors)
    try:
        data = get_file(path)
        if isinstance(data, bytes):
            errors.clear()
        return data
    finally:
        _download_errors.reset(token)
        scope[1].extend(errors)


def install_scoring_observer(startup_budget=None):
    send = requests.Session.send
    if getattr(send, 'ais_scoring_observer', False):
        return

    @wraps(send)
    def observed(session, request, **kwargs):
        if startup_budget is not None:
            remaining = startup_budget()
            if remaining is not None:
                # Several upstream reset/setup calls leave Requests' timeout unset.
                timeout = kwargs.get('timeout')
                if isinstance(timeout, tuple):
                    kwargs['timeout'] = tuple(remaining if t is None else min(t, remaining) for t in timeout)
                elif timeout is None or isinstance(timeout, (int, float)):
                    kwargs['timeout'] = remaining if timeout is None else min(timeout, remaining)
        scope = _score.get()
        if scope is None or not request.url.startswith(scope[0] + '/'):
            return send(session, request, **kwargs)
        route = urlsplit(request.url).path
        try:
            response = send(session, request, **kwargs)
        except requests.RequestException as exc:
            errors = scope[1]
            download = _download_errors.get()
            if (download is not None and route == '/file' and request.method == 'POST'
                    and isinstance(exc, requests.exceptions.ChunkedEncodingError)):
                errors = download
            errors.append(f'{route} {type(exc).__name__}')
            raise
        if response.status_code >= 500:
            scope[1].append(f'{route} HTTP {response.status_code}')
        elif route == '/setup/execute' and response.ok:
            command = json.loads(request.body or '{}').get('command')
            if isinstance(command, list) and command[:2] == ['/usr/bin/timeout', '--kill-after=2']:
                code = response.json().get('returncode')
                if code != 0:
                    scope[1].append(f'{route} timed command returned {code}')
        return response

    observed.ais_scoring_observer = True
    requests.Session.send = observed


@contextmanager
def scoring_requests(base_url):
    errors = []
    token = _score.set((base_url.rstrip('/'), errors))
    try:
        yield
        if errors:
            raise RuntimeError('Guest verifier failed during scoring: ' + ', '.join(errors[:5]))
    finally:
        _score.reset(token)


def run_bash_once(base_url, script, timeout=30, working_dir=None):
    # An HTTP error can occur after the script has already changed the guest.
    try:
        response = requests.post(base_url + '/run_bash_script',
                                 json=dict(script=script, timeout=timeout, working_dir=working_dir),
                                 timeout=timeout + 100)
        response.raise_for_status()
        result = response.json()
        if (not isinstance(result, dict) or type(result.get('returncode')) != int or
                result.get('status') != ('success' if result['returncode'] == 0 else 'error') or
                not isinstance(result.get('output'), str)):
            raise ValueError('Invalid guest bash result')
        return result
    except (requests.RequestException, ValueError) as exc:
        scope = _score.get()
        if scope is not None:
            scope[1].append(f'/run_bash_script {type(exc).__name__}')
        raise


def prepare_guest_verifier(vm):
    base = f'http://{vm.address}:{vm.port}'
    def probe():
        return run_bash_once(base, 'printf cua-verifier', timeout=5)

    try:
        result = probe()
    except requests.HTTPError as exc:
        if exc.response.status_code != 500 or "name '_append_event' is not defined" not in exc.response.text:
            raise
        # Werkzeug watches this file inside the writable session overlay.
        try:
            vm.guest(Path(__file__).with_name('ais_guest_verifier.py').read_text())
        except (ConnectionError, OSError):
            pass  # A reload can close /execute before its response is delivered.
        deadline = time.monotonic() + 100
        while True:
            try:
                result = probe()
                break
            except requests.RequestException:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    if result['output'] != 'cua-verifier' or result['returncode'] != 0:
        raise RuntimeError(f'Guest bash positive control failed: {result!r}')
    negative = run_bash_once(base, 'printf cua-negative >&2; exit 3', timeout=5)
    if negative['returncode'] != 3 or negative['output'] != 'cua-negative':
        raise RuntimeError(f'Guest bash negative control failed: {negative!r}')
