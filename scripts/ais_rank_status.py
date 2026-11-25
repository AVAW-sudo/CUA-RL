"""Read AIS rank status across workers on the shared filesystem."""
import errno
import json
import sys
import time


def read_rank_status(path, *, missing_ok=False, poll=False):
    """With poll=True, let the caller retry transient or incomplete reads on its own clock."""
    delays = iter((1, 2, 4, 8, 15, 30, 30, 30))
    while True:
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                raise ValueError(f'Expected a rank status object in {path}')
            return record
        except json.JSONDecodeError as exc:
            error = json.JSONDecodeError(f'{path}: {exc.msg}', exc.doc, exc.pos)
            if exc.doc.strip() and not poll:
                raise error from exc
        except OSError as exc:
            if missing_ok and exc.errno == errno.ENOENT:
                return None
            if exc.errno not in (errno.EACCES, errno.ESTALE, errno.ENOENT):
                raise
            error = exc
        if poll:
            return None
        delay = next(delays, None)
        if delay is None:
            raise error
        print(f'Cannot read rank status {path} ({error}), retrying in {delay}s', file=sys.stderr, flush=True)
        time.sleep(delay)
