"""Conservative recovery: repeat setup failures, never resample an acted episode."""
import json
from pathlib import Path


def classify_attempt(attempt):
    directory = Path(attempt['attempt_dir'])
    actions = int(attempt.get('action_records') or 0)
    sampled = bool(attempt.get('model_requested'))
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        for row in json.loads(summary_path.read_text()).get('tasks', []):
            actions = max(actions, int(row.get('action_records') or 0))
            sampled |= bool(row.get('model_requested'))
    for path in (directory / 'episode-audit').glob('*/*.json'):
        try:
            evidence = json.loads(path.read_text())
            requested = evidence['model_requested']
            if type(requested) is not bool:
                raise ValueError('Invalid model_requested flag')
            sampled |= requested
        except (OSError, ValueError, KeyError, TypeError):
            return {'kind': 'unclassified_failure', 'retryable': False, 'action_records': actions,
                    'reason': f'Unreadable model-call evidence: {path}'}
    for path in directory.rglob('traj.jsonl'):
        recorded = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError:
                recorded += 1  # Corrupt evidence cannot establish a zero-action episode.
                continue
            if isinstance(item, dict) and set(item) == {'Error'}:
                continue  # Upstream appends a diagnostic row even when setup failed.
            recorded += 1
        actions = max(actions, recorded)
    log_path = Path(attempt.get('log', directory.parent / (directory.name + '.log')))
    log = log_path.read_text(errors='replace') if log_path.exists() else ''
    error = attempt.get('error') or ''
    if actions or sampled:
        return {'kind': 'acted_episode_needs_artifact_review', 'retryable': False,
                'action_records': actions, 'reason': 'Keep original trajectory; do not sample again to recover a score'}
    setup_markers = ['[Init Fail]', 'VM not ready after', 'Setup step ',
                     'Setup HTTP failure', 'Task setup failed', 'Application API did not start']
    if any(marker in error + '\n' + log for marker in setup_markers):
        return {'kind': 'zero_action_initialization_failure', 'retryable': True,
                'action_records': 0, 'reason': 'Explicit initialization failure and no recorded model actions'}
    return {'kind': 'unclassified_failure', 'retryable': False, 'action_records': actions,
            'reason': 'Absence of an action file alone does not prove that the model was never sampled'}


def retry_eligible(row, max_attempts=3):
    if row['status'] != 'infrastructure_error' or len(row.get('attempts', [])) >= max_attempts:
        return False
    return bool(row['attempts'] and classify_attempt(row['attempts'][-1])['retryable'])
