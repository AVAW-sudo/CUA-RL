"""Preserve model-call and verifier failures that the released agents catch internally."""
import json
import logging
from pathlib import Path


def evidence_path(output, domain, task_id):
    return Path(output) / 'episode-audit' / domain / (task_id + '.json')


def verifier_fault(record):
    if record.levelno < logging.ERROR:
        return False
    message = record.getMessage()
    return ((record.name == 'desktopenv.scienceboard.evaluator' and message.startswith('ScienceBoard evaluation error for '))
            or (record.name == 'desktopenv.getters.chrome' and '] All 3 attempts failed' in message)
            or (record.name == 'desktopenv.metric.docs' and message.startswith('OCR processing failed for ')))


class EpisodeAudit(logging.Handler):
    def __init__(self, path):
        super().__init__(logging.ERROR)
        self.path = path
        self.state = {'status': 'running', 'model_requested': False, 'model_calls': 0, 'errors': []}
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state, ensure_ascii=False) + '\n')
        temp.replace(self.path)

    def error(self, reason):
        if reason not in self.state['errors']:
            self.state['errors'].append(reason)
            self.save()

    def emit(self, record):
        if verifier_fault(record):
            self.error('Verifier: ' + record.getMessage()[:1000])

    def run(self, runner, task):
        call_llm = runner.call_llm

        def checked_call(*args, **kwargs):
            self.state['model_calls'] += 1
            if not self.state['model_requested']:
                self.state['model_requested'] = True
                self.save()
            try:
                return call_llm(*args, **kwargs)
            except Exception as exc:
                self.error(f'Inference: {type(exc).__name__}: {str(exc)[:900]}')
                raise

        root_logger = logging.getLogger()
        root_logger.addHandler(self)
        try:
            runner.call_llm = checked_call
            row = runner._worker_run(task)
            if row is None or row[2] is None:
                self.error('Upstream worker returned no score')
            elif not self.state['model_requested']:
                self.error('Worker returned a score before any model request')
            self.state['status'] = 'finished'
            return row
        finally:
            runner.call_llm = call_llm
            root_logger.removeHandler(self)
            self.save()
