"""CPU-only fault injection: kill a supervisor, reattach its surviving command."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from persistent_job import alive, identity, write
from pipeline_watchdog import tick
from run_benchmark import ROOT, now


def main():
    directory=ROOT/'runs/watchdog-validation'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    config=directory/'config.json';write(config,{'enabled':True})
    dummy=directory/'dummy_supervisor.py'
    code='''import fcntl,os,time,json,sys
from pathlib import Path
sys.path.insert(0,SCRIPTS)
from persistent_job import start,identity,write
from run_benchmark import now
root=Path(DIRECTORY)
lease=(root/'supervisor.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
command=[sys.executable,'-c',"from pathlib import Path; import time; p=Path('starts.txt'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); time.sleep(4)"]
job=start(root/'job',command,root)
while True:
 write(root/'state.json',{'phase':'fault_test','process_identity':identity(os.getpid()),'updated_at_utc':now(),'job_pid':job.pid})
 time.sleep(.1)
'''.replace('SCRIPTS',repr(str(ROOT/'scripts'))).replace('DIRECTORY',repr(str(directory)))
    dummy.write_text(code)
    report={'status':'running','started_at_utc':now(),'directory':str(directory),'gpu_allocation':0}
    command=[sys.executable,str(dummy)]
    current=None
    try:
        report['first_tick']=tick(directory,config,command)
        deadline=time.monotonic()+15
        while not (directory/'state.json').exists() and time.monotonic()<deadline:time.sleep(.05)
        first=json.loads((directory/'state.json').read_text());current=first['process_identity']
        report['original_supervisor']=current
        report['job_pid_before']=first['job_pid']
        os.kill(current['pid'],signal.SIGKILL)
        while alive(current) and time.monotonic()<deadline:time.sleep(.05)
        report['recovery_tick']=tick(directory,config,command)
        while time.monotonic()<deadline:
            second=json.loads((directory/'state.json').read_text())
            if second['process_identity']!=current:break
            time.sleep(.05)
        current=second['process_identity']
        report['new_supervisor']=current
        report['job_pid_after']=second['job_pid']
        assert current!=first['process_identity']
        assert report['job_pid_before']==report['job_pid_after']
        while not (directory/'job/receipt.json').exists() and time.monotonic()<deadline:time.sleep(.05)
        report['command_receipt']=json.loads((directory/'job/receipt.json').read_text())
        report['command_start_count']=len((directory/'starts.txt').read_text())
        assert report['command_start_count']==1
        assert report['command_receipt']['exit_code']==0
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        if alive(current):os.kill(current['pid'],signal.SIGTERM)
        report['finished_at_utc']=now()
        write(ROOT/'reports/watchdog-recovery-validation.json',report)


if __name__=='__main__':main()
