"""Concurrent full-snapshot reset and cross-VM isolation acceptance; no GPU usage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import statistics
import time
from snapshot_vm import SnapshotVM, implementation_hash
from run_benchmark import ROOT, now, write_json
from check_vm_concurrency import exercise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 32 or not 1 <= args.rounds <= 5 or args.report.exists():
        parser.error('Use 1..32 VMs, 1..5 rounds and a new report')
    report = {'status':'running','hostname':socket.gethostname(),'count':args.count,
              'implementation_sha256':implementation_hash(),'started_at_utc':now(),
              'scope':'Concurrent snapshot reset isolation and guest API, not benchmark score',
              'gpu_allocation':0,'rounds':[]}
    vms=[]
    write_json(args.report, report)
    try:
        for i in range(args.count):
            vm=SnapshotVM(f'127.0.5.{20+i}')
            for port in [5000,9222,8080,8000,5900]:
                with socket.socket() as check: check.bind((vm.address,port))
            vms.append(vm);vm.start();time.sleep(1)
        def boot(vm):
            boot_seconds=vm.wait_ready(timeout=400)
            vm.save_clean()
            return {'id':vm.id,'ready_wait_seconds':boot_seconds}
        with ThreadPoolExecutor(max_workers=args.count) as pool:
            report['vms']=list(pool.map(boot,vms));write_json(args.report,report)
            for iteration in range(args.rounds):
                def exercise_reset(vm):
                    vm.guest("from pathlib import Path; import subprocess,pyautogui; "
                        f"Path('/tmp/cua-reset-dirty').write_text({vm.id!r}); "
                        "pyautogui.moveTo(123,456); "
                        "subprocess.Popen(['python','-c','import time; time.sleep(600)','cua-dirty-process'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)")
                    result=vm.restore_clean()
                    # RAM/process state, not just disk files, must be clean.
                    result['dirty_processes']=vm.guest("import subprocess; print(subprocess.run(['pgrep','-f','^python -c import time; time.sleep\\(600\\) cua-dirty-process$'],capture_output=True,text=True).stdout.strip())")
                    if result['dirty_processes']: raise RuntimeError('Task process survived snapshot restore')
                    result['api']=exercise(vm,2)
                    return result
                row={'iteration':iteration+1,'resets':list(pool.map(exercise_reset,vms))}
                report['rounds'].append(row);write_json(args.report,report)
        report['status']='passed'
        report['reset_median_seconds']=statistics.median(r['reset_seconds'] for row in report['rounds'] for r in row['resets'])
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        for vm in vms:vm.stop()
        report['cleanup_passed']=all(vm.process is None or vm.process.poll() is not None for vm in vms)
        report['finished_at_utc']=now();write_json(args.report,report)


if __name__=='__main__':main()
