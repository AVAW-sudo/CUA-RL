import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from run_benchmark import ROOT
import snapshot_vm


def load_controller():
    class Base:
        def __init__(self, delegation): pass
    desktop=types.ModuleType('desktop_env')
    desktop.desktop_env=types.ModuleType('desktop_env.desktop_env')
    desktop.providers=types.ModuleType('desktop_env.providers')
    stubs={'agentrl.worker.environment':types.SimpleNamespace(EnvironmentController=Base),
           'desktop_env':desktop,'desktop_env.desktop_env':desktop.desktop_env,
           'desktop_env.providers':desktop.providers,
           'src.task':types.SimpleNamespace(OSWorldTask=object),
           'src.environment':types.SimpleNamespace(create_manager_and_provider=lambda *a:None)}
    spec=importlib.util.spec_from_file_location('test_pool_adapter',ROOT/'scripts/local_osworld.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,stubs):spec.loader.exec_module(module)
    return module


class SnapshotPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_lease_reuses_vm_but_stale_end_cannot_reset_active_episode(self):
        module=load_controller()
        with tempfile.TemporaryDirectory() as tmp:
            made=[]
            class VM:
                def __init__(self,address,**kwargs):
                    self.address=address;self.started=0;self.restored=0;self.stopped=0
                    self.process=types.SimpleNamespace(poll=lambda:None)
                    self.audit_directory=Path(tmp);made.append(self)
                def start(self):self.started+=1
                def wait_ready(self):pass
                def save_clean(self):pass
                def restore_clean(self):self.restored+=1
                def stop(self):self.stopped+=1
            report=Path(tmp)/'accept.json'
            report.write_text(json.dumps({'status':'passed','hostname':socket.gethostname(),
                'count':1,'cleanup_passed':True,'implementation_sha256':'fixed'}))
            with patch.object(snapshot_vm,'implementation_hash',return_value='fixed'), \
                 patch.object(snapshot_vm,'SnapshotVM',VM),patch.object(module.atexit,'register'):
                pool=module.QemuController(None,1,warm_reset_report=report)
                first,_,_=await pool.start_session('Ubuntu')
                await pool.end_session(first)
                second,_,_=await pool.start_session('Ubuntu')
                self.assertNotEqual(first,second)
                self.assertEqual(len(made),1)
                self.assertEqual(made[0].started,1)
                self.assertEqual(made[0].restored,1)
                await pool.end_session(first)
                self.assertIn(second,pool.instances)
                self.assertEqual(made[0].restored,1)
                # A failed reset must discard this VM and return the capacity.
                def fail():raise RuntimeError('injected reset failure')
                made[0].restore_clean=fail
                await pool.end_session(second)
                self.assertEqual(made[0].stopped,1)
                self.assertFalse(pool.idle)
                self.assertEqual(pool.addresses.qsize(),1)

    async def test_acceptance_from_different_implementation_is_rejected(self):
        module=load_controller()
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'accept.json'
            report.write_text(json.dumps({'status':'passed','hostname':socket.gethostname(),
                'count':32,'cleanup_passed':True,'implementation_sha256':'old-code'}))
            with self.assertRaises(RuntimeError):
                module.QemuController(None,1,warm_reset_report=report)


if __name__=='__main__':unittest.main()
