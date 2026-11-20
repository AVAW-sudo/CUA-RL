"""Scripted desktop capability probe inside the pinned disposable PRoot image.

Do not run directly on a host: this script expects a private container filesystem.
"""
import json, os, pathlib, subprocess, time, signal, traceback, argparse
if os.environ.get('CUA_USERSPACE_PROBE') != '1':
    raise SystemExit('Run through probe_userspace_desktop.py in the disposable image')
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--gnome',action='store_true')
parser.add_argument('--office',action='store_true')
parser.add_argument('--applications',action='store_true')
args=parser.parse_args()
# Reap desktop services that double-fork instead of leaving zombies with container PID 1.
import ctypes
if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0) != 0:
    raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
root=pathlib.Path(args.output)
root.mkdir(exist_ok=False)
report={'status':'running','scope':'Trusted scripted GUI capability probe; not OSWorld evaluation','gnome_requested':args.gnome,'isolation':'PRoot, not a security sandbox','gpu_allocation':0}
processes=[]
started=time.monotonic()
try:
    env=dict(os.environ, DISPLAY=':94')
    xlog=(root/'xserver.log').open('w')
    processes.append(subprocess.Popen(['/usr/bin/Xvnc',':94','-depth','24','-geometry','1920x1080','-SecurityTypes','None','-localhost','yes','-rfbport','15994','-alwaysshared'],env=env,stdout=xlog,stderr=subprocess.STDOUT,start_new_session=True))
    os.environ['DISPLAY']=':94'
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        if processes[0].poll() is not None: raise RuntimeError('Xvnc exited: '+(root/'xserver.log').read_text()[-3000:])
        r=subprocess.run(['xdpyinfo','-display',':94'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if r.returncode==0:break
        time.sleep(.2)
    else: raise RuntimeError('Display not ready')
    if args.gnome:
        runtime=root/'runtime';runtime.mkdir(mode=0o700)
        env.update(XDG_RUNTIME_DIR=str(runtime),LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',
                   XDG_SESSION_TYPE='x11',XDG_SESSION_ID='1',XDG_CURRENT_DESKTOP='ubuntu:GNOME',
                   DBUS_SYSTEM_BUS_ADDRESS='unix:path='+str(runtime/'system-bus'),
                   DBUS_SESSION_BUS_ADDRESS='unix:path='+str(runtime/'session-bus'))
        for name,address in [('system',env['DBUS_SYSTEM_BUS_ADDRESS']),('session',env['DBUS_SESSION_BUS_ADDRESS'])]:
            log=(root/(name+'-bus.log')).open('w')
            # Two private buses, both session policy; no host system bus is mounted.
            processes.append(subprocess.Popen(['dbus-daemon','--session','--address='+address,'--nofork','--nopidfile'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        time.sleep(.5)
        log=(root/'logind.log').open('w')
        processes.append(subprocess.Popen(['/usr/bin/python3.10','-m','dbusmock','--system','org.freedesktop.login1','/org/freedesktop/login1','org.freedesktop.login1.Manager'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        time.sleep(1)
        code="import dbus; obj=dbus.SystemBus().get_object('org.freedesktop.login1','/org/freedesktop/login1'); obj.AddTemplate('logind',dbus.Dictionary({},signature='sv'),dbus_interface='org.freedesktop.DBus.Mock')"
        report['logind_setup_returncode']=subprocess.run(['/usr/bin/python3.10','-c',code],env=env,capture_output=True,text=True).returncode
        log=(root/'gnome.log').open('w')
        gnome=subprocess.Popen(['gnome-shell','--x11','--replace'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(gnome)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if gnome.poll() is not None: raise RuntimeError('GNOME exited: '+(root/'gnome.log').read_text()[-4000:])
            check=subprocess.run(['dbus-send','--session','--dest=org.freedesktop.DBus','--print-reply','/org/freedesktop/DBus','org.freedesktop.DBus.NameHasOwner','string:org.gnome.Shell'],env=env,capture_output=True,text=True)
            if 'boolean true' in check.stdout:
                time.sleep(5);report['gnome_ready_seconds']=time.monotonic()-started;break
            time.sleep(.5)
        else: raise RuntimeError('GNOME bus name not ready')
        os.environ.update({k:v for k,v in env.items() if k.startswith(('DBUS_','XDG_','LIBGL_','GALLIUM_'))})
    # python-xlib's loader expects an authority file even with Xvnc access control off.
    auth=root/'Xauthority';auth.touch();os.environ['XAUTHORITY']=str(auth);env['XAUTHORITY']=str(auth)
    env['CUA_PROBE_TYPED_PATH']=str(root/'typed.txt')
    env['CUA_PROBE_GEOMETRY_PATH']=str(root/'geometry.json')
    gui="""import tkinter as tk, os, json
r=tk.Tk();r.geometry('800x500+100+100');r.title('CUA trusted interaction probe')
r.configure(bg='#ddeeff')
tk.Label(r,text='CUA desktop runtime probe',font=('Sans',24),bg='#ddeeff').place(x=80,y=40)
e=tk.Entry(r,font=('Sans',20));e.place(x=60,y=130,width=600,height=50)
def save():
 open(os.environ['CUA_PROBE_TYPED_PATH'],'w').write(e.get())
b=tk.Button(r,text='Save test input',command=save,font=('Sans',20));b.place(x=60,y=230,width=300,height=60)
def record():
 r.update_idletasks()
 json.dump({'entry':[e.winfo_rootx()+30,e.winfo_rooty()+20],'button':[b.winfo_rootx()+30,b.winfo_rooty()+20]},open(os.environ['CUA_PROBE_GEOMETRY_PATH'],'w'))
r.after(500,record)
r.mainloop()
"""
    glog=(root/'gui.log').open('w')
    processes.append(subprocess.Popen(['/usr/bin/python3','-c',gui],env=env,stdout=glog,stderr=subprocess.STDOUT,start_new_session=True))
    import pyautogui
    time.sleep(2)
    if args.gnome:
        pyautogui.press('esc');time.sleep(1)
    pyautogui.screenshot().save(root/'initial.png')
    report['desktop_ready_seconds']=time.monotonic()-started
    geometry=json.loads((root/'geometry.json').read_text())
    pyautogui.click(*geometry['entry']);pyautogui.write('cua-probe-123',interval=.04);pyautogui.click(*geometry['button'])
    deadline=time.monotonic()+5
    while not (root/'typed.txt').exists() and time.monotonic()<deadline:time.sleep(.1)
    report['typed_value']=(root/'typed.txt').read_text()
    assert report['typed_value']=='cua-probe-123',report['typed_value']
    report['screenshot_control_seconds']=[]
    for i in range(10):
        start=time.monotonic(); im=pyautogui.screenshot();size=pyautogui.size()
        report['screenshot_control_seconds'].append(time.monotonic()-start)
        assert im.size==(1920,1080)
    im.save(root/'acted.png')
    # Match the VM probe shape: PNG over HTTP, then a fresh Python process importing PyAutoGUI.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import io, threading, urllib.request
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            image=pyautogui.screenshot();buf=io.BytesIO();image.save(buf,format='PNG')
            self.send_response(200);self.end_headers();self.wfile.write(buf.getvalue())
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            proc=subprocess.run(data['command'],capture_output=True,text=True)
            body=json.dumps({'returncode':proc.returncode,'output':proc.stdout,'error':proc.stderr}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.4.92',19594),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report['http_screenshot_fresh_process_control_seconds']=[]
    for i in range(10):
        start=time.monotonic()
        with opener.open('http://127.0.4.92:19594/screenshot',timeout=30) as response: assert response.read().startswith(b'\x89PNG')
        request=urllib.request.Request('http://127.0.4.92:19594/execute',data=json.dumps({'command':['python3','-c','import pyautogui; print(pyautogui.size())']}).encode(),headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=30) as response: result=json.load(response)
        assert result['returncode']==0 and 'width=1920' in result['output'],result
        report['http_screenshot_fresh_process_control_seconds'].append(time.monotonic()-start)
    server.shutdown();server.server_close();thread.join()
    if args.office:
        office_log=(root/'office.log').open('w')
        processes.append(subprocess.Popen(['libreoffice','--writer','--nologo','--norestore'],env=env,stdout=office_log,stderr=subprocess.STDOUT,start_new_session=True))
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            windows=subprocess.run(['wmctrl','-l'],env=env,capture_output=True,text=True).stdout
            candidate=next((line.split()[0] for line in windows.splitlines() if 'LibreOffice Writer' in line),None)
            if candidate:break
            time.sleep(.5)
        else:raise RuntimeError('LibreOffice Writer window not ready')
        subprocess.run(['wmctrl','-ia',candidate],env=env,check=True)
        time.sleep(2)
        pyautogui.write('CUA native office probe 123',interval=.08)
        pyautogui.hotkey('ctrl','shift','s');time.sleep(3)
        pyautogui.hotkey('ctrl','a');pyautogui.write(str(root/'probe.odt'),interval=.03);pyautogui.press('enter')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and not (root/'probe.odt').exists():time.sleep(.5)
        import zipfile
        with zipfile.ZipFile(root/'probe.odt') as archive:
            report['office_text_saved']='CUA native office probe 123' in archive.read('content.xml').decode()
        assert report['office_text_saved'],'Writer saved file lacks typed content'
        pyautogui.screenshot().save(root/'office.png')

    if args.applications:
        # Trusted local page only. Browser sandbox availability is recorded;
        # this PRoot experiment is never exposed to model-generated commands.
        page=root/'browser.html'
        page.write_text('<html><head><title>CUA browser probe</title></head><body style="font:30px sans-serif"><p>Local keyboard probe</p><input autofocus style="font:30px sans-serif" oninput="document.title=\'CUA typed: \'+this.value"></body></html>')
        import shutil
        browser=next((shutil.which(x) for x in ['google-chrome','google-chrome-stable','chromium','chromium-browser'] if shutil.which(x)),None)
        if not browser:raise RuntimeError('No Chromium browser in pinned image')
        browserlog=(root/'browser.log').open('w')
        command=[browser,'--no-sandbox','--no-first-run','--disable-dev-shm-usage','--disable-background-networking',
                 '--disable-component-update','--user-data-dir='+str(root/'browser-profile'),
                 '--remote-debugging-port=19595','--remote-debugging-address=127.0.0.1','--app=file://'+str(page)]
        processes.append(subprocess.Popen(command,env=env,stdout=browserlog,stderr=subprocess.STDOUT,start_new_session=True))
        report['browser_sandbox_disabled_for_trusted_probe']=True
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            windows=subprocess.run(['wmctrl','-l'],env=env,capture_output=True,text=True).stdout
            window=next((line.split()[0] for line in windows.splitlines() if 'CUA browser probe' in line),None)
            if window:break
            time.sleep(.5)
        else:raise RuntimeError('Browser window not ready')
        subprocess.run(['wmctrl','-ia',window],env=env,check=True);time.sleep(2)
        pyautogui.write('cua-native-browser-123',interval=.08)
        time.sleep(1)
        with opener.open('http://127.0.0.1:19595/json/list',timeout=10) as response:tabs=json.load(response)
        report['browser_keyboard_verified']=any(t.get('title')=='CUA typed: cua-native-browser-123' for t in tabs)
        assert report['browser_keyboard_verified'],tabs
        pyautogui.screenshot().save(root/'browser.png')
        calclog=(root/'calc.log').open('w')
        processes.append(subprocess.Popen(['libreoffice','--calc','--nologo','--norestore',
            '-env:UserInstallation=file://'+str(root/'calc-profile')],env=env,stdout=calclog,stderr=subprocess.STDOUT,start_new_session=True))
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            windows=subprocess.run(['wmctrl','-l'],env=env,capture_output=True,text=True).stdout
            window=next((line.split()[0] for line in windows.splitlines() if 'LibreOffice Calc' in line),None)
            if window:break
            time.sleep(.5)
        else:raise RuntimeError('Calc window not ready')
        subprocess.run(['wmctrl','-ia',window],env=env,check=True);time.sleep(2)
        pyautogui.press('esc');pyautogui.write('123\t456\n',interval=.15)
        pyautogui.hotkey('ctrl','shift','s');time.sleep(3)
        pyautogui.hotkey('ctrl','a');pyautogui.write(str(root/'probe.ods'),interval=.03);pyautogui.press('enter')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and not (root/'probe.ods').exists():time.sleep(.5)
        import zipfile
        with zipfile.ZipFile(root/'probe.ods') as archive:content=archive.read('content.xml').decode()
        report['calc_values_saved']=all('office:value="'+str(v)+'"' in content for v in [123,456])
        assert report['calc_values_saved'],content[:500]
        pyautogui.screenshot().save(root/'calc.png')

    report['status']='passed' 
except BaseException as exc:
    report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
finally:
    for p in reversed(processes):
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    report['adopted_children_reaped']=0
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        try:pid,_=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:break
        if pid>0:report['adopted_children_reaped']+=1
        else:time.sleep(.1)
    report['cleanup_passed']=all(p.poll() is not None for p in processes)
    (root/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
