# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Run original-data preparation across QS Pods with separate cloud mounts.

Run from the local Relax checkout with qs-cli installed. The source Pod only
reads original inputs; only new preparation files and a temporary authenticated
file-transfer endpoint are created. No Ray/Gym/training operations are issued.
"""

import argparse
import base64
import gzip
import json
import pathlib
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any


parser = argparse.ArgumentParser(description=__doc__)
for name in (
    "qs-command",
    "source-pod",
    "source-container",
    "target-pod",
    "target-container",
    "target-repo",
    "source-root",
    "output-root",
    "bbs-wheel",
    "dressage-wheel",
    "image-map",
):
    parser.add_argument("--" + name, required=True)
arguments = parser.parse_args()
ROOT = pathlib.Path(__file__).resolve().parents[3]
QS = shlex.split(arguments.qs_command) + ["exec"]
SOURCE = QS + [arguments.source_pod, "-c", arguments.source_container, "--", "python3", "-u", "-c"]
TARGET = QS + [arguments.target_pod, "-c", arguments.target_container, "--", "python3", "-u", "-c"]
REPO = arguments.target_repo.rstrip("/")
BUNDLE = REPO + "/repro/claude-code-harbor-r5"
DATA = arguments.output_root
SOURCE_DATA = arguments.source_root
STAGE = "/tmp/relax-original-data-" + uuid.uuid4().hex


def progress(stage: str, **extra: Any) -> None:
    value = {"stage": stage, **extra}
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


def call(where: list[str], code: str) -> Any:
    result = subprocess.run(where + [code], capture_output=True, text=True, timeout=150)
    # QS adds terminal line endings and a completion line. Only our framed value is read.
    for line in result.stdout.splitlines():
        if line.startswith("VALUE:"):
            return json.loads(line[6:])
    raise RuntimeError("Remote operation returned no framed result; remote output suppressed")


def put(where: list[str], path: str, content: bytes) -> None:
    for i in range(0, len(content), 48000):
        chunk = base64.b64encode(content[i : i + 48000]).decode()
        call(
            where,
            f"import pathlib,base64,json;p=pathlib.Path({path!r});p.parent.mkdir(parents=True,exist_ok=True);f=p.open({'wb' if i == 0 else 'ab'!r});f.write(base64.b64decode({chunk!r}));f.close();print('VALUE:'+json.dumps(True))",
        )


progress("stage_original_inputs")
call(SOURCE, f"import pathlib,json;p=pathlib.Path({STAGE!r});p.mkdir(mode=0o700);print('VALUE:'+json.dumps(True))")
image_map = call(
    TARGET,
    f"import pathlib,base64,json;p=pathlib.Path({arguments.image_map!r});print('VALUE:'+json.dumps(base64.b64encode(p.read_bytes()).decode()))",
)
image_bytes = base64.b64decode(image_map)
if image_bytes.startswith(b"\x1f\x8b"):
    image_bytes = gzip.decompress(image_bytes)
patch = (ROOT / "examples/claude_code_harbor/mimo_git_reflog.patch").read_bytes()
code = (ROOT / "examples/claude_code_harbor/scripts/prepare_data.py").read_bytes()
put(TARGET, REPO + "/examples/claude_code_harbor/scripts/prepare_data.py", code)
put(TARGET, BUNDLE + "/input-config/ordinary-image-map.json", image_bytes)
put(SOURCE, STAGE + "/ordinary-image-map.json", image_bytes)
put(SOURCE, STAGE + "/mimo_git_reflog.patch", patch)
worker = f"""import argparse,base64,json,pathlib,tarfile,traceback
stage=pathlib.Path({STAGE!r})
try:
 scope={{'__name__':'prepare_from_original','__file__':{REPO + "/examples/claude_code_harbor/scripts/prepare_data.py"!r}}}
 exec(compile(base64.b64decode({base64.b64encode(code).decode()!r}),scope['__file__'],'exec'),scope)
 args=argparse.Namespace(source_root=pathlib.Path({SOURCE_DATA!r}),output_root=stage/'prepared',published_root=pathlib.Path({DATA!r}),bbs_wheel=pathlib.Path({arguments.bbs_wheel!r}),dressage_wheel=pathlib.Path({arguments.dressage_wheel!r}),image_map=stage/'ordinary-image-map.json',runtime_patch=stage/'mimo_git_reflog.patch',expected_rows=2683)
 result=scope['prepare'](args)
 with tarfile.open(stage/'prepared.tar.gz','w:gz',compresslevel=1) as archive:
  for p in sorted((stage/'prepared').rglob('*')):
   if p.is_file():archive.add(p,arcname=str(p.relative_to(stage/'prepared')),recursive=False)
 result['archive_bytes']=(stage/'prepared.tar.gz').stat().st_size
except Exception as e:
 result={{'status':'FAIL','error_class':type(e).__name__,'error':str(e)}}
(stage/'build-status.json').write_text(json.dumps(result))
"""
call(
    SOURCE,
    f"import pathlib,subprocess,json;p=pathlib.Path({STAGE!r});f=(p/'build.log').open('w');c=subprocess.Popen(['python3','-u','-c',{worker!r}],stdout=f,stderr=f,start_new_session=True);print('VALUE:'+json.dumps({{'started':True}}))",
)
progress("processing_original_dataset")
for _ in range(120):
    result = call(
        SOURCE,
        f"import pathlib,json;p=pathlib.Path({STAGE + '/build-status.json'!r});print('VALUE:'+p.read_text() if p.exists() else 'VALUE:null')",
    )
    if result:
        break
    time.sleep(3)
else:
    raise RuntimeError("Original-data preparation exceeded transfer budget")
if result["status"] != "PASS":
    progress("source_build_failed", **result)
    call(SOURCE, f"import pathlib,shutil,json;shutil.rmtree({STAGE!r});print('VALUE:'+json.dumps(True))")
    sys.exit(1)
progress("original_dataset_prepared", **result)
# A short-lived, authenticated endpoint serves exactly one newly-created archive.
server = f"""import pathlib,http.server,threading,secrets,json,socket,os
stage=pathlib.Path({STAGE!r});token=secrets.token_urlsafe(32);archive=stage/'prepared.tar.gz'
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path!='/archive' or self.headers.get('Authorization')!='Bearer '+token:
   self.send_error(403);return
  self.send_response(200);self.send_header('Content-Length',str(archive.stat().st_size));self.end_headers()
  with archive.open('rb') as f:
   while data:=f.read(1024*1024):self.wfile.write(data)
 def log_message(self,*args):pass
httpd=http.server.ThreadingHTTPServer(('0.0.0.0',0),Handler)
(stage/'transfer-private.json').write_text(json.dumps({{'host':socket.gethostbyname(socket.gethostname()),'port':httpd.server_port,'token':token,'pid':os.getpid()}}));(stage/'transfer-private.json').chmod(0o600)
threading.Timer(1200,httpd.shutdown).start();httpd.serve_forever()
"""
call(
    SOURCE,
    f"import pathlib,subprocess,json;p=pathlib.Path({STAGE!r});f=(p/'transfer.log').open('w');c=subprocess.Popen(['python3','-u','-c',{server!r}],stdout=f,stderr=f,start_new_session=True);print('VALUE:'+json.dumps(True))",
)
for _ in range(10):
    private = call(
        SOURCE,
        f"import pathlib,json;p=pathlib.Path({STAGE + '/transfer-private.json'!r});print('VALUE:'+p.read_text() if p.exists() else 'VALUE:null')",
    )
    if private:
        break
    time.sleep(1)
progress("copying_prepared_dataset_to_cloud")
# Credentials are held in this process and never printed or saved into the report.
transfer = f"""import pathlib,urllib.request,json,tempfile,tarfile,os,shutil
out=pathlib.Path({DATA!r});out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise FileExistsError('Prepared output already exists')
stage=pathlib.Path(tempfile.mkdtemp(prefix=out.name+'.incoming-',dir=out.parent))
try:
 archive=stage/'data.tar.gz'
 request=urllib.request.Request({"http://" + private["host"] + ":" + str(private["port"]) + "/archive"!r},headers={{'Authorization':{"Bearer " + private["token"]!r}}})
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
 with opener.open(request,timeout=120) as source,archive.open('wb') as target:shutil.copyfileobj(source,target,1024*1024)
 payload=stage/'payload';payload.mkdir()
 from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
 def write_member(path,content,mode):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);path.chmod(mode & 0o777)
 with ThreadPoolExecutor(max_workers=16) as pool,tarfile.open(archive,'r|gz') as tar:
  pending=set()
  for member in tar:
   path=(payload/member.name).resolve()
   if not member.isfile() or not path.is_relative_to(payload):raise ValueError('Invalid transfer member')
   pending.add(pool.submit(write_member,path,tar.extractfile(member).read(),member.mode))
   if len(pending)>=32:
    done,pending=wait(pending,return_when=FIRST_COMPLETED)
    for item in done:item.result()
  for item in pending:item.result()
 import hashlib
 records=json.loads((payload/'file-manifest.json').read_text())['records']
 def check_record(record):
  p=(payload/record['path']).resolve()
  if not p.is_relative_to(payload) or p.stat().st_size!=record['bytes']:raise ValueError('Transfer file size differs')
  h=hashlib.sha256()
  with p.open('rb') as f:
   while data:=f.read(8*1024*1024):h.update(data)
  if h.hexdigest()!=record['sha256']:raise ValueError('Transfer checksum differs')
 with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(check_record,records))
 os.rename(payload,out)
 result={{'status':'PASS','archive_bytes':archive.stat().st_size}}
finally:shutil.rmtree(stage)
print('VALUE:'+json.dumps(result))
"""
try:
    transfer_status = BUNDLE + "/provenance/prepared-data-transfer.json"
    transfer_worker = (
        "import pathlib,json\ntry:\n exec("
        + repr(transfer)
        + ")\n value=result\nexcept Exception as e:\n value={'status':'FAIL','error_class':type(e).__name__,'http_status':getattr(e,'code',None)}\npathlib.Path("
        + repr(transfer_status)
        + ").write_text(json.dumps(value))\n"
    )
    call(
        TARGET,
        f"import pathlib,subprocess,json;p=pathlib.Path({transfer_status!r});p.parent.mkdir(parents=True,exist_ok=True);p.unlink(missing_ok=True);f=(p.parent/'prepared-data-transfer.log').open('w');subprocess.Popen(['python3','-u','-c',{transfer_worker!r}],stdout=f,stderr=f,start_new_session=True);print('VALUE:'+json.dumps(True))",
    )
    for _ in range(180):
        copied = call(
            TARGET,
            f"import pathlib,json;p=pathlib.Path({transfer_status!r});print('VALUE:'+p.read_text() if p.exists() else 'VALUE:null')",
        )
        if copied:
            break
        time.sleep(3)
    else:
        raise RuntimeError("Cloud transfer did not finish within budget")
    if copied["status"] != "PASS":
        progress("cloud_copy_failed", **copied)
        raise RuntimeError("Cloud file transfer failed")
    progress("cloud_copy_complete", **copied)
finally:
    call(
        SOURCE,
        f"import pathlib,json,os,signal,shutil;p=pathlib.Path({STAGE!r});v=json.loads((p/'transfer-private.json').read_text());cmd=(pathlib.Path('/proc')/str(v['pid'])/'cmdline').read_text();assert str(p) in cmd;os.kill(v['pid'],signal.SIGTERM);shutil.rmtree(p);print('VALUE:'+json.dumps(True))",
    )
progress("complete", rows=result["rows"], unique_uploads=result["unique_uploads"], test_dirs=result["test_dirs"])
