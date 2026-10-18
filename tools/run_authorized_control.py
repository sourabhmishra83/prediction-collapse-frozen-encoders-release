"""Finish the authorized pipeline after the first smoke process exits."""
import sys, time, subprocess, json, os, platform, traceback
from pathlib import Path
import psutil
from common import ROOT, write_json, sha256

base=ROOT/'experiments/grouped_control'
def status(phase,**extra):
    write_json(base/'execution_status.json',{'phase':phase,'updated_unix_seconds':time.time(),**extra})

def run(arguments):
    with (base/'execution.log').open('a',encoding='utf-8') as log:
        subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

try:
    smoke_pid=int(sys.argv[1]);status('waiting_for_smoke_job',smoke_pid=smoke_pid,worker_pid=os.getpid())
    while psutil.pid_exists(smoke_pid):time.sleep(15)
    job='coaid_p0.50_unweighted_ce_seed0'
    result=base/'results'/job/'result.json'
    if not result.exists():raise RuntimeError('Smoke job did not produce a completed result; full sweep not started')
    status('verifying_smoke_job',worker_pid=os.getpid())
    run(['04_analyze_grouped_control.py'])
    run(['tools/verify_grouped_checkpoint.py',job])
    status('running_remaining_jobs',worker_pid=os.getpid())
    run(['03_run_grouped_control.py','--execute','--resume'])
    status('verifying_all_results',worker_pid=os.getpid())
    run(['04_analyze_grouped_control.py'])
    summary=json.loads((base/'analysis/verification.json').read_text())
    if not summary['complete']:raise RuntimeError('Sweep exited without all planned results')
    status('completed',verified=summary)
except Exception:
    status('failed',error=traceback.format_exc());raise
