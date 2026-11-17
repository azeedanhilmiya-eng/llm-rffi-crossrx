"""Run a chain of commands one after another, each with its own log, stopping at the first failure; used as one
pool_dyn.py job in the baseline reproduction (2026-11-08), e.g. IB-RFF stage 1 -> stage 2 CR -> stage 2 CRD for one
test round (stage 2 loads stage 1's checkpoint, and CR / CRD stage 2 can share a checkpoint name, so they must not
run at the same time).
usage: python run_chain.py CHAIN.json    {"cwd": DIR, "env": {NAME: VALUE}, "steps": [[LOG, arg, arg, ...], ...]}
"""
import json
import os
import subprocess
import sys

c = json.load(open(sys.argv[1], encoding="utf-8"))
env = dict(os.environ, **c.get("env", {}))
for step in c["steps"]:
    log, args = step[0], step[1:]
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call([sys.executable] + args, cwd=c["cwd"], env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"{log} rc={rc}", flush=True)
    if rc != 0:
        sys.exit(rc)
print("CHAIN DONE", flush=True)
