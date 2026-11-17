"""Like pool.py, but the jobs file is re-read every time a slot frees up, so jobs can be appended, reordered or
removed while the pool runs (overnight queue, 2026-11-05). A job runs once: names already started are skipped.
A line "#STOP" makes the pool start nothing new (running jobs finish).

jobs file: one job per line, "LOGNAME<TAB>arguments for python" (blank lines and other # comments ignored).
usage: python pool_dyn.py JOBS.txt OUTDIR [N]
Writes OUTDIR/pool.log (START / END of every job, "POOL DONE" when nothing is left to start or running).
"""
import os
import subprocess
import sys
import time

if os.name == "nt":  # keep the laptop awake while the pool runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
jobs_file, outdir = sys.argv[1], sys.argv[2]
n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.makedirs(outdir, exist_ok=True)
plog = open(os.path.join(outdir, "pool.log"), "a", encoding="utf-8")
note = lambda msg: (plog.write(f"{time.strftime('%H:%M:%S')} {msg}\n"), plog.flush())


def pending(started):
    lines = [ln.rstrip("\n") for ln in open(jobs_file, encoding="utf-8")]
    if any(ln.strip() == "#STOP" for ln in lines):
        return []
    jobs = [ln.split("\t", 1) for ln in lines if ln.strip() and not ln.startswith("#") and "\t" in ln]
    return [(name, cmd) for name, cmd in jobs if name not in started]


started, running = set(), []
while True:
    for j in [j for j in running if j[1].poll() is not None]:
        note(f"END {j[0]} rc={j[1].returncode}")
        running.remove(j)
    todo = pending(started)
    while todo and len(running) < n_max:
        name, cmd = todo.pop(0)
        f = open(os.path.join(outdir, f"log_{name}.txt"), "w", encoding="utf-8")
        running.append((name, subprocess.Popen([sys.executable] + cmd.split(), stdout=f, stderr=subprocess.STDOUT)))
        started.add(name)
        note(f"START {name}")
        time.sleep(10)
    if not todo and not running:
        break
    time.sleep(15)
note("POOL DONE")
