"""Chains for the frequency-domain FAN control on lib613server's RTX 4090 (2026-11-17).
FAN = IB-RFF stage 1 (alpha 0), author's ManyRx protocol (non-equalized), through fan_freq_run.py (FFT input):
  FF_freq_s2023_r<r>   seed 2023 x 4 folds, each CR train+test -> CRD test (loads the CR checkpoint)
The IQ arm is NE_none_s2023_r<r> of make_chains_4090.py (same machine, seed, folds and arguments; already run).
Logs: ROOT/runs/fanfreq/train{_smoke}/freq_s2023_r<r>_FAN_{CR,CRD}.log
--smoke: FF_freq_s2023_r0 with 1 epoch.
usage: python3 make_chains_fanfreq.py OUT_DIR [--smoke]     (OUT_DIR is ROOT/pilot/<basename OUT_DIR> on the server)
"""
import json
import os
import sys

out, smoke = sys.argv[1], "--smoke" in sys.argv
ROOT = "/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx"
CODE = f"{ROOT}/baselines/train/IB-RFF"
HOOK = f"{ROOT}/baselines/hook/fan_freq_run.py"
SEEDS = (2023,)
ROUNDS = (0,) if smoke else (0, 1, 2, 3)
EPOCHS = "1" if smoke else "200"
TAG = "_smoke" if smoke else ""
LOGS = f"{ROOT}/runs/fanfreq/train{TAG}"
WORK = f"{ROOT}/baselines/runs_fanfreq{TAG}/freq"
SUB = os.path.basename(out.rstrip("/"))
os.makedirs(out, exist_ok=True)


def step(log, seed, r, exp, state):
    return [log, HOOK, "--code", CODE, "--workdir", WORK, "--",
            "--dataset_name", "ManyRx", "--exp", exp, "--patch_size", "16", "--test_round", str(r),
            "--pre_train", "0", "--alpha", "0.0", "--code_state", state, "--epochs", EPOCHS,
            "--batch_size", "128", "--lr", "0.001", "--seed", str(seed), "--gpu", "0"]


jobs = []
for seed in SEEDS:
    for r in ROUNDS:
        tag = f"freq_s{seed}_r{r}"
        chain = {"cwd": f"{ROOT}/baselines/hook",
                 "env": {"PYTHONPATH": f"{ROOT}/baselines/_shims", "PYTHONUTF8": "1"},
                 "steps": [step(f"{LOGS}/{tag}_FAN_CR.log", seed, r, "CR", "train_test"),
                           step(f"{LOGS}/{tag}_FAN_CRD.log", seed, r, "CRD", "only_test")]}
        json.dump(chain, open(os.path.join(out, f"FF_{tag}.json"), "w"), indent=1)
        jobs.append(f"FF_{tag}\trun_chain.py {ROOT}/pilot/{SUB}/FF_{tag}.json")
name = "jobs_smoke.txt" if smoke else "jobs.txt"
open(os.path.join(out, name), "w", newline="\n").write(
    "# 4090: plain FAN on FFT input (same transform as qwen_lm_v2 --input freq), author protocol (2026-11-17)\n"
    + "\n".join(jobs) + "\n")
print(len(jobs), "chains")
