"""Strong-baseline chains for lib613server's RTX 4090 (2026-11-17; plan: 03_experiments/强基线方案_11-17.md).
All on the author's ManyRx protocol (non-equalized, first 100 packets, 4 fixed folds, train days 1-2, CR = test rx
days 1-2, CRD = test rx days 3-4), seed 2023, 4 folds, every method trained once per fold and tested on CR and CRD:
  IB_r<r>    IB-RFF stage 2 (TIFS 2026; author's main.py through align_run.py --align none, author alpha per fold from
             run_all.sh), starting from the 4090's own stage-1 FAN checkpoint (NE_none_s2023, copied, not retrained)
  WM_r<r>    WaveMLP (IMNS 2026; author's main.py, --wavelet_levels 5 --wavelet_kernel_size 32): CRD train+test,
             then the same checkpoint tested on CR
  AK_r<r>    ASKNet (IEEE CL 2026; author's main.py, defaults): CRD train+test, then CR test of the same checkpoint
  ADV_r<r>   Shen et al. TMC 2024 receiver-adversarial term (gradient reversal, coefficient 1) on FAN (fan_adv_run.py)
  GPT_r<r>   frozen pre-trained GPT-2, One Fits All recipe (gpt2_run.py, R/env/qwen)
  ADV0_r<r>  control: fan_adv_run.py with --lam 0 (same script, no adversarial gradient) -> isolates the term
Logs: ROOT/runs/bl/train{_smoke}/<chain>_<step>.log; work directories ROOT/baselines/runs_<method>{_smoke}.
--smoke: fold 0, 1 epoch, ADV0 left out (same script as ADV).
usage: python3 make_chains_bl.py OUT_DIR [--smoke]     (OUT_DIR is ROOT/pilot/<basename OUT_DIR> on the server)
"""
import json
import os
import sys

out, smoke = sys.argv[1], "--smoke" in sys.argv
ROOT = "/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx"
B = f"{ROOT}/baselines"
IB, WM, AK = f"{B}/train/IB-RFF", f"{B}/train/WaveMLP-RFF", f"{B}/train/ASKNet-RFF/WiSig"
HOOK = f"{B}/hook"
SEED = "2023"
ROUNDS = (0,) if smoke else (0, 1, 2, 3)
EP = ["--epochs", "1"] if smoke else []
EPOCHS = "1" if smoke else "200"
TAG = "_smoke" if smoke else ""
LOGS = f"{ROOT}/runs/bl/train{TAG}"
W = {m: f"{B}/runs_{m}{TAG}" for m in ("ibrff", "wavemlp", "asknet", "fanadv", "gpt2")}
ALPHA = {"CR": (0.08, 0.008, 0.004, 0.01), "CRD": (0.01, 0.008, 0.08, 0.004)}   # run_all.sh, ManyRx, per round
STAGE1 = f"{B}/runs_align_ne/none/weights"   # 4090 NE_none chains (make_chains_4090.py), seed 2023 = FAN-IQ reference
SUB = os.path.basename(out.rstrip("/"))
ENV_SH = {"PYTHONPATH": f"{B}/_shims", "PYTHONUTF8": "1"}
ENV_Q = dict(ENV_SH, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=f"{ROOT}/cache/hf")
os.makedirs(out, exist_ok=True)


def ib_main(exp, r, alpha):
    return [f"{HOOK}/align_run.py", "--code", IB, "--workdir", f"{W['ibrff']}/none", "--align", "none", "--",
            "--dataset_name", "ManyRx", "--exp", exp, "--patch_size", "16", "--test_round", str(r), "--pre_train", "1",
            "--alpha", str(alpha), "--code_state", "train_test", "--epochs", EPOCHS, "--batch_size", "128",
            "--lr", "0.001", "--seed", SEED, "--gpu", "0"]


def author(code, exp, r, state, extra=()):
    return [f"{code}/main.py", "--dataset_name", "ManyRx", "--exp", exp, "--test_round", str(r), *extra,
            "--code_state", state, "--seed", SEED, "--gpu", "0", *EP]


chains = {}
for r in ROUNDS:
    L = lambda name, step: f"{LOGS}/{name}_{step}.log"
    s1 = f"FAN_DSCNN_16_128_2.0_0.0_0_ManyRx_[1, 2]_test_round={r}_seed={SEED}.pth"
    chains[f"IB_r{r}"] = (f"{HOOK}", ENV_SH, [
        [L(f"IB_r{r}", "copy_stage1"), f"{HOOK}/cp_ckpt.py", "copy", f"{STAGE1}/{s1}", f"{W['ibrff']}/none/weights/{s1}"],
        [L(f"IB_r{r}", f"CR_a{ALPHA['CR'][r]}")] + ib_main("CR", r, ALPHA["CR"][r]),
        [L(f"IB_r{r}", f"CRD_a{ALPHA['CRD'][r]}")] + ib_main("CRD", r, ALPHA["CRD"][r])])
    wm_x = ("--wavelet_levels", "5", "--wavelet_kernel_size", "32")
    chains[f"WM_r{r}"] = (W["wavemlp"], {"PYTHONUTF8": "1"}, [
        [L(f"WM_r{r}", "CRD")] + author(WM, "CRD", r, "train_test", wm_x),
        [L(f"WM_r{r}", "swap"), f"{HOOK}/cp_ckpt.py", "swap", "weights", "ManyRx_CRD_", "ManyRx_CR_",
         f"round{r}_seed{SEED}"],
        [L(f"WM_r{r}", "CR")] + author(WM, "CR", r, "only_test", wm_x)])
    chains[f"AK_r{r}"] = (W["asknet"], {"PYTHONUTF8": "1"}, [
        [L(f"AK_r{r}", "CRD")] + author(AK, "CRD", r, "train_test"),
        [L(f"AK_r{r}", "swap"), f"{HOOK}/cp_ckpt.py", "swap", "weights", "ManyRx_CRD_", "ManyRx_CR_",
         f"round{r}_seed{SEED}"],
        [L(f"AK_r{r}", "CR")] + author(AK, "CR", r, "only_test")])
    for name, lam in (("ADV", "1"), ("ADV0", "0")):
        if smoke and name == "ADV0":
            continue
        chains[f"{name}_r{r}"] = (f"{HOOK}", ENV_SH, [
            [L(f"{name}_r{r}", "CR_CRD"), f"{HOOK}/fan_adv_run.py", "--code", IB, "--workdir", W["fanadv"],
             "--round", str(r), "--lam", lam, "--seed", SEED, "--epochs", EPOCHS]])
    chains[f"GPT_r{r}"] = (f"{HOOK}", ENV_Q, [
        [L(f"GPT_r{r}", "CR_CRD"), f"{HOOK}/with_py.py", f"{ROOT}/env/qwen/bin/python", f"{HOOK}/gpt2_run.py",
         "--code", IB, "--workdir", W["gpt2"], "--gpt2", f"{ROOT}/models/gpt2", "--round", str(r), "--seed", SEED,
         "--epochs", EPOCHS]])

ORDER = ("IB", "WM", "AK", "ADV", "GPT", "ADV0")
names = sorted(chains, key=lambda n: (ORDER.index(n.split("_")[0]), n))
jobs = []
for n in names:
    cwd, env, steps = chains[n]
    json.dump({"cwd": cwd, "env": env, "steps": steps}, open(os.path.join(out, f"{n}.json"), "w"), indent=1)
    jobs.append(f"{n}\trun_chain.py {ROOT}/pilot/{SUB}/{n}.json")
name = "jobs_smoke.txt" if smoke else "jobs.txt"
open(os.path.join(out, name), "w", newline="\n").write(
    "# 4090 strong baselines, author's ManyRx protocol, seed 2023 (2026-11-17): IB-RFF stage 2, WaveMLP, ASKNet, "
    "Shen-adversarial FAN, frozen GPT-2, lam-0 control\n" + "\n".join(jobs) + "\n")
print(len(jobs), "chains:", " ".join(names))
