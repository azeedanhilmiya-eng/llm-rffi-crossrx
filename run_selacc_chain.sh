#!/bin/bash
# 2026-11-18 (user: "先修 SFT"). Qwen3.5-0.8B, lm_head kept, freq input, full fine-tuning, seed 2023, no noise.
# SFT selected by validation ACCURACY (qwen_lm_v5.py --select acc; the run also evaluates its own loss-best state =
# the old v2 rule) -> GRPO from that SFT with v2's GRPO rule, folds 0 and 1. Question: how much does GRPO still add
# once the SFT model is not stopped too early?
# Waits until the GPU is free (shared server), checks disk space, smoke-tests v5 first (stops if the smoke fails).
# A failed fold does not stop the other.
#   usage (in R/pilot): nohup bash run_selacc_chain.sh > ../logs/selacc.log 2>&1 &
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R" || exit 1
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
echo "== $(date) start"; md5sum pilot/qwen_lm_v5.py pilot/run_selacc_chain.sh
df -h "$R" | tail -1
avail=$(df --output=avail -BG "$R" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 20 ] || { echo "== only ${avail} GB free on the project disk (need 20): stop"; exit 1; }
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 4000 ] && break
  echo "== $(date) GPU busy (${used} MiB used), waiting 5 min"; sleep 300
done
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
COMMON="--code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-0.8B --seed 2023 --input freq --tag_extra selacc"
# ---- smoke: real batch size, a few steps
WS=$R/runs/qwen_lm_v5_smoke
S="env/qwen/bin/python pilot/qwen_lm_v5.py $COMMON --workdir $WS --round 0 --eval_max 256 --base_n 256 --max_steps 6 --warmup 2"
echo "== $(date) smoke sft select acc"; $S --select acc || { echo "SMOKE_FAIL sft"; exit 1; }
echo "== $(date) smoke grpo"; $S --stage grpo --max_steps 4 --init_from $WS/weights/lm_freq_sft_selacc_r0_s2023.pt \
  || { echo "SMOKE_FAIL grpo"; exit 1; }
echo "== $(date) SMOKE_OK v5"
# ---- folds 0 and 1
W=$R/runs/qwen_lm_v5
B="env/qwen/bin/python pilot/qwen_lm_v5.py $COMMON --workdir $W"
for r in 0 1; do
  CK=$W/weights/lm_freq_sft_selacc_r${r}_s2023.pt
  echo "== $(date) sft select acc r$r"; $B --round $r --select acc
  echo "== $(date) grpo r$r"; [ -f "$CK" ] && $B --round $r --stage grpo --init_from "$CK"
done
echo "== $(date) SELACC_DONE"
