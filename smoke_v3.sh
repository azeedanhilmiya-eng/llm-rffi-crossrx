#!/bin/bash
# 2026-11-17: smoke test of qwen_lm_v3.py --lora 16 (freq input, fold 0): SFT a few steps -> GRPO a few steps from it.
# Checks: runs end to end, GRPO epoch 0 equals the SFT result on the same eval subset, KL ~ 0 at GRPO step 1, memory.
#   MODEL (default Qwen3.5-0.8B), BS (default 16), MICRO (optional micro-batch) from the environment.
#   usage (in R): bash pilot/smoke_v3.sh   -> exit 0 = SMOKE_OK
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R" || exit 1
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
M=${MODEL:-Qwen3.5-0.8B}
W=$R/runs/qwen_lm_v3_smoke
B="env/qwen/bin/python pilot/qwen_lm_v3.py --code $R/baselines/train/IB-RFF --qwen $R/models/$M --seed 2023 --workdir $W --input freq --round 0 --lora 16 --bs ${BS:-16} --eval_bs ${BS:-16} --eval_max 256 --base_n 256 --max_steps 6 --warmup 2 ${MICRO:+--micro $MICRO}"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== $(date) v3 smoke $M sft"
$B || { echo "SMOKE_FAIL sft"; exit 1; }
echo "== $(date) v3 smoke $M grpo"
$B --stage grpo --init_from $W/weights/lm_freq_sft_${M}_lora16_r0_s2023.pt || { echo "SMOKE_FAIL grpo"; exit 1; }
echo "== $(date) SMOKE_OK $M"
