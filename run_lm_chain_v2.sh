#!/bin/bash
# 2026-11-17 overnight chain, v2 (qwen_lm_v2.py: review fixes). Qwen3.5-0.8B with lm_head kept, fold 0, seed 2023,
# clean data only (user: first round without noise). Starts with a 5-step v2 smoke of time SFT -> GRPO
# (runs/qwen_lm_smoke2), then the real runs in priority order; a failed run does not stop the chain.
# The text variant (slowest, most likely weakest) goes last.
#   usage (in R): nohup bash pilot/run_lm_chain_v2.sh EPOCHS_REF EPOCHS_TEXT > logs/qwen_lm_v2.log 2>&1 &
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R" || exit 1
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
EREF=${1:-4}
ETXT=${2:-3}
W=$R/runs/qwen_lm
S=$R/runs/qwen_lm_smoke2
B="env/qwen/bin/python pilot/qwen_lm_v2.py --code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-0.8B --round 0 --seed 2023"
P="$B --workdir $W"
Q="$B --workdir $S --max_steps 5 --eval_max 256 --base_n 256"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== $(date) smoke v2 time sft";  $Q --input time
echo "== $(date) smoke v2 time grpo"; $Q --input time --stage grpo --init_from $S/weights/lm_time_sft_r0_s2023.pt
echo "== $(date) SMOKE2_DONE"
echo "== $(date) A time sft";  $P --input time
echo "== $(date) A time grpo"; [ -f $W/weights/lm_time_sft_r0_s2023.pt ] && $P --input time --stage grpo --init_from $W/weights/lm_time_sft_r0_s2023.pt
echo "== $(date) C freq sft";  $P --input freq
echo "== $(date) C freq grpo"; [ -f $W/weights/lm_freq_sft_r0_s2023.pt ] && $P --input freq --stage grpo --init_from $W/weights/lm_freq_sft_r0_s2023.pt
echo "== $(date) F ref sft (epoch cap $EREF)"; $P --input ref --epochs $EREF --micro 32 --eval_bs 32
echo "== $(date) E text sft (epoch cap $ETXT)"; $P --input text --epochs $ETXT --micro 16 --eval_bs 32
echo "== $(date) CHAIN_DONE"
