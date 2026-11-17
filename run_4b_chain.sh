#!/bin/bash
# 2026-11-17 (user "换4b试试", queued after the 0.8B folds; 师兄: 4B first, then 5-6 strong baselines).
# Waits for run_freq4_chain.sh (its bash pid = $1) to end, then Qwen3.5-4B with the lm_head kept, LoRA 16, freq input,
# fold 0, seed 2023, no noise: smoke at the real batch size (falls back to micro-batches of 32 if that fails) -> SFT -> GRPO.
#   usage (in R/pilot): nohup bash run_4b_chain.sh PID > ../logs/qwen4b.log 2>&1 &
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
P=$1
echo "== $(date) waiting for pid $P"
while kill -0 "$P" 2>/dev/null; do sleep 60; done
echo "== $(date) pid $P ended; last lines of freq4.log:"; tail -3 $R/logs/freq4.log
cd "$R" || exit 1
MI=""
if ! MODEL=Qwen3.5-4B BS=128 bash pilot/smoke_v3.sh; then
  echo "== $(date) smoke at batch 128 failed; retrying with micro-batches of 32"
  MODEL=Qwen3.5-4B BS=128 MICRO=32 bash pilot/smoke_v3.sh || { echo "== $(date) 4B smoke failed: stop"; exit 1; }
  MI="--micro 32"
fi
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
W=$R/runs/qwen_lm4b
B="env/qwen/bin/python pilot/qwen_lm_v3.py --code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-4B --seed 2023 --workdir $W --input freq --round 0 --lora 16 $MI"
CK=$W/weights/lm_freq_sft_Qwen3.5-4B_lora16_r0_s2023.pt
echo "== $(date) 4B freq sft r0"; $B
echo "== $(date) 4B freq grpo r0"; [ -f $CK ] && $B --stage grpo --init_from $CK
echo "== $(date) CHAIN4B_DONE"
