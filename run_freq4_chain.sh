#!/bin/bash
# 2026-11-17 (user: "4 折 + 频域对照"). Seed 2023, clean data only.
#  1. control: plain FAN on FFT input, folds 0-3 (pool of 3, ~45 min) -> is the freq gain the representation or Qwen?
#  2. Qwen3.5-0.8B lm_head kept, freq input, SFT -> GRPO on folds 1, 2, 3 (fold 0 ran on 11-17 00:41-03:57).
# A failed run does not stop the chain.
#   usage (in R/pilot): nohup bash run_freq4_chain.sh > ../logs/freq4.log 2>&1 &
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R/pilot" || exit 1
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== $(date) FAN freq pool (4 folds)"
$R/env/rffi/bin/python pool_dyn.py $R/pilot/chains_fanfreq/jobs.txt $R/runs/pool_fanfreq 3
echo "== $(date) FAN freq done"
cd "$R" || exit 1
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
W=$R/runs/qwen_lm
B="env/qwen/bin/python pilot/qwen_lm_v2.py --code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-0.8B --seed 2023 --workdir $W"
for r in 1 2 3; do
  echo "== $(date) C freq sft r$r";  $B --input freq --round $r
  echo "== $(date) C freq grpo r$r"; [ -f $W/weights/lm_freq_sft_r${r}_s2023.pt ] && $B --input freq --round $r --stage grpo --init_from $W/weights/lm_freq_sft_r${r}_s2023.pt
done
echo "== $(date) CHAIN_DONE"
