#!/bin/bash
# 2026-11-17 (师兄: GRPO reward too simple / not enough real reward -> "先用 0.8 继续把方法找好"; user: reward variants
# first, then 4B). Waits for run_freq4_chain.sh (its bash pid = $1), then, all Qwen3.5-0.8B, freq input, fold 0,
# seed 2023, no noise, full fine-tuning (as v2), workdir runs/qwen_lm (where the v2 SFT checkpoint is):
#   C1  sftc from the v2 SFT model, all units          control: is GRPO's gain just more training?
#   G1  SFT on rxA (18 rx) -> GRPO on rxB (6 held-out rx)  vs  -> GRPO on rxA (its own data)
#   G2  SFT on day1         -> GRPO on day2              vs  -> GRPO on day1
# GRPO epoch cap 20 (patience 3 decides; the held-out sets are small). Then the 4B run (as run_4b_chain.sh).
# A failed run does not stop the chain.
#   usage (in R/pilot): nohup bash run_reward_chain.sh PID > ../logs/reward.log 2>&1 &
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
P=$1
echo "== $(date) waiting for pid $P"
while [ -n "$P" ] && kill -0 "$P" 2>/dev/null; do sleep 60; done
echo "== $(date) pid $P ended; last lines of freq4.log:"; tail -3 $R/logs/freq4.log
cd "$R" || exit 1
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
W=$R/runs/qwen_lm
B="env/qwen/bin/python pilot/qwen_lm_v4.py --code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-0.8B --seed 2023 --workdir $W --input freq --round 0"
K=$W/weights
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== $(date) C1 sftc all";     $B --stage sftc --init_from $K/lm_freq_sft_r0_s2023.pt
echo "== $(date) G1 sft rxA";      $B --stage sft --units rxA
if [ -f $K/lm_freq_sft_rxA_r0_s2023.pt ]; then
  echo "== $(date) G1 grpo rxB";   $B --stage grpo --units rxB --epochs 20 --init_from $K/lm_freq_sft_rxA_r0_s2023.pt
  echo "== $(date) G1 grpo rxA";   $B --stage grpo --units rxA --epochs 20 --init_from $K/lm_freq_sft_rxA_r0_s2023.pt
fi
echo "== $(date) G2 sft day1";     $B --stage sft --units day1
if [ -f $K/lm_freq_sft_day1_r0_s2023.pt ]; then
  echo "== $(date) G2 grpo day2";  $B --stage grpo --units day2 --epochs 20 --init_from $K/lm_freq_sft_day1_r0_s2023.pt
  echo "== $(date) G2 grpo day1";  $B --stage grpo --units day1 --epochs 20 --init_from $K/lm_freq_sft_day1_r0_s2023.pt
fi
echo "== $(date) REWARD_DONE"
# ---------------- 4B (same steps as run_4b_chain.sh)
MI=""
if ! MODEL=Qwen3.5-4B BS=128 bash pilot/smoke_v3.sh; then
  echo "== $(date) 4B smoke at batch 128 failed; retrying with micro-batches of 32"
  MODEL=Qwen3.5-4B BS=128 MICRO=32 bash pilot/smoke_v3.sh || { echo "== $(date) 4B smoke failed: stop"; exit 1; }
  MI="--micro 32"
fi
W4=$R/runs/qwen_lm4b
B4="env/qwen/bin/python pilot/qwen_lm_v3.py --code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-4B --seed 2023 --workdir $W4 --input freq --round 0 --lora 16 $MI"
CK4=$W4/weights/lm_freq_sft_Qwen3.5-4B_lora16_r0_s2023.pt
echo "== $(date) 4B freq sft r0";  $B4
echo "== $(date) 4B freq grpo r0"; [ -f $CK4 ] && $B4 --stage grpo --init_from $CK4
echo "== $(date) CHAIN_ALL_DONE"
