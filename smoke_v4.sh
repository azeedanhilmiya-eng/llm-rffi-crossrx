#!/bin/bash
# 2026-11-17: smoke test of qwen_lm_v4.py (held-out units, sftc), Qwen3.5-0.8B with LoRA 16 and batch 16 so it fits
# next to the running 0.8B chain (~2-3 GB): sft on rxA -> grpo on rxB -> sftc on day2, a few steps each.
#   usage (in R): bash pilot/smoke_v4.sh   -> exit 0 = SMOKE_OK
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R" || exit 1
export PYTHONPATH=$R/baselines/_shims PYTHONUTF8=1 HF_HOME=$R/cache/hf
W=$R/runs/qwen_lm_v4_smoke
B="env/qwen/bin/python pilot/qwen_lm_v4.py --code $R/baselines/train/IB-RFF --qwen $R/models/Qwen3.5-0.8B --seed 2023 --workdir $W --input freq --round 0 --lora 16 --bs 16 --eval_bs 16 --eval_max 256 --base_n 256 --max_steps 6 --warmup 2"
CK=$W/weights/lm_freq_sft_rxA_Qwen3.5-0.8B_lora16_r0_s2023.pt
echo "== $(date) v4 smoke sft rxA";  $B --units rxA || { echo "SMOKE_FAIL sft"; exit 1; }
echo "== $(date) v4 smoke grpo rxB"; $B --units rxB --stage grpo --init_from $CK || { echo "SMOKE_FAIL grpo"; exit 1; }
echo "== $(date) v4 smoke sftc day2"; $B --units day2 --stage sftc --init_from $CK || { echo "SMOKE_FAIL sftc"; exit 1; }
echo "== $(date) SMOKE_OK v4"
