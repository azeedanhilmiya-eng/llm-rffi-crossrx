#!/bin/bash
# 2026-11-17 strong baselines (user: "你给我安排好要训练的东西，我之后就不操作服务器了"). Writes only inside R.
#  1. dataset links for WaveMLP / ASKNet (author code from bl_1117.tar.gz, extracted with --skip-old-files), work dirs
#  2. GPT-2 weights (config.json + model.safetensors, ~550 MB) from ModelScope into R/models/gpt2 (HF mirror fallback)
#  3. smoke: fold 0, 1 epoch, one chain at a time (small GPU memory next to the running Qwen job)
#  4. if IB / WaveMLP / ASKNet / adversarial pass: start run_baselines_chain.sh (waits for the reward + 4B chain)
#   usage (in R): nohup bash pilot/setup_baselines.sh > logs/bl_setup.log 2>&1 &
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R" || exit 1
echo "== $(date) code + links"
cat baselines/train/BL_SOURCES_1117.txt
for c in baselines/train/WaveMLP-RFF baselines/train/ASKNet-RFF/WiSig; do
  mkdir -p "$c/dataset"
  [ -e "$c/dataset/ManyRx" ] || ln -s "$R/baselines/wisig_data/ManyRx" "$c/dataset/ManyRx"
  echo "$c/dataset/ManyRx -> $(readlink "$c/dataset/ManyRx"): $(ls "$c/dataset/ManyRx/non_equalized/date1" | wc -l) files in date1"
done
for m in ibrff wavemlp asknet fanadv gpt2; do mkdir -p "baselines/runs_$m" "baselines/runs_${m}_smoke"; done
mkdir -p runs/bl logs
ls -la baselines/runs_align_ne/none/weights | grep "seed=2023" | grep "0.0_0_"
echo "== $(date) gpt2"
export MODELSCOPE_CACHE="$R/cache/modelscope" HF_HOME="$R/cache/hf"
if [ ! -s models/gpt2/model.safetensors ]; then
  env/qwen/bin/modelscope download --model AI-ModelScope/gpt2 config.json model.safetensors --local_dir models/gpt2 \
  || env/qwen/bin/modelscope download --model openai-community/gpt2 config.json model.safetensors --local_dir models/gpt2 \
  || HF_ENDPOINT=https://hf-mirror.com env/qwen/bin/python -c "from huggingface_hub import hf_hub_download as d; [d('openai-community/gpt2', f, local_dir='models/gpt2') for f in ('config.json', 'model.safetensors')]"
fi
ls -la models/gpt2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env/qwen/bin/python -c "
from transformers import GPT2Model
m = GPT2Model.from_pretrained('$R/models/gpt2', n_layer=6)
print('GPT2_LOAD_OK blocks', len(m.h), 'params', sum(p.numel() for p in m.parameters()))" 2>&1 | grep -v Warning | tail -3
echo "== $(date) smoke"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
P=$R/runs/pool_bl_smoke
cd "$R/pilot" || exit 1
"$R/env/rffi/bin/python" pool_dyn.py "$R/pilot/chains_bl_smoke/jobs_smoke.txt" "$P" 1
cat "$P/pool.log"
grep -H "Test Accuracy" "$R"/runs/bl/train_smoke/*.log
ok=1
for j in IB_r0 WM_r0 AK_r0 ADV_r0; do
  grep -q "END $j rc=0" "$P/pool.log" || { echo "SMOKE_FAIL $j"; tail -5 "$P/log_$j.txt"; ok=0; }
done
if grep -q "END GPT_r0 rc=0" "$P/pool.log"; then echo "SMOKE_GPT_OK"; else
  echo "SMOKE_GPT_FAIL (the queue still starts; fix gpt2_run before the pool reaches GPT_r0)"; tail -8 "$R/runs/bl/train_smoke/GPT_r0_CR_CRD.log"; fi
if [ $ok = 1 ]; then
  if pgrep -f run_baselines_chain.sh > /dev/null; then echo "waiter already running"; else
    nohup bash "$R/pilot/run_baselines_chain.sh" > "$R/logs/baselines.log" 2>&1 &
    echo "SMOKE_BL_OK, waiter pid $!"
  fi
else
  echo "SMOKE_BL_FAIL, queue NOT started"
fi
echo "== $(date) SETUP_BL_DONE"
