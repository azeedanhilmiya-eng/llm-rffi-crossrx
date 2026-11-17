#!/bin/bash
# 2026-11-14 千问试点的环境和模型（用户已同意）。只写本项目目录 R，不改 R/env/rffi、不写 /home。
# 新环境 R/env/qwen：venv，继承 R/env/rffi 的 torch 2.5.1 / numpy / sklearn（--system-site-packages），
# 只在新环境里装 transformers>=5.3（千问 3.5 需要）和 modelscope（下载模型用）。
# 模型从 ModelScope 下到 R/models：Qwen3.5-0.8B（约 1.8 GB），Qwen3.5-4B（约 9.3 GB）。
# 用法（在 R 里）：nohup bash pilot/setup_qwen.sh > logs/qwen_setup.log 2>&1 &
set -e
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R"
mkdir -p logs models cache/pip cache/modelscope cache/hf
export PIP_CACHE_DIR="$R/cache/pip" MODELSCOPE_CACHE="$R/cache/modelscope" HF_HOME="$R/cache/hf"
echo "== $(date) venv"
[ -x env/qwen/bin/python ] || "$R/env/rffi/bin/python" -m venv --system-site-packages env/qwen
echo "== $(date) pip"
env/qwen/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=5.3" modelscope
env/qwen/bin/python -c "import transformers, torch, numpy; print('transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"
"$R/env/rffi/bin/python" -c "import torch, numpy; print('rffi env unchanged: torch', torch.__version__, 'numpy', numpy.__version__)"
for m in Qwen3.5-0.8B Qwen3.5-4B; do
  echo "== $(date) download $m"
  env/qwen/bin/modelscope download --model "Qwen/$m" --local_dir "models/$m"
  du -sh "models/$m"
done
echo "== $(date) SETUP DONE"
