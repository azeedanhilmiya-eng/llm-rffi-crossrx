#!/bin/bash
# 2026-11-17 strong-baseline queue on the 4090 (started by setup_baselines.sh after its smoke test).
# Waits until BOTH earlier chains have ended: run_freq4_chain.sh (pid 400384, Qwen freq folds 1-3) and
# run_reward_chain.sh (pid 412511, reward variants C1/G1/G2 -> 4B smoke -> 4B SFT -> GRPO), then runs the 24 chains of
# chains_bl (make_chains_bl.py), 4 at a time. pool_dyn re-reads jobs.txt whenever a slot frees up ("#STOP" line = start
# nothing new). A failed chain does not stop the others.
R=/media/lib613/748f1d0b-13d9-4518-9f0e-6e3fd5b6a9e5/rffi_crossrx
cd "$R/pilot" || exit 1
echo "== $(date) waiting for pids 400384 412511"
ps -o pid,etime,args -p 400384,412511
while kill -0 412511 2>/dev/null || kill -0 400384 2>/dev/null; do sleep 60; done
echo "== $(date) earlier chains ended"
tail -3 "$R/logs/reward.log"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
"$R/env/rffi/bin/python" pool_dyn.py "$R/pilot/chains_bl/jobs.txt" "$R/runs/pool_bl" 4
grep -c "rc=0" "$R/runs/pool_bl/pool.log"
echo "== $(date) BASELINES_DONE"
