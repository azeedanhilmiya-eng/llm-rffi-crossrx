"""Frequency-domain input control for the author's UNMODIFIED IB-RFF main.py (plain FAN, stage 1), 2026-11-17.
Question: is the Qwen 'freq' gain (fold 0: CR 92.1 SFT / 93.4 GRPO vs FAN 91.9 on IQ) the frequency representation
or the language model? Same transform as qwen_lm_v2.py --input freq, applied to EVERY packet (training, validation
and test receivers alike) before the author's code sees it:
  z = I + jQ (256 samples);  Z = fftshift(fft(z)) / sqrt(256);  x' = [Re Z, Im Z]
The unitary FFT keeps the packet power, so the author's per-packet power normalisation (applied after loading) gives
the same result as normalising first. Nothing else changes (FAN, patch 16, lr 1e-3, batch 128, early stopping, folds).
The hook replaces main.load_single_dataset (main.prepare_dataset calls it by name once per (receiver, day)); the
receivers / days it transformed are printed at the end, so the log shows that test receivers went through it too.
usage: python fan_freq_run.py --code DIR --workdir DIR -- MAIN_ARGS
"""
import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--workdir", required=True)
a, rest = p.parse_known_args()
rest = [r for r in rest if r != "--"]

sys.path.insert(0, a.code)
os.makedirs(os.path.join(a.workdir, "weights"), exist_ok=True)
os.chdir(a.workdir)
sys.argv = ["main.py"] + rest
import main  # noqa: E402  (the author's main.py; nothing runs at import except definitions)


def to_freq(x):
    assert x.ndim == 3 and x.shape[1] == 2, x.shape
    z = x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64)
    Z = np.fft.fftshift(np.fft.fft(z, axis=1), axes=1) / np.sqrt(z.shape[1])
    return np.stack([Z.real, Z.imag], 1).astype(x.dtype)


conf = main.parse_args()
rx_num = 32 if conf.dataset_name == "ManyRx" else 12
rx_train, rx_test = main.split_receivers(rx_num, conf.all_test_round, conf.test_round)
orig = main.load_single_dataset
DONE = []


def hooked(dataset, rx_index, date_index, n_tx, is_eq):
    x, y = orig(dataset, rx_index, date_index, n_tx, is_eq)
    DONE.append((int(rx_index), int(date_index)))
    return to_freq(x), y


main.load_single_dataset = hooked
print(f"[fan_freq_run] workdir={a.workdir} train={sorted(int(r) for r in rx_train)} test={sorted(int(r) for r in rx_test)}",
      flush=True)
main.main()
seen = sorted(set(DONE))
test_units = [u for u in seen if u[0] in {int(r) for r in rx_test}]
print(f"[fan_freq_run] transformed {len(seen)} (receiver, day) units, {len(test_units)} of them on test receivers: "
      f"{seen}", flush=True)
