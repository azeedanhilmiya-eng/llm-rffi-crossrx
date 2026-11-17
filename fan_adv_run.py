"""Receiver-adversarial training (Shen et al., "Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint
Identification", IEEE TMC 2024) on the FAN backbone of IB-RFF, author's ManyRx protocol (2026-11-17).
Taken from Shen et al.: a receiver classifier (dense 128, ReLU, softmax over the TRAINING receivers) on the feature,
behind a gradient-reversal layer with coefficient --lam (theirs: 1), the two cross-entropies added 1 : 1, so the feature
extractor is pushed to make the training receivers indistinguishable.
Kept from the FAN / IB-RFF protocol (the author's main.py is imported, not re-implemented, where possible): FAN (patch 16,
embed 128, DSCNN bottleneck, mlp ratio 2), the data loader (first 100 packets per tx per (rx, day), per-packet power
normalisation, non-equalized), the receiver split, the 70/30 train/validation split of the training receivers' packets
(same sklearn call and random_state, so the same packets; asserted against main.prepare_dataset), Adam lr 1e-3, batch
128, ReduceLROnPlateau (0.1, 5, min 1e-6), early stopping (10) and model selection on the validation TRANSMITTER loss,
at most 200 epochs.
Not taken from Shen: their ResNet on LoRa spectrograms (the backbone is FAN here, so the comparison isolates the
adversarial term), their SGD and AWGN augmentation, their 5-packet fusion (single-packet decisions here).
The feature fed to both heads is FAN's pooled embedding (the input of cls_head, 128-d), captured with a forward hook.
--lam 0 is the control: same script, the receiver head still trains, but no gradient from it reaches the backbone.
One run trains once and tests CR (test receivers, days 1-2) and CRD (test receivers, days 3-4).
usage: python fan_adv_run.py --code IBRFF_DIR --workdir DIR --round r [--lam 1] [--seed 2023] [--epochs 200]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--workdir", required=True)
p.add_argument("--round", type=int, required=True)
p.add_argument("--lam", type=float, default=1.0)
p.add_argument("--seed", type=int, default=2023)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--patch_size", type=int, default=16)
a = p.parse_args()
sys.path.insert(0, a.code)
os.makedirs(os.path.join(a.workdir, "weights"), exist_ok=True)
os.chdir(a.workdir)
sys.argv = ["main.py"]
import main  # noqa: E402  (the author's IB-RFF main.py; nothing runs at import except definitions)

t0 = time.time()
TAG = f"fanadv_lam{a.lam:g}_r{a.round}_s{a.seed}"
main.setup_seed(a.seed)
rx_train, rx_test = main.split_receivers(32, 4, a.round)
print(f"[fan_adv_run] {TAG} workdir={a.workdir} train={rx_train} test={rx_test}", flush=True)


def units(rxs, days):
    xs, ys, rs = [], [], []
    for i, rx in enumerate(rxs):
        for d in days:
            x, y = main.load_single_dataset("ManyRx", rx, d, 10, "non_equalized")
            xs.append(x)
            ys.append(y)
            rs.append(np.full(len(y), i))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)


x, y, r = units(rx_train, [1, 2])
x_tr, x_va, y_tr, y_va, r_tr, r_va = train_test_split(x, y, r, test_size=0.3, random_state=a.seed)
ref = main.prepare_dataset("ManyRx", rx_train, [1, 2], 10, "non_equalized", True, a.seed)
assert np.array_equal(ref[0], x_tr) and np.array_equal(ref[1], x_va) and np.array_equal(ref[2], y_tr) \
    and np.array_equal(ref[3], y_va), "split differs from the author's prepare_dataset"
del ref, x, y, r
print(f"[fan_adv_run] train {len(y_tr)} / val {len(y_va)} packets (same split as the author's code)", flush=True)

ten = lambda v, dt: torch.tensor(v, dtype=dt)
train_loader = DataLoader(TensorDataset(ten(x_tr, torch.float32), ten(y_tr, torch.long), ten(r_tr, torch.long)),
                          batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(ten(x_va, torch.float32), ten(y_va, torch.long), ten(r_va, torch.long)),
                        batch_size=128, shuffle=False)

model = main.choose_backbone("FAN", "DSCNN", 10, a.patch_size, 128, 2.0).cuda()
FEAT = {}
model.cls_head.register_forward_pre_hook(lambda m, inp: FEAT.__setitem__("f", inp[0]))


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, lam):
        ctx.lam = lam
        return v.view_as(v)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


rx_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(rx_train))).cuda()
opt = torch.optim.Adam(list(model.parameters()) + list(rx_head.parameters()), lr=1e-3)
sched = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, min_lr=1e-6)


def run_epoch(loader, train):
    model.train(train)
    rx_head.train(train)
    n, ltx_s, lrx_s, ctx_s, crx_s = 0, 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb, rb in loader:
            xb, yb, rb = xb.cuda(), yb.cuda(), rb.cuda()
            out = model(xb)[1]
            orx = rx_head(GRL.apply(FEAT["f"], a.lam))
            ltx, lrx = F.cross_entropy(out, yb), F.cross_entropy(orx, rb)
            if train:
                opt.zero_grad()
                (ltx + lrx).backward()
                opt.step()
            k = len(yb)
            n += k
            ltx_s += ltx.item() * k
            lrx_s += lrx.item() * k
            ctx_s += (out.argmax(1) == yb).sum().item()
            crx_s += (orx.argmax(1) == rb).sum().item()
    return ltx_s / n, lrx_s / n, 100.0 * ctx_s / n, 100.0 * crx_s / n


best, wait, hist, ep = float("inf"), 0, [], 0
ck = os.path.join("weights", TAG + ".pt")
for ep in range(1, a.epochs + 1):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    sched.step(va[0])
    hist.append(dict(ep=ep, train=tr, val=va, lr=opt.param_groups[0]["lr"]))
    print(f"ep {ep} train tx_loss {tr[0]:.4f} rx_loss {tr[1]:.4f} acc {tr[2]:.2f} rx_acc {tr[3]:.2f} | "
          f"val tx_loss {va[0]:.4f} rx_loss {va[1]:.4f} acc {va[2]:.2f} rx_acc {va[3]:.2f} "
          f"(chance rx_acc {100 / len(rx_train):.1f}; {(time.time() - t0) / 60:.1f} min)", flush=True)
    if va[0] < best:
        best, wait = va[0], 0
        torch.save({"model": model.state_dict(), "rx_head": rx_head.state_dict(), "ep": ep}, ck)
    else:
        wait += 1
        if wait >= 10:
            print(f"Early stopping triggered at epoch {ep}.", flush=True)
            break

sd = torch.load(ck, map_location="cuda")
model.load_state_dict(sd["model"])
model.eval()
res = dict(method="fan_adv (Shen TMC24 receiver-adversarial on FAN)", lam=a.lam, round=a.round, seed=a.seed,
           best_ep=sd["ep"], epochs_run=ep, best_val_tx_loss=best, rx_train=rx_train, rx_test=rx_test)
for exp, days in (("CR", [1, 2]), ("CRD", [3, 4])):
    xt, yt = main.prepare_dataset("ManyRx", rx_test, days, 10, "non_equalized", False, a.seed)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(yt), 512):
            pred = model(torch.tensor(xt[i:i + 512], dtype=torch.float32).cuda())[1].argmax(1).cpu().numpy()
            correct += int((pred == yt[i:i + 512]).sum())
    res[exp] = correct / len(yt)
    res[f"n_{exp}"] = int(len(yt))
    print(f"{exp} Test Accuracy: {res[exp]:.4f}", flush=True)
res["minutes"] = (time.time() - t0) / 60
res["hist"] = hist
json.dump(res, open(TAG + ".json", "w"), indent=1)
print(f"[fan_adv_run] wrote {os.path.join(a.workdir, TAG + '.json')} ({res['minutes']:.1f} min)", flush=True)
