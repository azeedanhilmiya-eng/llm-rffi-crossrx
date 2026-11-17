"""Frozen pre-trained GPT-2 as the RFFI backbone, "One Fits All" recipe (Zhou et al., NeurIPS 2023, GPT4TS: the common
way a pre-trained LLM is reused for signal classification; cf. Zha26r with GPT-2 on WiSig), on the author's ManyRx
protocol (2026-11-17). The "existing LLM-for-RFFI" baseline for the Qwen method.
Model (GPT4TS classification head):
  IQ (2 x 256) -> patches of 16 samples, stride 8 (31 patches, I and Q of a patch concatenated, 32 values)
  -> trainable linear embedding to 768 -> GPT-2 (pre-trained, first --n_layer = 6 blocks, as in GPT4TS), with the
  pre-trained position embeddings; attention and feed-forward weights FROZEN, layer norms and position embeddings
  trainable -> GELU, dropout 0.1, flatten, LayerNorm -> linear head over the 10 transmitters.
Protocol (identical to FAN / IB-RFF; the author's main.py is imported for the loader, split and seeding): first 100
packets per tx per (rx, day), per-packet power normalisation, non-equalized; 24 training / 8 test receivers per fold;
training days 1-2; 70/30 train/validation split of the training receivers' packets (author's prepare_dataset); Adam
lr 1e-3 on the trainable parameters, batch 128, ReduceLROnPlateau (0.1, 5, min 1e-6), early stopping (10) and model
selection on the validation loss, at most 200 epochs. One run trains once and tests CR (days 1-2) and CRD (days 3-4).
Needs transformers (run with R/env/qwen) and the GPT-2 weights in --gpt2 (config.json + model.safetensors).
usage: python gpt2_run.py --code IBRFF_DIR --workdir DIR --gpt2 DIR --round r [--seed 2023] [--epochs 200]
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--workdir", required=True)
p.add_argument("--gpt2", required=True)
p.add_argument("--round", type=int, required=True)
p.add_argument("--seed", type=int, default=2023)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--patch", type=int, default=16)
p.add_argument("--stride", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
a = p.parse_args()
sys.path.insert(0, a.code)
os.makedirs(os.path.join(a.workdir, "weights"), exist_ok=True)
os.chdir(a.workdir)
sys.argv = ["main.py"]
import main  # noqa: E402  (the author's IB-RFF main.py; nothing runs at import except definitions)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
t0 = time.time()
TAG = f"gpt2_l{a.n_layer}_p{a.patch}s{a.stride}_r{a.round}_s{a.seed}"
main.setup_seed(a.seed)
rx_train, rx_test = main.split_receivers(32, 4, a.round)
print(f"[gpt2_run] {TAG} workdir={a.workdir} train={rx_train} test={rx_test}", flush=True)
x_tr, x_va, y_tr, y_va = main.prepare_dataset("ManyRx", rx_train, [1, 2], 10, "non_equalized", True, a.seed)
assert x_tr.shape[1:] == (2, 256), x_tr.shape
print(f"[gpt2_run] train {len(y_tr)} / val {len(y_va)} packets", flush=True)
ten = lambda v, dt: torch.tensor(v, dtype=dt)
train_loader = DataLoader(TensorDataset(ten(x_tr, torch.float32), ten(y_tr, torch.long)), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(ten(x_va, torch.float32), ten(y_va, torch.long)), batch_size=128, shuffle=False)


class GPT4RF(nn.Module):
    def __init__(self, path, n_layer, patch, stride, n_cls, seq=256, chans=2):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.n_tok = (seq - patch) // stride + 1
        self.gpt2 = GPT2Model.from_pretrained(path, n_layer=n_layer)  # loads the first n_layer pre-trained blocks
        assert len(self.gpt2.h) == n_layer
        d = self.gpt2.config.n_embd
        self.embed = nn.Linear(chans * patch, d)
        for n, prm in self.gpt2.named_parameters():
            prm.requires_grad = ("ln" in n) or ("wpe" in n)
        self.act, self.drop = nn.GELU(), nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(d * self.n_tok)
        self.out = nn.Linear(d * self.n_tok, n_cls)

    def forward(self, x):  # x: (B, 2, 256)
        t = x.unfold(-1, self.patch, self.stride)                      # (B, 2, n_tok, patch)
        t = t.permute(0, 2, 1, 3).reshape(x.shape[0], self.n_tok, -1)  # (B, n_tok, 2 * patch)
        h = self.gpt2(inputs_embeds=self.embed(t), use_cache=False).last_hidden_state
        h = self.drop(self.act(h)).reshape(x.shape[0], -1)
        return self.out(self.ln_proj(h))


model = GPT4RF(a.gpt2, a.n_layer, a.patch, a.stride, 10).cuda()
TRAIN = [n for n, prm in model.named_parameters() if prm.requires_grad]
n_tr = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
n_all = sum(prm.numel() for prm in model.parameters())
print(f"[gpt2_run] {model.n_tok} tokens; trainable {n_tr / 1e6:.2f} M of {n_all / 1e6:.2f} M parameters; "
      f"frozen GPT-2 blocks {a.n_layer}", flush=True)
opt = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad], lr=a.lr)
sched = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, min_lr=1e-6)


def run_epoch(loader, train):
    model.train(train)
    n, l_s, c_s = 0, 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            n += len(yb)
            l_s += loss.item() * len(yb)
            c_s += (out.argmax(1) == yb).sum().item()
    return l_s / n, 100.0 * c_s / n


best, wait, hist, ep = float("inf"), 0, [], 0
ck = os.path.join("weights", TAG + ".pt")
for ep in range(1, a.epochs + 1):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    sched.step(va[0])
    hist.append(dict(ep=ep, train=tr, val=va, lr=opt.param_groups[0]["lr"]))
    print(f"ep {ep} train loss {tr[0]:.4f} acc {tr[1]:.2f} | val loss {va[0]:.4f} acc {va[1]:.2f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if va[0] < best:
        best, wait = va[0], 0
        sd_all = model.state_dict()
        torch.save({"trainable": {k: sd_all[k] for k in TRAIN}, "ep": ep}, ck)  # frozen GPT-2 weights not re-saved
    else:
        wait += 1
        if wait >= 10:
            print(f"Early stopping triggered at epoch {ep}.", flush=True)
            break

sd = torch.load(ck, map_location="cuda")
missing, unexpected = model.load_state_dict(sd["trainable"], strict=False)
assert not unexpected and all(k not in TRAIN for k in missing), (missing, unexpected)
model.eval()
res = dict(method="gpt2_frozen (One Fits All / GPT4TS recipe)", round=a.round, seed=a.seed, n_layer=a.n_layer,
           patch=a.patch, stride=a.stride, trainable_params=n_tr, all_params=n_all, best_ep=sd["ep"], epochs_run=ep,
           best_val_loss=best, rx_train=rx_train, rx_test=rx_test)
for exp, days in (("CR", [1, 2]), ("CRD", [3, 4])):
    xt, yt = main.prepare_dataset("ManyRx", rx_test, days, 10, "non_equalized", False, a.seed)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(yt), 512):
            pred = model(torch.tensor(xt[i:i + 512], dtype=torch.float32).cuda()).argmax(1).cpu().numpy()
            correct += int((pred == yt[i:i + 512]).sum())
    res[exp] = correct / len(yt)
    res[f"n_{exp}"] = int(len(yt))
    print(f"{exp} Test Accuracy: {res[exp]:.4f}", flush=True)
res["minutes"] = (time.time() - t0) / 60
res["hist"] = hist
json.dump(res, open(TAG + ".json", "w"), indent=1)
print(f"[gpt2_run] wrote {os.path.join(a.workdir, TAG + '.json')} ({res['minutes']:.1f} min)", flush=True)
