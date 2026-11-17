"""Qwen3.5 backbone + classification head on the authors' ManyRx protocol (2026-11-14; 师兄's suggestion).
Zero target data: trained on the 24 training receivers, days 1-2, exactly like the authors' FAN (IB-RFF main.py
stage 1); nothing from the test receivers is used before testing.

Data and protocol: the authors' own functions, imported from their UNMODIFIED main.py
  main.split_receivers (4 fixed folds, 8 test receivers), main.prepare_dataset (first 100 packets per tx, per-packet
  power normalisation, 30 % validation split with random_state = seed), main.setup_seed.
Training rule: the authors' train_and_evaluate rule (Adam, wd 0, batch 128, NLL on log-softmax, ReduceLROnPlateau
  factor 0.1 / patience 5 / min_lr 1e-6 on validation loss, early stop after 10 epochs without improvement, epoch cap
  200, best checkpoint by validation loss). Deviations, the same for both Qwen arms: lr (--lr, default 1e-4 instead
  of 1e-3, which would wreck pretrained weights), bf16 autocast, gradient clipping at 1.0, linear warm-up over the
  first --warmup steps.
Model: x (B, 2, 256) -> Conv1d patch embedding (2 -> H, kernel 16, stride 8 = FAN's patch geometry, 31 tokens)
  -> LayerNorm (no affine) x learnable gain initialised to the RMS of the checkpoint's token embeddings (puts the
  tokens on the scale the language model was trained on) -> Qwen3.5 text decoder (all layers, causal, inputs_embeds,
  final norm) -> mean over tokens -> dropout -> Linear(H, 10). Returns (features, logits) like FAN.
  --init pretrained : the downloaded checkpoint, all decoder weights trained (full fine-tuning)
  --init random     : SAME architecture and config, the library's own random initialisation (control: is any gain
                      due to language pretraining, or just to a big transformer?)
  The vision tower and the token-embedding table are dropped (inputs_embeds only); nothing else is frozen.
  --lora r          : (pretrained only; for Qwen3.5-4B, which cannot be fully fine-tuned in 24 GB) decoder frozen in
                      bf16, rank-r LoRA (alpha 2r, B = 0 at start) on every Linear inside the decoder layers; the patch
                      embedding, gain and head are trained as above.
Evaluation after training (best checkpoint), the same sets and probes as diag_fan_errors.py so the numbers line up
  with the 4090 FAN results in results/4090/diag_1110/runs/diag_fan_errors/diag_fan_errors.json:
  conditions seen_same (the authors' 30 % validation split), seen_d34, CR, CRD -- overall, by receiver model, per unit;
  nuisance probes (white noise 5/10/20/40 % of signal power, etc.; np.random.RandomState(12345) per probe on the
  whole set, same packet order) on CR and seen_same.
Outputs in WORKDIR, TAG = <model dir name>_<init>[_lora<r>]_r<round>_s<seed>: qwen_TAG.json (args, history, eval),
  probs_TAG.npz, weights/qwen_TAG.pt (full state for full fine-tuning; trainable tensors only with LoRA; the best
  state is kept in CPU RAM during training and written once at the end, 2026-11-15).
Run from anywhere; --code = the IB-RFF directory; diag_fan_errors.py must sit next to this file.
usage: python qwen_run.py --code DIR --qwen MODEL_DIR --workdir DIR --init {pretrained,random} [--round 0]
                          [--seed 2023] [--lr 1e-4] [--lora 0] [--epochs 200] [--max_steps 0 (smoke)]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--qwen", required=True)
p.add_argument("--workdir", required=True)
p.add_argument("--init", required=True, choices=["pretrained", "random"])
p.add_argument("--round", type=int, default=0)
p.add_argument("--seed", type=int, default=2023)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--lora", type=int, default=0, help="0 = full fine-tuning; r > 0 = freeze the decoder (bf16) and "
               "train rank-r LoRA on every Linear inside its layers (for the 4B model, which cannot be fully "
               "fine-tuned in 24 GB)")
p.add_argument("--lora_alpha", type=float, default=0, help="LoRA scale alpha (default 2r)")
p.add_argument("--grad_ckpt", type=int, default=1, help="gradient checkpointing in the decoder (memory: the torch "
               "fallback of Qwen3.5's gated-delta layers stores large fp32 intermediates)")
p.add_argument("--micro", type=int, default=0, help="split each batch of --bs into micro-batches of this size and "
               "accumulate gradients (same effective batch, same loss); 0 = no split")
p.add_argument("--max_steps", type=int, default=0, help="smoke test: stop training after this many steps")
p.add_argument("--eval_bs", type=int, default=256)
p.add_argument("--no_probes", action="store_true")
a = p.parse_args()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, a.code)
os.makedirs(os.path.join(a.workdir, "weights"), exist_ok=True)
sys.argv = ["main.py"]
import main  # noqa: E402  (authors' main.py; only definitions run at import)
import diag_fan_errors as dfe  # noqa: E402  (pure-numpy helpers; its main() is not called)
from utils.load_data import load_single_dataset, preprocessing  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

TX, RX = 10, 32
MNAME = os.path.basename(os.path.normpath(a.qwen))
TAG = f"{MNAME}_{a.init}{f'_lora{a.lora}' if a.lora else ''}_r{a.round}_s{a.seed}"
t0 = time.time()


def log(msg):
    print(f"[qwen_run {TAG} {time.time() - t0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- model
def find_text_model(m):
    """the decoder stack inside a (vision-)language checkpoint: the module that owns .layers and .norm"""
    for name in ("language_model", "model.language_model", "model"):
        cur = m
        try:
            for part in name.split("."):
                cur = getattr(cur, part)
        except AttributeError:
            continue
        if hasattr(cur, "layers") and hasattr(cur, "norm"):
            return cur
    if hasattr(m, "layers") and hasattr(m, "norm"):
        return m
    raise RuntimeError("cannot find the text decoder in " + type(m).__name__)


class LoRALinear(nn.Module):
    """y = base(x) + (x A^T B^T) * alpha / r ; base frozen, B starts at 0 so the start is the pretrained model"""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def add_lora(root, r, alpha):
    n = 0
    for name, mod in list(root.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, LoRALinear(child, r, alpha))
                n += 1
    return n


class QwenRFF(nn.Module):
    def __init__(self, qwen_dir, init, n_cls=TX, patch=16, stride=8, dr=0.1, lora=0, lora_alpha=0):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(qwen_dir)
        dt = torch.bfloat16 if lora else torch.float32
        if init == "pretrained":
            full = AutoModel.from_pretrained(qwen_dir, dtype=dt)
        else:
            full = AutoModel.from_config(cfg)
        lm = find_text_model(full)
        emb = lm.get_input_embeddings()
        H = emb.weight.shape[1]
        rms = float(emb.weight.detach().float().pow(2).mean().sqrt())
        lm.set_input_embeddings(nn.Embedding(1, H))   # never used (inputs_embeds); drops the 248k x H table
        self.lm = lm.to(dt)
        del full
        if a.grad_ckpt:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.n_lora = 0
        if lora:
            self.lm.requires_grad_(False)
            self.n_lora = add_lora(self.lm.layers, lora, lora_alpha or 2 * lora)
        self.cfg_name = type(lm).__name__
        self.H, self.emb_rms = H, rms
        self.patch = nn.Conv1d(2, H, kernel_size=patch, stride=stride)
        self.in_norm = nn.LayerNorm(H, elementwise_affine=False)
        self.gain = nn.Parameter(torch.tensor(rms))
        self.drop = nn.Dropout(dr)
        self.head = nn.Linear(H, n_cls)

    def forward(self, x):
        t = self.patch(x).transpose(1, 2)                     # (B, 31, H)
        t = self.in_norm(t) * self.gain
        h = self.lm(inputs_embeds=t, use_cache=False).last_hidden_state
        f = h.float().mean(1)
        return f, self.head(self.drop(f))


# ---------------------------------------------------------------- data (authors' functions)
main.setup_seed(a.seed)
rx_train, rx_test = main.split_receivers(RX, 4, a.round)
assert (rx_train, rx_test) == dfe.fold_split(a.round)
x_tr, x_va, y_tr, y_va = main.prepare_dataset("ManyRx", rx_train, [1, 2], TX, "non_equalized", True, a.seed)
log(f"train rx {rx_train}\n  test rx {rx_test}\n  train {x_tr.shape} val {x_va.shape}")
DL = torch.utils.data.DataLoader
TD = torch.utils.data.TensorDataset
train_loader = DL(TD(torch.Tensor(x_tr), torch.Tensor(y_tr)), batch_size=a.bs, shuffle=True)
val_loader = DL(TD(torch.Tensor(x_va), torch.Tensor(y_va)), batch_size=a.eval_bs, shuffle=False)

model = QwenRFF(a.qwen, a.init, dr=a.dropout, lora=a.lora, lora_alpha=a.lora_alpha).cuda()
n_par = sum(p.numel() for p in model.parameters())
train_params = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in train_params)
log(f"model {model.cfg_name} H={model.H} emb_rms={model.emb_rms:.4f} params {n_par / 1e6:.1f} M, trainable "
    f"{n_train / 1e6:.2f} M, LoRA modules {model.n_lora}, layers {len(model.lm.layers)}, torch {torch.__version__}")
assert a.init == "pretrained" or not a.lora, "random init + LoRA is not a meaningful arm"

opt = torch.optim.Adam(train_params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, min_lr=1e-6)
wpath = os.path.join(a.workdir, "weights", f"qwen_{TAG}.pt")


def run_eval(loader):
    model.eval()
    tot, cor, n = 0.0, 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for xb, yb in loader:
            xb, yb = xb.cuda(non_blocking=True), yb.long().cuda(non_blocking=True)
            out = F.log_softmax(model(xb)[1].float(), 1)
            tot += F.nll_loss(out, yb, reduction="sum").item()
            cor += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot / n, 100.0 * cor / n


# ---------------------------------------------------------------- training (authors' rule, see docstring)
hist, best, bad, step, stop = [], float("inf"), 0, 0, False
best_state, best_ep = None, 0
for ep in range(1, a.epochs + 1):
    model.train()
    te = time.time()
    tl, tc, tn = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.long().cuda(non_blocking=True)
        if step < a.warmup:
            for g in opt.param_groups:
                g["lr"] = a.lr * (step + 1) / a.warmup
        opt.zero_grad(set_to_none=True)
        bl = 0.0
        for xs, ys in (zip(xb.split(a.micro), yb.split(a.micro)) if a.micro else [(xb, yb)]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, logit = model(xs)
            out = F.log_softmax(logit.float(), 1)
            loss = F.nll_loss(out, ys, reduction="sum") / len(yb)   # = mean over the full batch of --bs
            loss.backward()
            bl += loss.item()
            tc += (out.argmax(1) == ys).sum().item()
        gn = torch.nn.utils.clip_grad_norm_(train_params, a.clip)
        opt.step()
        step += 1
        tl += bl * len(yb)
        tn += len(yb)
        if not math.isfinite(bl):
            raise RuntimeError(f"non-finite loss at step {step}")
        if a.max_steps and step >= a.max_steps:
            stop = True
            break
    vl, va = run_eval(val_loader)
    lr_now = opt.param_groups[0]["lr"]
    if step >= a.warmup:
        sched.step(vl)
    hist.append(dict(epoch=ep, train_loss=tl / tn, train_acc=100 * tc / tn, val_loss=vl, val_acc=va, lr=lr_now,
                     sec=time.time() - te, grad_norm=float(gn), mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
    msg = (f"ep {ep} train {tl / tn:.4f} / {100 * tc / tn:.2f}%  val {vl:.4f} / {va:.2f}%  lr {lr_now:.2e}  "
           f"{time.time() - te:.0f}s  mem {hist[-1]['mem_gb']:.1f} GB")
    if vl < best:
        # 2026-11-15: keep the best state in CPU RAM and write it to disk once after training. Saving a 2 GB
        # checkpoint at every improving epoch put large write bursts on the server's USB disk, which went
        # offline on 11-14 right after such writes (uas write timeouts).
        best, bad, best_ep = vl, 0, ep
        trainable = {n for n, q in model.named_parameters() if q.requires_grad}
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                      if k in trainable or not a.lora}
        msg += "  * best"
    else:
        bad += 1
    log(msg)
    if stop or bad >= 10:
        log(f"stop at epoch {ep} ({'max_steps' if stop else 'early stopping'}), best val loss {best:.4f}")
        break

# ---------------------------------------------------------------- evaluation: same sets / probes as diag_fan_errors
assert best_state is not None, "validation loss never finite"
torch.save({"state": best_state, "args": vars(a), "epoch": best_ep}, wpath)   # written once per run
log(f"saved best (epoch {best_ep}) to {wpath}")
missing, unexpected = model.load_state_dict(best_state, strict=not a.lora)
assert not unexpected and all(not q.requires_grad for n, q in model.named_parameters() if n in set(missing))
model.eval()


def predict_probs(x):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(x), a.eval_bs):
            _, o = model(torch.from_numpy(np.ascontiguousarray(x[i:i + a.eval_bs], dtype=np.float32)).cuda())
            out.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(out)


U = {(r, d): load_single_dataset("ManyRx", r, d, TX, "non_equalized") for r in range(RX) for d in (1, 2, 3, 4)}
tr_units = [(r, d) for r in rx_train for d in (1, 2)]
uid_all = np.concatenate([np.full(len(U[u][1]), u[0] * 10 + u[1]) for u in tr_units])
_, vidx = train_test_split(np.arange(len(uid_all)), test_size=0.3, random_state=a.seed)
xt = np.concatenate([U[u][0] for u in tr_units])
yt = np.concatenate([U[u][1] for u in tr_units])
assert np.array_equal(yt[vidx], y_va) and np.allclose(xt[vidx], x_va), "val split differs from main.prepare_dataset"
sets = {"seen_same": (xt[vidx], yt[vidx], uid_all[vidx])}
for c, rxs, days in (("seen_d34", rx_train, (3, 4)), ("CR", rx_test, (1, 2)), ("CRD", rx_test, (3, 4))):
    us = [(r, d) for r in rxs for d in days]
    sets[c] = (np.concatenate([U[u][0] for u in us]), np.concatenate([U[u][1] for u in us]),
               np.concatenate([np.full(len(U[u][1]), u[0] * 10 + u[1]) for u in us]))
entry = {"cond": {}, "probes": {}, "n_val": int(len(vidx))}
P, Y, UI, CI = [], [], [], []
for ci, c in enumerate(dfe.CONDS):
    x, y, uid = sets[c]
    pr = predict_probs(x)
    pred = pr.argmax(1)
    entry["cond"][c] = dict(overall=dfe.by_model(pred == y, uid), units=dfe.unit_table(pred, y, uid))
    P.append(pr.astype(np.float16)), Y.append(y.astype(np.int8)), UI.append(uid.astype(np.int16))
    CI.append(np.full(len(y), ci, np.int8))
np.savez_compressed(os.path.join(a.workdir, f"probs_{TAG}.npz"), probs=np.concatenate(P), label=np.concatenate(Y),
                    unit=np.concatenate(UI), cond=np.concatenate(CI), cond_names=np.array(dfe.CONDS))
log("eval " + " ".join(f"{c} {entry['cond'][c]['overall']['all']:.2f}" for c in dfe.CONDS))
if not a.no_probes:
    for s in ("CR", "seen_same"):
        x, y, uid = sets[s]
        z = dfe.to_c(x)
        entry["probes"][s] = {}
        for name, fn, renorm in dfe.probe_list():
            rng = np.random.RandomState(12345)
            xp = dfe.to_r(fn(z, rng))
            if renorm:
                xp = preprocessing(xp.copy())
            entry["probes"][s][name] = dfe.by_model(predict_probs(xp).argmax(1) == y, uid)
    log("probes CR " + " ".join(f"{n} {v['all']:.1f}" for n, v in entry["probes"]["CR"].items()
                                 if n.startswith(("clean", "noise"))))
json.dump({"args": vars(a), "model": dict(cls=model.cfg_name, H=model.H, emb_rms=model.emb_rms, params=n_par,
                                          trainable=n_train, n_lora=model.n_lora),
           "history": hist, "res": {f"r{a.round}_s{a.seed}": entry}},
          open(os.path.join(a.workdir, f"qwen_{TAG}.json"), "w"), indent=1)
log("DONE")
