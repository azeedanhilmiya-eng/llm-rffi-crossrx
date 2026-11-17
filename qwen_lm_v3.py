"""Qwen3.5 with its language-model head KEPT: the device identity is generated as text (one digit 0-9), trained by
SFT and then GRPO (2026-11-16; 师兄: "保留结构，然后上 GRPO"; first round without any noise, user 11-16).
v2 (2026-11-17, after an independent review): random fixed eval subsets (base/eval_max), GRPO epoch-0 start from
the SFT model, fp32 reference model (KL exactly 0 at start), SFT numbers reported on the bf16-rounded weights that
GRPO starts from, non-finite loss keeps the best state, results written before the weights.
v3 (2026-11-17, user "换4b试试"; Qwen3.5-4B cannot be fully fine-tuned in 24 GB): --lora r. Everything else is v2.
  The whole Qwen model (decoder, token embeddings, lm_head) is loaded in bf16 and FROZEN; rank-r LoRA (alpha 2r,
  B = 0 at start, fp32 parameters) on every Linear inside the decoder layers, the same LoRALinear as qwen_run.py's
  4B classification-head run; the signal projector (Conv1d patches + gain) is trained in fp32 as in v2.
  Defaults with LoRA: lr 2e-4 (sft, as qwen_run.py's 4B run) / 3e-5 (grpo; LoRA usually needs ~10x the full
  fine-tuning lr, v2 used 3e-6).
  GRPO reference = the SFT model without a second copy of the 4B weights: every LoRA layer keeps a frozen copy of
  its SFT adapters (A_ref, B_ref) and the projector is copied; the reference forward switches to those copies (and
  to eval mode, as v2's separate reference model) and switches back before the policy's backward pass.
  Checkpoints with LoRA hold only the trainable tensors (fp32: LoRA + projector), so no bf16 rounding step is
  needed: the reported SFT numbers and the GRPO starting point are the same weights.
  TAG with LoRA: lm_<input>_<stage>_<model dir name>_lora<r>_r<round>_s<seed> (no clash with the 0.8B files).
Zero target data, the authors' ManyRx protocol, exactly as qwen_run.py (same data functions, same folds, same
30 % validation split, same evaluation sets seen_same / seen_d34 / CR / CRD); no noise anywhere in this round.

Prompt (Qwen chat template, thinking disabled), answer = "<digit><|im_end|>":
  user: "<text> <SIGNAL> <text> ... Which transmitter (0-9) sent it? Answer with one digit."
  assistant: "<k>"
Input variants (--input), the "data organisation" being compared:
  time : 256-point IQ (power-normalised by the authors' code) -> Conv1d patches (2 -> H, kernel 16, stride 8 = FAN's
         geometry) = 31 signal tokens -> LayerNorm (no affine) x gain (init = RMS of the token embeddings), placed at
         <SIGNAL> via inputs_embeds.
  freq : the same projector on the centred FFT of the complex preamble (fftshift(fft(z)) / sqrt(256): same power),
         31 tokens over frequency instead of time.
  text : NO projector, NO new parameters: the first --text_n complex samples, round(x * --text_scale) clipped to
         +-99, written as text "i,q i,q ..." inside the prompt (the LLM reads the numbers as ordinary tokens).
  ref  : in-context references: one packet of every transmitter (drawn from the TRAINING split of the training
         receivers, labelled "Transmitter k:") followed by the query packet, all through the time projector
         (11 x 31 signal tokens). Train: fresh random references per query and epoch; evaluation: references fixed
         by np.random.RandomState(12345). Still zero target data.
Stages:
  sft  : full fine-tuning (decoder, embeddings / tied lm_head, projector). Loss = mean cross-entropy over the two
         answer tokens (digit, <|im_end|>). The authors' training rule as in qwen_run.py: Adam (wd 0), batch 128,
         ReduceLROnPlateau(0.1, patience 5, min 1e-6) on validation loss, early stop after --patience epochs without a
         better validation loss, --epochs cap; lr 1e-4 with --warmup linear warm-up, bf16 autocast, clip 1.0.
         Before training: "base" evaluation of the untrained model (pretrained Qwen, untrained projector) on up to
         --base_n samples of seen_same and CR.
  grpo : from --init_from (an sft checkpoint). The answer is ONE token, so a rollout is one sample from the policy's
         next-token distribution at the answer position (full vocabulary, temperature --temp): --group samples per
         prompt, reward 1 if the sampled token is the correct digit else 0 (a non-digit answer also gets 0),
         advantage = (r - group mean) / (group std + 1e-4), loss = -mean(A * log pi(sample)) + --beta * KL(pi || pi_ref)
         at the answer position (exact KL over the vocabulary; pi_ref = the frozen sft model). One gradient step per
         batch (on-policy, so the PPO ratio is 1 and clipping is inactive). Constant lr --lr (default 3e-6), no
         reasoning tokens (answer only). Validation each epoch: greedy accuracy on seen_same; best state = highest
         greedy accuracy; stop after --patience epochs without improvement or at --epochs.
Accuracy is reported two ways: "acc" = argmax restricted to the ten digit tokens (main number, comparable with the
classification head), "free" = argmax over the whole vocabulary equals the correct digit (true generation).
Outputs in WORKDIR, TAG = lm_<input>_<stage>_r<round>_s<seed>: lm_TAG.json (args, prompt, history, base, eval),
probs_lm_TAG.npz (softmax over the ten digits, float16), weights/lm_TAG.pt (best state, written once at the end).
usage: python qwen_lm_v3.py --code IBRFF_DIR --qwen MODEL_DIR --workdir DIR --input {time,freq,text,ref}
                         [--stage sft|grpo --init_from W.pt] [--round 0] [--lora 16] [--max_steps N --eval_max M (smoke)]
"""
import argparse
import copy
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
p.add_argument("--input", required=True, choices=["time", "freq", "text", "ref"])
p.add_argument("--stage", default="sft", choices=["sft", "grpo"])
p.add_argument("--init_from", default="")
p.add_argument("--round", type=int, default=0)
p.add_argument("--seed", type=int, default=2023)
p.add_argument("--lr", type=float, default=0, help="default 1e-4 (sft) / 3e-6 (grpo)")
p.add_argument("--bs", type=int, default=128)
p.add_argument("--epochs", type=int, default=0, help="epoch cap; default 200 (sft) / 8 (grpo)")
p.add_argument("--patience", type=int, default=0, help="default 10 (sft) / 3 (grpo)")
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--grad_ckpt", type=int, default=1)
p.add_argument("--micro", type=int, default=0)
p.add_argument("--freeze_embed", type=int, default=0, help="1 = freeze the token embeddings / tied lm_head "
               "(memory fallback; default: everything is trained)")
p.add_argument("--group", type=int, default=8)
p.add_argument("--beta", type=float, default=0.04)
p.add_argument("--temp", type=float, default=1.0)
p.add_argument("--text_n", type=int, default=128)
p.add_argument("--text_scale", type=float, default=45.0)
p.add_argument("--base_n", type=int, default=4000)
p.add_argument("--max_steps", type=int, default=0, help="smoke test: stop training after this many steps")
p.add_argument("--eval_max", type=int, default=0, help="smoke test: evaluate at most this many samples per set")
p.add_argument("--eval_bs", type=int, default=128)
p.add_argument("--lora", type=int, default=0, help="0 = full fine-tuning (v2); r > 0 = frozen bf16 Qwen + rank-r "
               "LoRA on every Linear in the decoder layers (for Qwen3.5-4B)")
p.add_argument("--lora_alpha", type=float, default=0, help="LoRA scale alpha (default 2r)")
a = p.parse_args()
if not a.lr:
    if a.lora:
        a.lr = 2e-4 if a.stage == "sft" else 3e-5
    else:
        a.lr = 1e-4 if a.stage == "sft" else 3e-6
if not a.epochs:
    a.epochs = 200 if a.stage == "sft" else 8
if not a.patience:
    a.patience = 10 if a.stage == "sft" else 3
assert a.stage == "sft" or a.init_from, "grpo needs --init_from (an sft checkpoint)"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, a.code)
os.makedirs(os.path.join(a.workdir, "weights"), exist_ok=True)
sys.argv = ["main.py"]
import main  # noqa: E402  (authors' main.py; only definitions run at import)
import diag_fan_errors as dfe  # noqa: E402
from utils.load_data import load_single_dataset  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TX, RX = 10, 32
MNAME = os.path.basename(os.path.normpath(a.qwen))
TAG = f"lm_{a.input}_{a.stage}" + (f"_{MNAME}_lora{a.lora}" if a.lora else "") + f"_r{a.round}_s{a.seed}"
t0 = time.time()


def log(msg):
    print(f"[qwen_lm {TAG} {time.time() - t0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- data (authors' functions, as in qwen_run.py)
main.setup_seed(a.seed)
rx_train, rx_test = main.split_receivers(RX, 4, a.round)
assert (rx_train, rx_test) == dfe.fold_split(a.round)
x_tr, x_va, y_tr, y_va = main.prepare_dataset("ManyRx", rx_train, [1, 2], TX, "non_equalized", True, a.seed)
y_tr, y_va = y_tr.astype(np.int64), y_va.astype(np.int64)
log(f"train rx {rx_train}\n  test rx {rx_test}\n  train {x_tr.shape} val {x_va.shape}")

U = {(r, d): load_single_dataset("ManyRx", r, d, TX, "non_equalized") for r in range(RX) for d in (1, 2, 3, 4)}
tr_units = [(r, d) for r in rx_train for d in (1, 2)]
uid_all = np.concatenate([np.full(len(U[u][1]), u[0] * 10 + u[1]) for u in tr_units])
_, vidx = train_test_split(np.arange(len(uid_all)), test_size=0.3, random_state=a.seed)
xt = np.concatenate([U[u][0] for u in tr_units])
yt = np.concatenate([U[u][1] for u in tr_units])
assert np.array_equal(yt[vidx], y_va) and np.allclose(xt[vidx], x_va), "val split differs from main.prepare_dataset"
SETS = {"seen_same": (xt[vidx], yt[vidx].astype(np.int64), uid_all[vidx])}
for c, rxs, days in (("seen_d34", rx_train, (3, 4)), ("CR", rx_test, (1, 2)), ("CRD", rx_test, (3, 4))):
    us = [(r, d) for r in rxs for d in days]
    SETS[c] = (np.concatenate([U[u][0] for u in us]), np.concatenate([U[u][1] for u in us]).astype(np.int64),
               np.concatenate([np.full(len(U[u][1]), u[0] * 10 + u[1]) for u in us]))
del U, xt, yt


def to_freq(x):
    z = x[:, 0].astype(np.float64) + 1j * x[:, 1]
    Z = np.fft.fftshift(np.fft.fft(z, axis=1), axes=1) / np.sqrt(z.shape[1])
    return np.stack([Z.real, Z.imag], 1).astype(np.float32)


def to_text(x):
    q = np.clip(np.rint(x[:, :, :a.text_n] * a.text_scale), -99, 99).astype(np.int64)
    return [" ".join(f"{i},{j}" for i, j in zip(r[0], r[1])) for r in q]


# ---------------------------------------------------------------- prompt and tokens
tok = AutoTokenizer.from_pretrained(a.qwen)
SIG = "<SIGNAL>"
ASK = "Which transmitter (0-9) sent it? Answer with one digit."
if a.input in ("time", "freq"):
    content = f"Here is the preamble of a received WiFi packet: {SIG}\n{ASK}"
elif a.input == "text":
    content = (f"Here are the first {a.text_n} complex samples (I,Q as scaled integers) of a received WiFi packet "
               f"preamble: {SIG}\n{ASK}")
else:
    content = ("Reference packets from the known transmitters:\n"
               + "".join(f"Transmitter {k}: {SIG}\n" for k in range(TX))
               + f"Query packet: {SIG}\nWhich transmitter (0-9) sent the query packet? Answer with one digit.")
try:
    prompt = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
except Exception as e:   # noqa: BLE001
    log(f"chat template failed ({e!r}); using the plain ChatML form")
    prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
log("prompt: " + repr(prompt))
SEGS = [tok(s, add_special_tokens=False)["input_ids"] for s in prompt.split(SIG)]
N_SLOT = len(SEGS) - 1
DIG = []
for k in range(TX):
    ids = tok(str(k), add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"digit {k} is not one token: {ids}"
    DIG.append(ids[0])
EOS = tok.convert_tokens_to_ids("<|im_end|>")
assert isinstance(EOS, int) and EOS >= 0
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
log(f"segments {[len(s) for s in SEGS]} slots {N_SLOT} digits {DIG} eos {EOS}")


class Data:
    """one evaluation/training set in model-ready form"""

    def __init__(self, x, y):
        self.y = y
        if a.input == "text":
            body = tok(to_text(x), add_special_tokens=False)["input_ids"]
            seqs = [SEGS[0] + b + SEGS[1] for b in body]
            self.plen = np.array([len(s) for s in seqs])
            L = int(self.plen.max()) + 2
            self.ids = np.full((len(seqs), L), PAD, np.int64)
            for i, s in enumerate(seqs):
                self.ids[i, :len(s)] = s
            self.x = None
        else:
            self.x = to_freq(x) if a.input == "freq" else x.astype(np.float32)

    def __len__(self):
        return len(self.y)


# ---------------------------------------------------------------- model
REF = [False]   # True while the GRPO reference forward runs (LoRA layers then use their frozen SFT copies)


class LoRALinear(nn.Module):
    """y = base(x) + (x A^T B^T) * alpha / r ; base frozen, B starts at 0 so the start is the pretrained model
    (same as qwen_run.py). A_ref / B_ref: frozen copies of the SFT adapters, used while REF[0] is True."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.A_ref = self.B_ref = None   # plain attributes: not in state_dict

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        A, B = (self.A_ref, self.B_ref) if REF[0] else (self.A, self.B)
        return self.base(x) + F.linear(F.linear(x, A), B) * self.scale


def add_lora(root, r, alpha):
    n = 0
    for name, mod in list(root.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, LoRALinear(child, r, alpha))
                n += 1
    return n


class LMRFF(nn.Module):
    def __init__(self, qwen_dir):
        super().__init__()
        try:
            from transformers import Qwen3_5ForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
        full = Cls.from_pretrained(qwen_dir, dtype=torch.bfloat16 if a.lora else torch.float32)
        lm = full.model.language_model
        self.head = full.lm_head
        emb = lm.get_input_embeddings()
        self.tied = self.head.weight.data_ptr() == emb.weight.data_ptr()
        self.H = emb.weight.shape[1]
        self.emb_rms = float(emb.weight.detach().float().pow(2).mean().sqrt())
        self.lm = lm
        del full
        if a.grad_ckpt:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.n_lora = 0
        if a.lora:   # frozen Qwen (decoder, embeddings, lm_head); adapters created after freezing stay trainable
            self.lm.requires_grad_(False)
            self.head.requires_grad_(False)
            self.n_lora = add_lora(self.lm.layers, a.lora, a.lora_alpha or 2 * a.lora)
        if a.input != "text":
            self.patch = nn.Conv1d(2, self.H, kernel_size=16, stride=8)
            self.in_norm = nn.LayerNorm(self.H, elementwise_affine=False)
            self.gain = nn.Parameter(torch.tensor(self.emb_rms))

    def sig(self, x):
        return (self.in_norm(self.patch(x).transpose(1, 2).float()) * self.gain).float()

    def answer_logits(self, emb, pos):
        """emb (B, L, H) with the answer tokens appended; pos (B,) index of the last prompt token.
        Returns logits (float32) predicting the digit (at pos) and <|im_end|> (at pos + 1)."""
        h = self.lm(inputs_embeds=emb, use_cache=False).last_hidden_state
        ar = torch.arange(len(pos), device=h.device)
        return self.head(h[ar, pos]).float(), self.head(h[ar, pos + 1]).float()


SEG_T = None
REF_POOL = None   # (x tensor on GPU, list of index tensors per class) for --input ref


def gather(d, idx, ref_rng=None):
    """model-independent raw inputs for samples idx of Data d (reference packets are drawn HERE, once, so policy
    and reference model in GRPO see exactly the same prompt)"""
    y = torch.from_numpy(d.y[idx]).cuda()
    ans = torch.stack([torch.tensor(DIG, device="cuda")[y], torch.full_like(y, EOS)], 1)
    B = len(idx)
    if a.input == "text":
        ids = torch.from_numpy(d.ids[idx]).cuda()
        plen = torch.from_numpy(d.plen[idx]).cuda()
        ar = torch.arange(B, device="cuda")
        ids[ar, plen] = ans[:, 0]
        ids[ar, plen + 1] = ans[:, 1]
        return dict(ids=ids, pos=plen - 1, y=y)
    xs = [torch.from_numpy(d.x[idx]).cuda()]
    if a.input == "ref":
        xr, cls_idx = REF_POOL
        refs = []
        for k in range(TX):
            n = len(cls_idx[k])
            j = torch.from_numpy(ref_rng.randint(0, n, B)).cuda() if ref_rng is not None else \
                torch.randint(0, n, (B,), device="cuda")
            refs.append(xr[cls_idx[k][j]])
        xs = refs + xs
    plen = sum(len(s) for s in SEGS) + 31 * len(xs)
    return dict(xs=xs, ans=ans, pos=torch.full((B,), plen - 1, device="cuda", dtype=torch.long), y=y)


def embed(model, g):
    """inputs_embeds (answer = true digit appended, teacher forcing; causal, so the digit logits never see it)"""
    E = model.lm.get_input_embeddings()
    if "ids" in g:
        return E(g["ids"])
    B = len(g["y"])
    parts = []
    for i, s in enumerate(SEG_T):
        parts.append(E(s).unsqueeze(0).expand(B, -1, -1))
        if i < len(g["xs"]):
            parts.append(model.sig(g["xs"][i]).to(parts[0].dtype))
    parts.append(E(g["ans"]))
    return torch.cat(parts, 1)


def build(model, d, idx, ref_rng=None):
    g = gather(d, idx, ref_rng)
    return embed(model, g), g["pos"], g["y"]


def sft_loss(ld, le, y):
    return 0.5 * (F.cross_entropy(ld, torch.tensor(DIG, device="cuda")[y]) +
                  F.cross_entropy(le, torch.full_like(y, EOS)))


@torch.no_grad()
def evaluate(model, d, n_max=0, fixed_refs=True):
    """returns loss, restricted acc, free acc, digit probs (n, 10), restricted pred, free-correct mask, sample index.
    A subset (n_max < len) is a fixed random draw: the sets are ordered by (receiver, day, tx), so the first n
    samples would cover only a few receivers/transmitters."""
    model.eval()
    n = len(d) if not n_max else min(n_max, len(d))
    sel = np.arange(n) if n == len(d) else np.sort(np.random.RandomState(0).permutation(len(d))[:n])
    rng = np.random.RandomState(12345) if (a.input == "ref" and fixed_refs) else None
    tot, P, OKF = 0.0, [], []
    dig = torch.tensor(DIG, device="cuda")
    for i in range(0, n, a.eval_bs):
        idx = sel[i:i + a.eval_bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb, pos, y = build(model, d, idx, rng)
            ld, le = model.answer_logits(emb, pos)
        tot += sft_loss(ld, le, y).item() * len(idx)
        P.append(torch.softmax(ld[:, dig], 1).cpu().numpy())
        OKF.append((ld.argmax(1) == dig[y]).cpu().numpy())
    P, okf = np.concatenate(P), np.concatenate(OKF)
    pred = P.argmax(1)
    return tot / n, 100 * float((pred == d.y[sel]).mean()), 100 * float(okf.mean()), P, pred, okf, sel


# ---------------------------------------------------------------- build everything
D_tr = Data(x_tr, y_tr)
D = {c: Data(SETS[c][0], SETS[c][1]) for c in dfe.CONDS}
if a.input == "text":
    log(f"text lengths: train max {D_tr.ids.shape[1]} median {int(np.median(D_tr.plen))}")
log(f"data ready ({time.time() - t0:.0f}s)")

model = LMRFF(a.qwen).cuda()
SEG_T = [torch.tensor(s, dtype=torch.long, device="cuda") for s in SEGS]
if a.input == "ref":
    xr = torch.from_numpy(x_tr.astype(np.float32)).cuda()
    yr = torch.from_numpy(y_tr).cuda()
    REF_POOL = (xr, [torch.nonzero(yr == k).squeeze(1) for k in range(TX)])
SKIP = {"head.weight"} if model.tied else set()   # tied to lm.embed_tokens.weight: stored once
if a.lora:   # checkpoints hold only the trainable tensors (LoRA + projector); the frozen Qwen comes from --qwen
    TRAINABLE = {n for n, q in model.named_parameters() if q.requires_grad}
    SKIP = set(model.state_dict().keys()) - TRAINABLE


def load_state(sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= SKIP, (missing, unexpected)
    if a.lora:
        assert set(sd.keys()) == TRAINABLE, "LoRA checkpoint does not match this model's trainable tensors"


if a.init_from:
    ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
    load_state(ck["state"])
    log(f"loaded {a.init_from} (epoch {ck.get('epoch')})")
if a.freeze_embed:
    model.lm.get_input_embeddings().requires_grad_(False)
    model.head.requires_grad_(False)
train_params = [q for q in model.parameters() if q.requires_grad]
n_par = sum(q.numel() for q in model.parameters())
log(f"model {MNAME} H={model.H} emb_rms={model.emb_rms:.4f} tied lm_head {model.tied} params {n_par / 1e6:.1f} M "
    f"(trainable {sum(q.numel() for q in train_params) / 1e6:.2f} M, LoRA modules {model.n_lora}) "
    f"layers {len(model.lm.layers)} torch {torch.__version__}")


class RefView:
    """GRPO reference for LoRA runs: shares the frozen Qwen with the policy; the SFT adapters (copied into every
    LoRALinear's A_ref / B_ref) and a copy of the SFT projector stand in for a second model. eval mode during the
    reference forward, as v2's separate reference model; REF and the mode are restored before the policy's backward
    (the non-reentrant checkpoint recomputation of the policy then uses the policy adapters)."""

    def __init__(self, m):
        self.m = m
        for mod in m.lm.modules():
            if isinstance(mod, LoRALinear):
                mod.A_ref, mod.B_ref = mod.A.detach().clone(), mod.B.detach().clone()
        if a.input != "text":
            self.patch = copy.deepcopy(m.patch).requires_grad_(False)
            self.in_norm = m.in_norm   # no parameters
            self.gain = m.gain.detach().clone()

    @property
    def lm(self):
        return self.m.lm

    def sig(self, x):
        return (self.in_norm(self.patch(x).transpose(1, 2).float()) * self.gain).float()

    def answer_logits(self, emb, pos):
        was = self.m.training
        self.m.eval()
        REF[0] = True
        try:
            return self.m.answer_logits(emb, pos)
        finally:
            REF[0] = False
            self.m.train(was)


ref_model = None
if a.stage == "grpo":
    if a.lora:
        ref_model = RefView(model)
    else:
        ref_model = copy.deepcopy(model).eval()   # fp32 like the policy: KL is exactly 0 at the start
        ref_model.requires_grad_(False)

base = {}
if a.stage == "sft" and a.base_n:
    for c in ("seen_same", "CR"):
        nb = min(a.base_n, a.eval_max or a.base_n)
        bl, bacc, bfree, *_ = evaluate(model, D[c], nb)
        base[c] = dict(n=nb, loss=bl, acc=bacc, free=bfree)
        log(f"base (untrained) {c}: n {nb} acc {bacc:.2f} free {bfree:.2f} loss {bl:.3f}")

opt = torch.optim.Adam(train_params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, min_lr=1e-6) \
    if a.stage == "sft" else None
wpath = os.path.join(a.workdir, "weights", f"{TAG}.pt")
dig_t = torch.tensor(DIG, device="cuda")

# ---------------------------------------------------------------- training
hist, step, stop, bad, best_ep, best_state, nonfinite = [], 0, False, 0, 0, None, False
best = float("inf") if a.stage == "sft" else -1.0
ntr = len(D_tr)


def snapshot():
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items() if k not in SKIP}


if a.stage == "grpo":   # epoch 0 = the loaded SFT model: same-weights starting point, and selectable as the best
    vl0, va0, vf0, *_ = evaluate(model, D["seen_same"], a.eval_max)
    hist.append(dict(epoch=0, val_loss=vl0, val_acc=va0, val_free=vf0, steps=0))
    best, best_ep, best_state = vf0, 0, snapshot()
    log(f"ep 0 (SFT start) val {vl0:.4f} / {va0:.2f}% (free {vf0:.2f})  * best")
for ep in range(1, a.epochs + 1):
    model.train()
    te = time.time()
    perm = np.random.permutation(ntr)
    agg = dict(loss=0.0, acc=0, n=0, reward=0.0, active=0.0, kl=0.0, nb=0)
    for i in range(0, ntr, a.bs):
        idx = perm[i:i + a.bs]
        if a.stage == "sft" and step < a.warmup:
            for g in opt.param_groups:
                g["lr"] = a.lr * (step + 1) / a.warmup
        opt.zero_grad(set_to_none=True)
        chunks = np.array_split(idx, max(1, math.ceil(len(idx) / a.micro))) if a.micro else [idx]
        for ch in chunks:
            w = len(ch) / len(idx)
            g = gather(D_tr, ch)
            pos, y = g["pos"], g["y"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld, le = model.answer_logits(embed(model, g), pos)
            if a.stage == "sft":
                loss = sft_loss(ld, le, y)
                agg["acc"] += (ld[:, dig_t].argmax(1) == y).sum().item()
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ld_ref, _ = ref_model.answer_logits(embed(ref_model, g), pos)
                lp = F.log_softmax(ld / a.temp, 1)
                with torch.no_grad():
                    samp = torch.multinomial(lp.exp(), a.group, replacement=True)
                    r = (samp == dig_t[y][:, None]).float()
                    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
                lpr = F.log_softmax(ld_ref.float() / a.temp, 1)
                kl = (lp.exp() * (lp - lpr)).sum(1).mean()
                loss = -(adv * lp.gather(1, samp)).mean() + a.beta * kl
                agg["reward"] += r.mean().item() * len(ch)
                agg["active"] += (r.std(1) > 0).float().sum().item()
                agg["kl"] += kl.item() * len(ch)
                agg["acc"] += (ld[:, dig_t].argmax(1) == y).sum().item()
            (loss * w).backward()
            agg["loss"] += loss.item() * len(ch)
            agg["n"] += len(ch)
        if not math.isfinite(agg["loss"]):   # keep the best state found so far instead of losing the run
            log(f"non-finite loss at step {step + 1}: stopping, best state so far is kept")
            nonfinite = stop = True
            break
        gn = torch.nn.utils.clip_grad_norm_(train_params, a.clip)
        opt.step()
        step += 1
        if step <= 3 or step % 100 == 0:
            log(f"  step {step} loss {agg['loss'] / agg['n']:.4f} {(time.time() - te) / (i // a.bs + 1):.2f} s/step "
                f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB"
                + (f" kl {agg['kl'] / agg['n']:.2e}" if a.stage == "grpo" else ""))
        if a.max_steps and step >= a.max_steps:
            stop = True
            break
    if nonfinite:
        break
    vl, va, vf, *_ = evaluate(model, D["seen_same"], a.eval_max)
    lr_now = opt.param_groups[0]["lr"]
    if sched is not None and step >= a.warmup:
        sched.step(vl)
    h = dict(epoch=ep, train_loss=agg["loss"] / agg["n"], train_acc=100 * agg["acc"] / agg["n"], val_loss=vl,
             val_acc=va, val_free=vf, lr=lr_now, sec=time.time() - te, steps=step, grad_norm=float(gn),
             mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    if a.stage == "grpo":
        h.update(reward=agg["reward"] / agg["n"], active_groups=100 * agg["active"] / agg["n"], kl=agg["kl"] / agg["n"])
    hist.append(h)
    msg = (f"ep {ep} train {h['train_loss']:.4f} / {h['train_acc']:.2f}%  val {vl:.4f} / {va:.2f}% (free {vf:.2f})  "
           f"lr {lr_now:.2e}  {h['sec']:.0f}s  mem {h['mem_gb']:.1f} GB")
    if a.stage == "grpo":
        msg += f"  reward {h['reward']:.4f} active {h['active_groups']:.1f}% kl {h['kl']:.5f}"
    better = vl < best if a.stage == "sft" else vf > best
    if better or best_state is None:
        best, bad, best_ep = (vl if a.stage == "sft" else vf), 0, ep
        best_state = snapshot()
        msg += "  * best"
    else:
        bad += 1
    log(msg)
    if stop or bad >= a.patience:
        log(f"stop at epoch {ep} ({'max_steps' if stop else 'early stopping'}), best {best:.4f} at epoch {best_ep}")
        break
else:
    log(f"stop at the epoch cap {a.epochs}, best {best:.4f} at epoch {best_ep}")

if best_state is None:   # non-finite loss before the first validation: nothing valid to evaluate
    log("no valid state (non-finite loss in epoch 1); writing history only")
    json.dump({"args": vars(a), "base": base, "history": hist, "error": "non-finite loss in epoch 1"},
              open(os.path.join(a.workdir, f"{TAG}.json"), "w"), indent=1)
    sys.exit(1)
# the best state is rounded to bf16 BEFORE the final evaluation, so the reported numbers, the saved checkpoint
# (bf16, about 1.7 GB: smaller write bursts on the USB disk) and the GRPO starting point are the same model
if not a.lora:   # LoRA checkpoints stay fp32 (trainable tensors only, ~0.1 GB)
    best_state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in best_state.items()}
load_state(best_state)

# ---------------------------------------------------------------- evaluation (clean only in this round)
entry = {"cond": {}}
P, Y, UI, CI = [], [], [], []
for ci, c in enumerate(dfe.CONDS):
    d = D[c]
    loss, acc, free, pr, pred, okf, sel = evaluate(model, d, a.eval_max)
    n = len(pred)
    uid, yy = SETS[c][2][sel], d.y[sel]
    entry["cond"][c] = dict(n=n, loss=loss, acc=acc, free=free, overall=dfe.by_model(pred == yy, uid),
                            units=dfe.unit_table(pred, yy, uid))
    P.append(pr.astype(np.float16)), Y.append(yy.astype(np.int8)), UI.append(uid.astype(np.int16))
    CI.append(np.full(n, ci, np.int8))
np.savez_compressed(os.path.join(a.workdir, f"probs_{TAG}.npz"), probs=np.concatenate(P), label=np.concatenate(Y),
                    unit=np.concatenate(UI), cond=np.concatenate(CI), cond_names=np.array(dfe.CONDS))
log("eval " + " ".join(f"{c} {entry['cond'][c]['acc']:.2f}(free {entry['cond'][c]['free']:.2f})" for c in dfe.CONDS))
json.dump({"args": vars(a), "prompt": prompt, "segments": [len(s) for s in SEGS], "digits": DIG, "eos": EOS,
           "model": dict(name=MNAME, H=model.H, emb_rms=model.emb_rms, tied=model.tied, params=n_par,
                         trainable=sum(q.numel() for q in train_params), n_lora=model.n_lora),
           "base": base, "history": hist, "best_epoch": best_ep, "res": {f"r{a.round}_s{a.seed}": entry}},
          open(os.path.join(a.workdir, f"{TAG}.json"), "w"), indent=1)
try:   # weights last: results are already on disk if this large write fails (the USB disk has dropped twice)
    torch.save({"state": best_state, "args": vars(a), "epoch": best_ep}, wpath)
    log(f"saved best (epoch {best_ep}) to {wpath} (bf16)")
except Exception as e:
    log(f"WARNING: saving weights failed: {e!r}")
log("DONE")
