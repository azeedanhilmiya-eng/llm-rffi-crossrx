"""Where does plain FAN (IB-RFF stage 1, alpha 0, authors' ManyRx non-equalized protocol) lose its last ~10 %?
(2026-11-10) Evaluation only, no training, no change to the authors' code. For every (fold, seed) checkpoint:
  1. four conditions, kept per (receiver, day) unit: seen receivers / same days (the authors' exact 30 % validation
     split: main.prepare_dataset concatenates receivers (outer) x days 1-2 (inner), then
     train_test_split(test_size=0.3, random_state=seed); the same split of an index array keeps each sample's unit),
     seen receivers / days 3-4, CR (test receivers, days 1-2), CRD (test receivers, days 3-4).
     n, accuracy and the 10x10 confusion (true x predicted) per unit; softmax probabilities (float16), labels,
     unit ids and condition ids -> OUT/probs_r<r>_s<seed>.npz.
  2. nuisance probes on CR and on seen/same-days: accuracy overall and per receiver model.
  3. BatchNorm re-estimation per test unit (CR, CRD; unlabeled, no gradients) if the model has BatchNorm layers.
Once, model-independent, per unit -> OUT/units.json: median L-STF coherence (diag_rx_manyrx.py: lag 20 over the
first 180 samples), per-transmitter median fine CFO and signal count, blind LO offset (diag_fan_shift.py functions):
all 128 units aligned to one common reference, and per fold the test units aligned to that fold's training units
(days 1-2) as in align_run.py.
Run from the IB-RFF code directory (backbones/ and utils/ import as in main.py), PYTHONPATH = the shims.
usage: python diag_fan_errors.py WEIGHTS_DIR OUT_DIR [--seeds 2023 2024 2025] [--rounds 0 1 2 3]
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

FS, STEP = 25e6, 0.05
GRID = np.arange(-60, 60, STEP)
TX, RX, NROUND = 10, 32, 4
DAYS = (1, 2, 3, 4)
RX_NAMES = ["1-1", "1-19", "1-20", "2-1", "2-19", "3-19", "7-7", "7-14", "8-7", "8-8", "8-14", "13-7", "13-14", "14-7",
            "18-2", "18-19", "19-1", "19-2", "19-19", "19-20", "20-1", "20-19", "20-20", "23-1", "23-3", "23-5",
            "23-6", "23-7", "24-5", "24-6", "24-13", "24-16"]
# receiver hardware model in the authors' index order, checked against ~/rffi_data/wisig/manyrx_cache.npz
# (rx_list / rx_model) on 2026-11-10
RX_MODEL = ["N210", "N210", "N210", "N210", "N210", "B210", "N210", "N210", "B210", "N210", "B210", "B210", "B210",
            "N210", "B210", "B210", "N210", "N210", "N210", "N210", "N210", "N210", "N210", "X310", "X310", "X310",
            "X310", "X310", "X310", "X310", "X310", "X310"]
MODELS = ("N210", "B210", "X310")
CONDS = ("seen_same", "seen_d34", "CR", "CRD")


def fold_split(rd):   # main.split_receivers for 32 receivers / 4 rounds: contiguous block of 8 = test
    test = list(range(rd * 8, rd * 8 + 8))
    return [r for r in range(RX) if r not in test], test


# ---------------------------------------------------------------- signal statistics (copied from diag_fan_shift.py)
def to_c(x):
    return x[:, 0, :].astype(np.float64) + 1j * x[:, 1, :]


def to_r(z):
    return np.stack([z.real, z.imag], 1).astype(np.float32)


def cfo_fine(z, P=100):
    s = z[:, :160]
    c = np.angle((s[:, 20:] * np.conj(s[:, :-20])).sum(1)) / (2 * np.pi * 20) * FS / 1e3
    ph = np.angle((z[:, P:200] * np.conj(z[:, :200 - P])).sum(1))
    k = np.round((2 * np.pi * P * c * 1e3 / FS - ph) / (2 * np.pi))
    return (ph + 2 * np.pi * k) / (2 * np.pi * P) * FS / 1e3


def density(f, bw=0.3):
    f = f[np.abs(f) < 60]
    h = np.histogram(f, bins=np.append(GRID, GRID[-1] + STEP))[0].astype(float)
    k = np.exp(-0.5 * (np.arange(-60, 61) * STEP / bw) ** 2)
    return np.convolve(h, k / k.sum(), mode="same") / max(len(f), 1)


def align_to(f, ref, span=5.0):
    d, best = density(f), None
    for o in np.arange(-span, span + 1e-3, STEP):
        sc = float((np.roll(d, -int(round(o / STEP))) * ref).sum())
        best = (sc, o) if best is None or sc > best[0] else best
    return float(best[1])


def group(fl, iters=3):
    offs = np.zeros(len(fl))
    for _ in range(iters):
        ref = np.mean([density(f - o) for f, o in zip(fl, offs)], 0)
        offs = np.array([align_to(f, ref) for f in fl])
    return offs, np.mean([density(f - o) for f, o in zip(fl, offs)], 0)


def stf_coherence(z):   # diag_rx_manyrx.py: |sum x[n+20] x*[n]| / sum |x|^2 over the first 180 samples
    stf = z[:, :180]
    return np.abs((stf[:, 20:] * np.conj(stf[:, :-20])).sum(1)) / (np.abs(stf) ** 2).sum(1)


# ---------------------------------------------------------------- nuisance perturbations (complex in, complex out)
def p_phase(z, th, rng):
    if th == "random":
        return z * np.exp(1j * rng.uniform(0, 2 * np.pi, len(z)))[:, None]
    return z * np.exp(1j * th)


def p_tshift(z, tau):   # circular fractional delay by tau samples (FFT phase ramp); integer tau == np.roll
    k = np.fft.fftfreq(z.shape[1]) * z.shape[1]
    return np.fft.ifft(np.fft.fft(z, axis=1) * np.exp(-2j * np.pi * k * tau / z.shape[1])[None], axis=1)


def p_fshift(z, khz):
    return z * np.exp(2j * np.pi * khz * 1e3 * np.arange(z.shape[1]) / FS)[None]


def p_noise(z, p, rng):   # complex white noise of power p x the per-sample signal power (renormalized afterwards)
    ps = (np.abs(z) ** 2).mean(1, keepdims=True)
    return z + np.sqrt(p * ps / 2) * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))


def p_dc(z, c):   # complex DC of magnitude c x per-sample RMS, at 45 degrees
    rms = np.sqrt((np.abs(z) ** 2).mean(1, keepdims=True))
    return z + c * rms * np.exp(1j * np.pi / 4)


def p_iq(z, g_db, ph_deg):   # receiver IQ imbalance: y = mu z + nu conj(z), gain g (dB), phase phi (deg)
    g, ph = 10 ** (g_db / 20), np.deg2rad(ph_deg)
    mu, nu = (1 + g * np.exp(-1j * ph)) / 2, (1 - g * np.exp(1j * ph)) / 2
    return mu * z + nu * np.conj(z)


def p_tilt(z, db):   # linear-in-dB spectral tilt, total `db` from -fs/2 to +fs/2 (0 dB at DC)
    f = np.fft.fftfreq(z.shape[1])          # -0.5 .. 0.5 cycles/sample
    return np.fft.ifft(np.fft.fft(z, axis=1) * (10 ** (db * f / 20))[None], axis=1)


def probe_list():
    P = [("clean", lambda z, r: z, False)]
    for th, nm in ((np.pi / 4, "pi/4"), (np.pi / 2, "pi/2"), (np.pi, "pi"), ("random", "random")):
        P.append((f"phase_{nm}", lambda z, r, th=th: p_phase(z, th, r), False))
    for t in (-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2):
        P.append((f"tshift_{t:+g}", lambda z, r, t=t: p_tshift(z, t), False))
    for f in (-2, -1, 1, 2):
        P.append((f"fshift_{f:+g}kHz", lambda z, r, f=f: p_fshift(z, f), False))
    for p in (0.05, 0.1, 0.2, 0.4):
        P.append((f"noise_{p:g}", lambda z, r, p=p: p_noise(z, p, r), True))
    for c in (0.05, 0.1):
        P.append((f"dc_{c:g}", lambda z, r, c=c: p_dc(z, c), False))
    for g, ph in ((0.5, 2), (1, 5)):
        P.append((f"iq_{g:g}dB_{ph:g}deg", lambda z, r, g=g, ph=ph: p_iq(z, g, ph), False))
    for d in (-3, -1, 1, 3):
        P.append((f"tilt_{d:+g}dB", lambda z, r, d=d: p_tilt(z, d), False))
    return P


# ---------------------------------------------------------------- per-unit bookkeeping
def unit_table(pred, y, uid):
    """per unit id: n, acc (%), confusion (10x10 list, rows = true)"""
    res = {}
    for u in np.unique(uid):
        m = uid == u
        cm = np.zeros((TX, TX), int)
        np.add.at(cm, (y[m], pred[m]), 1)
        res[int(u)] = dict(n=int(m.sum()), acc=float(100 * (pred[m] == y[m]).mean()), conf=cm.tolist())
    return res


def by_model(ok, uid):
    rx = uid // 10
    out = {"all": float(100 * ok.mean())}
    for mdl in MODELS:
        m = np.isin(rx, [i for i in range(RX) if RX_MODEL[i] == mdl])
        if m.any():
            out[mdl] = float(100 * ok[m].mean())
    return out


def ranks(v):
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    for x in np.unique(v):   # average ties
        m = v == x
        r[m] = r[m].mean()
    return r


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return dict(n=len(a), pearson=None, spearman=None)
    return dict(n=len(a), pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(np.corrcoef(ranks(a), ranks(b))[0, 1]))


# ---------------------------------------------------------------- summary (pure numpy, testable without torch)
def summarize(res, units):
    S = {"cond_by_fold": {}, "cond_by_model": {}, "test_rx": {}, "top_confusions": {}, "coherence_vs_acc": {},
         "probes_mean_drop": {}, "bn": {}}
    folds = sorted({int(k.split("_")[0][1:]) for k in res})
    for rd in folds:
        keys = [k for k in res if k.startswith(f"r{rd}_")]
        S["cond_by_fold"][rd] = {}
        for c in CONDS:
            accs = [res[k]["cond"][c]["overall"]["all"] for k in keys]
            S["cond_by_fold"][rd][c] = dict(mean=float(np.mean(accs)), per_seed=accs)
        # per test receiver (mean over seeds, CR and CRD separately; days pooled by samples)
        _, test = fold_split(rd)
        for r in test:
            row = {"name": RX_NAMES[r], "model": RX_MODEL[r], "fold": rd}
            for c, days in (("CR", (1, 2)), ("CRD", (3, 4))):
                ok = [sum(res[k]["cond"][c]["units"][r * 10 + d]["acc"] * res[k]["cond"][c]["units"][r * 10 + d]["n"]
                          for d in days) / sum(res[k]["cond"][c]["units"][r * 10 + d]["n"] for d in days) for k in keys]
                row[c] = float(np.mean(ok))
                row[c + "_per_day"] = {d: float(np.mean([res[k]["cond"][c]["units"][r * 10 + d]["acc"] for k in keys]))
                                       for d in days}
            S["test_rx"][r] = row
        # largest confusions, summed over seeds and test units
        for c in ("CR", "CRD"):
            cm = sum(np.array(res[k]["cond"][c]["units"][u]["conf"]) for k in keys for u in res[k]["cond"][c]["units"])
            off = cm.copy()
            np.fill_diagonal(off, 0)
            idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:6], off.shape))[0]
            S["top_confusions"][f"r{rd}_{c}"] = dict(
                errors=int(off.sum()), total=int(cm.sum()),
                top=[dict(true=int(i), pred=int(j), count=int(off[i, j]), share_of_true=float(off[i, j] / cm[i].sum()),
                          share_of_errors=float(off[i, j] / max(off.sum(), 1))) for i, j in idx],
                per_tx_acc=[float(100 * cm[t, t] / cm[t].sum()) if cm[t].sum() else None for t in range(TX)])
            # the receiver x (true, pred) cells with most errors
            cells = []
            for u in res[keys[0]]["cond"][c]["units"]:
                cu = sum(np.array(res[k]["cond"][c]["units"][u]["conf"]) for k in keys)
                np.fill_diagonal(cu, 0)
                i, j = np.unravel_index(cu.argmax(), cu.shape)
                cells.append(dict(unit=int(u), rx=RX_NAMES[u // 10], day=int(u % 10), true=int(i), pred=int(j),
                                  count=int(cu[i, j]), unit_errors=int(cu.sum())))
            S["top_confusions"][f"r{rd}_{c}"]["worst_units"] = sorted(cells, key=lambda d: -d["unit_errors"])[:5]
    # coherence vs accuracy over test units (seed mean)
    for c in ("CR", "CRD", "seen_d34"):
        xs, ys, ms = [], [], []
        for rd in folds:
            keys = [k for k in res if k.startswith(f"r{rd}_")]
            for u in res[keys[0]]["cond"][c]["units"]:
                xs.append(units[str(u)]["coh_median"])
                ys.append(np.mean([res[k]["cond"][c]["units"][u]["acc"] for k in keys]))
                ms.append(RX_MODEL[u // 10])
        ms = np.array(ms)
        S["coherence_vs_acc"][c] = dict(all=corr(xs, ys), **{m: corr(np.array(xs)[ms == m], np.array(ys)[ms == m])
                                                              for m in MODELS})
    # probes: mean over (fold, seed) of probe acc - clean acc
    for s in ("CR", "seen_same"):
        names = list(next(iter(res.values()))["probes"][s])
        S["probes_mean_drop"][s] = {
            n: {g: float(np.mean([res[k]["probes"][s][n][g] - res[k]["probes"][s]["clean"][g] for k in res
                                  if g in res[k]["probes"][s][n]])) for g in ("all",) + MODELS
                if any(g in res[k]["probes"][s][n] for k in res)} for n in names}
        S["probes_mean_drop"][s]["clean_acc"] = {
            g: float(np.mean([res[k]["probes"][s]["clean"][g] for k in res if g in res[k]["probes"][s]["clean"]]))
            for g in ("all",) + MODELS if any(g in res[k]["probes"][s]["clean"] for k in res)}
    S["bn"] = {k: v["bn"] for k, v in res.items()}
    return S


def print_summary(S):
    print("\n== four conditions per fold (mean over seeds, %)")
    for rd, d in S["cond_by_fold"].items():
        print(f"  fold {rd}: " + "  ".join(f"{c} {d[c]['mean']:.2f}" for c in CONDS))
    print("== four conditions per receiver model (mean over folds/seeds, %)")
    for c, d in S["cond_by_model"].items():
        print(f"  {c}: " + "  ".join(f"{m} {v:.2f}" for m, v in d.items()))
    print("== test receivers (CR / CRD, %)")
    for r, d in S["test_rx"].items():
        print(f"  [{r:2d}] {d['name']:6s} {d['model']} fold {d['fold']}: {d['CR']:.1f} / {d['CRD']:.1f}")
    print("== top confusions")
    for k, d in S["top_confusions"].items():
        print(f"  {k}: err {d['errors']}/{d['total']}  " +
              ", ".join(f"{t['true']}->{t['pred']} {t['count']}" for t in d["top"][:4]))
    print("== coherence vs accuracy:", json.dumps(S["coherence_vs_acc"]))
    for s, d in S["probes_mean_drop"].items():
        print(f"== probes on {s} (drop vs clean, points):")
        print("  " + "  ".join(f"{n} {v['all']:+.2f}" for n, v in d.items() if n not in ("clean", "clean_acc")))
    print("== BN:", json.dumps(S["bn"])[:600])


# ---------------------------------------------------------------- model part (torch, authors' code)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--bs", type=int, default=1024)
    a = ap.parse_args()
    import torch
    from sklearn.model_selection import train_test_split
    sys.path.insert(0, os.getcwd())
    from utils.load_data import load_single_dataset, preprocessing   # noqa: E402  (authors' loader)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    U = {(r, d): load_single_dataset("ManyRx", r, d, TX, "non_equalized") for r in range(RX) for d in DAYS}
    for (r, d), (x, y) in U.items():
        assert x.ndim == 3 and x.shape[1] == 2, x.shape
    Z = {k: to_c(v[0]) for k, v in U.items()}
    F = {k: cfo_fine(z) for k, z in Z.items()}
    print(f"loaded {len(U)} units, {sum(len(v[1]) for v in U.values())} signals, shape {U[(0, 1)][0].shape}, "
          f"dtype {U[(0, 1)][0].dtype}, {time.time() - t0:.0f}s", flush=True)

    # ---- model-independent unit statistics
    upath = os.path.join(a.out, "units.json")
    keys = sorted(U)
    offs_all, _ = group([F[k] for k in keys])
    units = {}
    for k, o in zip(keys, offs_all):
        r, d = k
        y = U[k][1]
        units[str(r * 10 + d)] = dict(
            rx=r, name=RX_NAMES[r], model=RX_MODEL[r], day=d, n=int(len(y)),
            n_per_tx=[int((y == t).sum()) for t in range(TX)], coh_median=float(np.median(stf_coherence(Z[k]))),
            cfo_median_per_tx=[float(np.median(F[k][y == t])) if (y == t).any() else None for t in range(TX)],
            lo_offset_common=float(o), lo_offset_fold={})
    for rd in range(NROUND):
        train, test = fold_split(rd)
        toffs, ref = group([F[(r, d)] for r in train for d in (1, 2)])
        for (r, d), o in zip([(r, d) for r in train for d in (1, 2)], toffs):
            units[str(r * 10 + d)]["lo_offset_fold"][rd] = float(o)
        for r in test:
            for d in DAYS:
                units[str(r * 10 + d)]["lo_offset_fold"][rd] = align_to(F[(r, d)], ref)
    json.dump(units, open(upath, "w"), indent=1)
    print(f"units.json written, {time.time() - t0:.0f}s", flush=True)

    def predict_probs(model, x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), a.bs):
                _, o = model(torch.from_numpy(np.ascontiguousarray(x[i:i + a.bs], dtype=np.float32)).cuda())
                out.append(torch.softmax(o.float(), 1).cpu().numpy())
        return np.concatenate(out)

    def predict(model, x):
        return predict_probs(model, x).argmax(1)

    res = {}
    PROBES = probe_list()
    for rd in a.rounds:
        train, test = fold_split(rd)
        # seen / same days: the authors' 30 % validation split, reproduced on an index array
        tr_units = [(r, d) for r in train for d in (1, 2)]
        uid_all = np.concatenate([np.full(len(U[u][1]), u[0] * 10 + u[1]) for u in tr_units])
        for seed in a.seeds:
            path = os.path.join(a.weights, f"FAN_DSCNN_16_128_2.0_0.0_0_ManyRx_[1, 2]_test_round={rd}_seed={seed}.pth")
            if not os.path.exists(path):
                print("MISSING", path, flush=True)
                continue
            model = torch.load(path, map_location="cuda", weights_only=False).eval()
            _, vidx = train_test_split(np.arange(len(uid_all)), test_size=0.3, random_state=seed)
            xt = np.concatenate([U[u][0] for u in tr_units])
            yt = np.concatenate([U[u][1] for u in tr_units])
            sets = {"seen_same": (xt[vidx], yt[vidx], uid_all[vidx])}
            for c, rxs, days in (("seen_d34", train, (3, 4)), ("CR", test, (1, 2)), ("CRD", test, (3, 4))):
                us = [(r, d) for r in rxs for d in days]
                sets[c] = (np.concatenate([U[u][0] for u in us]), np.concatenate([U[u][1] for u in us]),
                           np.concatenate([np.full(len(U[u][1]), u[0] * 10 + u[1]) for u in us]))
            entry = {"cond": {}, "probes": {}, "bn": None, "n_val": int(len(vidx))}
            P, Y, UI, CI = [], [], [], []
            for ci, c in enumerate(CONDS):
                x, y, uid = sets[c]
                pr = predict_probs(model, x)
                pred = pr.argmax(1)
                entry["cond"][c] = dict(overall=by_model(pred == y, uid), units=unit_table(pred, y, uid))
                P.append(pr.astype(np.float16)), Y.append(y.astype(np.int8)), UI.append(uid.astype(np.int16))
                CI.append(np.full(len(y), ci, np.int8))
            np.savez_compressed(os.path.join(a.out, f"probs_r{rd}_s{seed}.npz"), probs=np.concatenate(P),
                                label=np.concatenate(Y), unit=np.concatenate(UI), cond=np.concatenate(CI),
                                cond_names=np.array(CONDS))
            print(f"r{rd} s{seed} " + " ".join(f"{c} {entry['cond'][c]['overall']['all']:.2f}" for c in CONDS)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            # ---- nuisance probes
            for s in ("CR", "seen_same"):
                x, y, uid = sets[s]
                z = to_c(x)
                entry["probes"][s] = {}
                for name, fn, renorm in PROBES:
                    rng = np.random.RandomState(12345)
                    xp = to_r(fn(z, rng))
                    if renorm:
                        xp = preprocessing(xp.copy())
                    entry["probes"][s][name] = by_model(predict(model, xp) == y, uid)
            print(f"r{rd} s{seed} probes CR: " + " ".join(
                f"{n} {v['all']:.1f}" for n, v in entry["probes"]["CR"].items()) + f"  ({time.time() - t0:.0f}s)",
                flush=True)
            # ---- BatchNorm re-estimation on each test unit's own signals (no labels, no gradients)
            bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            norm_types = sorted({type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__})
            if not bns:
                entry["bn"] = dict(has_bn=False, norm_layers=norm_types)
            else:
                bn = {"has_bn": True, "n_bn": len(bns), "norm_layers": norm_types}
                for c in ("CR", "CRD"):
                    x, y, uid = sets[c]
                    pred = np.empty(len(y), int)
                    for u in np.unique(uid):
                        m = uid == u
                        mm = copy.deepcopy(model).eval()
                        for b in mm.modules():
                            if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
                                b.reset_running_stats()
                                b.momentum = None
                                b.train()
                        with torch.no_grad():
                            xu = x[m]
                            for i in range(0, len(xu), a.bs):
                                mm(torch.from_numpy(np.ascontiguousarray(xu[i:i + a.bs], dtype=np.float32)).cuda())
                        mm.eval()
                        pred[m] = predict(mm, x[m])
                    bn[c] = dict(adapted=by_model(pred == y, uid), clean=entry["cond"][c]["overall"],
                                 units={int(u): float(100 * (pred[uid == u] == y[uid == u]).mean()) for u in np.unique(uid)})
                entry["bn"] = bn
            print(f"r{rd} s{seed} BN: {json.dumps({k: (v['adapted']['all'] if isinstance(v, dict) and 'adapted' in v else v) for k, v in entry['bn'].items()})}", flush=True)
            res[f"r{rd}_s{seed}"] = entry
            del model
            torch.cuda.empty_cache()

    S = summarize(res, units)
    # four conditions per receiver model, pooled over folds / seeds (mean of per-run accuracies)
    for c in CONDS:
        S["cond_by_model"][c] = {g: float(np.mean([v["cond"][c]["overall"][g] for v in res.values()
                                                   if g in v["cond"][c]["overall"]]))
                                 for g in ("all",) + MODELS if any(g in v["cond"][c]["overall"] for v in res.values())}
    out = dict(seeds=a.seeds, rounds=a.rounds, weights=a.weights, rx_names=RX_NAMES, rx_model=RX_MODEL,
               conds=list(CONDS), runs=res, summary=S, seconds=time.time() - t0)
    json.dump(out, open(os.path.join(a.out, "diag_fan_errors.json"), "w"))
    print_summary(S)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
