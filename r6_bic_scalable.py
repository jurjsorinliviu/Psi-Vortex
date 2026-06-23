"""Tier-2: empirical validation of the proposed scalable BIC approximations.

The differentiable BIC effective-DoF (core_adaptive_bic, Eq. 7)
    DoF = sum_i 1 / sum_j exp(-(w_i - w_j)^2 / (2*gamma^2))
is an O(W^2) double sum -- the reviewers' scalability concern (and the dominant
backward-pass cost). The manuscript PROPOSES two mitigations but did not validate
them. This script does:

  * mini-batch BIC : estimate the inner density of each w_i from a random subset of
                     B weights, scaled by W/B  -> O(W*B), unbiased.
  * random Fourier : phi(w)=sqrt(2/D)cos(omega w + b), omega~N(0,1/gamma^2); then
    features (RFF)   sum_j K(w_i,w_j) ~ phi(w_i) . sum_j phi(w_j)  -> O(W*D).

We check, vs the exact O(W^2) computation: (i) DoF accuracy, (ii) wall-clock scaling,
(iii) that the approximations remain feasible where the exact form is not, and
(iv) gradient alignment (cosine) so the cheap approximations are safe for backprop.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.1
FULL_MAX = 8000          # exact O(W^2) beyond this is skipped (memory)
B = 512                  # mini-batch size
D = 512                  # RFF feature count
WS = [500, 1000, 2000, 5000, 8000, 20000, 50000]


def make_weights(W, n_clusters=5, seed=0):
    """Clustered weight vector (DoF should be ~= n_clusters)."""
    rng = np.random.RandomState(seed)
    centers = np.linspace(-1.0, 1.0, n_clusters)
    w = np.concatenate([rng.normal(c, 0.02, W // n_clusters) for c in centers])
    return torch.tensor(w[:W], dtype=torch.float32, device=DEV)


def full_dof(w, gamma=GAMMA):
    d = (w.unsqueeze(1) - w.unsqueeze(0)) ** 2
    dens = torch.exp(-d / (2 * gamma ** 2)).sum(1)
    return torch.sum(1.0 / (dens + 1e-8))


def minibatch_dof(w, gamma=GAMMA, b=B, seed=0):
    W = len(w)
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.randint(0, W, (b,), generator=g, device=DEV)
    sub = w[idx]
    d = (w.unsqueeze(1) - sub.unsqueeze(0)) ** 2          # [W, b]
    dens = (W / b) * torch.exp(-d / (2 * gamma ** 2)).sum(1)
    return torch.sum(1.0 / (dens + 1e-8))


def rff_dof(w, gamma=GAMMA, d_feat=D, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    omega = torch.randn(d_feat, generator=g, device=DEV) / gamma
    bvec = 2 * np.pi * torch.rand(d_feat, generator=g, device=DEV)
    phi = np.sqrt(2.0 / d_feat) * torch.cos(torch.outer(w, omega) + bvec)  # [W, d]
    S = phi.sum(0)                                                          # [d]
    dens = phi @ S                                                          # [W]
    return torch.sum(1.0 / (dens.clamp_min(1e-3)))


def timed(fn, *a, reps=3):
    fn(*a)                       # warm-up (excludes CUDA kernel init from timing)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    val = None
    for _ in range(reps):
        val = fn(*a)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return float(val.item()), (time.perf_counter() - t0) / reps


def main():
    print(f"BIC scalability | device={DEV} gamma={GAMMA} B={B} D={D}")
    rows = []
    for W in WS:
        w = make_weights(W)
        rec = {"W": W}
        if W <= FULL_MAX:
            rec["full_dof"], rec["t_full"] = timed(full_dof, w)
        else:
            rec["full_dof"], rec["t_full"] = np.nan, np.nan
        rec["mb_dof"], rec["t_mb"] = timed(minibatch_dof, w)
        rec["rff_dof"], rec["t_rff"] = timed(rff_dof, w)
        rows.append(rec)
        fd = "  n/a (O(W^2))" if np.isnan(rec["full_dof"]) else f"{rec['full_dof']:7.2f}"
        print(f"  W={W:6d}  DoF full{fd}  mb {rec['mb_dof']:7.2f}  rff {rec['rff_dof']:7.2f}"
              f"  | t_full={rec['t_full'] if not np.isnan(rec['t_full']) else float('nan'):.4f}s"
              f"  t_mb={rec['t_mb']:.4f}s  t_rff={rec['t_rff']:.4f}s")

    df = pd.DataFrame(rows)
    # accuracy where exact is available
    sub = df.dropna(subset=["full_dof"])
    df.loc[sub.index, "mb_err_pct"] = 100 * np.abs(sub.mb_dof - sub.full_dof) / sub.full_dof
    df.loc[sub.index, "rff_err_pct"] = 100 * np.abs(sub.rff_dof - sub.full_dof) / sub.full_dof
    df.loc[sub.index, "speedup_mb"] = sub.t_full / df.loc[sub.index, "t_mb"]
    df.loc[sub.index, "speedup_rff"] = sub.t_full / df.loc[sub.index, "t_rff"]

    # gradient alignment at a moderate W (backward-pass safety)
    wg = make_weights(2000).clone().requires_grad_(True)
    gf = torch.autograd.grad(full_dof(wg), wg)[0]
    gm = torch.autograd.grad(minibatch_dof(wg), wg)[0]
    gr = torch.autograd.grad(rff_dof(wg), wg)[0]
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    cos_mb, cos_rff = cos(gf, gm), cos(gf, gr)
    print(f"\n  gradient cosine vs exact (W=2000):  mini-batch={cos_mb:.3f}  RFF={cos_rff:.3f}")

    df.to_csv(os.path.join(HERE, "r6_bic_scalable.csv"), index=False)
    with open(os.path.join(HERE, "r6_bic_scalable.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
        f.write("$W$ & full DoF & mini-batch DoF & RFF DoF & mini-batch err & RFF err \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            fd = "--" if np.isnan(r["full_dof"]) else f"{r['full_dof']:.2f}"
            me = "--" if np.isnan(r.get("mb_err_pct", np.nan)) else f"{r['mb_err_pct']:.1f}\\%"
            re_ = "--" if np.isnan(r.get("rff_err_pct", np.nan)) else f"{r['rff_err_pct']:.1f}\\%"
            f.write(f"{int(r['W'])} & {fd} & {r['mb_dof']:.2f} & {r['rff_dof']:.2f} & {me} & {re_} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"% gamma={GAMMA}, B={B}, D={D}. Exact O(W^2) skipped beyond W={FULL_MAX} (memory). "
                f"Gradient cosine vs exact: mini-batch={cos_mb:.3f}, RFF={cos_rff:.3f}.\n")

    # figure: wall-clock scaling
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ff = df.dropna(subset=["t_full"])
    ax.plot(ff.W, ff.t_full, "o-", c="C3", label="full  $O(W^2)$")
    ax.plot(df.W, df.t_mb, "s-", c="C0", label=f"mini-batch  $O(WB)$, B={B}")
    ax.plot(df.W, df.t_rff, "^-", c="C2", label=f"RFF  $O(WD)$, D={D}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of weights $W$ (log)")
    ax.set_ylabel("wall-clock per BIC eval (s, log)")
    ax.set_title("BIC effective-DoF: exact vs scalable approximations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r6_bic_scalable.png"), dpi=150)
    print("Saved: r6_bic_scalable.csv / .tex / .png")


if __name__ == "__main__":
    main()
