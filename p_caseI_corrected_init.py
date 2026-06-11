"""
Path B — ONE-SHOT corrected-init test (Case Study I, sequence mode).

Diagnosis: the 6.74x physics-init speedup died in sequence mode because the recurrent
prior (forced-negative diagonals + 14x shrink) hampers the now-active recurrence.
Targeted fix: restrict the symmetry prior to INPUT/OUTPUT projections; leave the
recurrent gates at plain orthogonal. Three arms on the SAME epochs-to-target harness
as item 4:
    random      : xavier/orthogonal baseline
    original    : apply_psi_vortex_init(pde_type='memristor')  (the published scheme)
    corrected   : symmetry prior on input/output only, recurrence left orthogonal

PRE-COMMITTED DECISION RULE (stated in REVISION_alpha_recovery_package.md sec.10 BEFORE
this ran): B succeeds iff corrected speedup >= 2x AND the 5-seed interval excludes 1x
(here: mean - std > 1.0). Below that -> go to path A, NO further init variants. One
principled fix, one run, one criterion.
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core_physics_init import PhysicsAwareInitializer, apply_psi_vortex_init
import p_caseI_seqmode_ablation as base

EPOCHS = int(os.environ.get("EPOCHS", 500))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123,456,7,99").split(",")]


class CorrectedInitializer(PhysicsAwareInitializer):
    """Physics-aware init with the recurrent over-constraint REMOVED: recurrent gates
    get plain orthogonal init (no dissipative mask, no forced-negative diagonal, no 0.1
    shrink); input/output projections keep the symmetry prior + scale."""
    def _initialize_weight(self, name, param):
        if any(k in name for k in ['R_', 'weight_hh', 'recurrent']):
            nn.init.orthogonal_(param, gain=1.0)
            return
        super()._initialize_weight(name, param)


def init_random(m):
    base.apply_random(m)


def init_original(m):
    apply_psi_vortex_init(m, pde_type="memristor")


def init_corrected(m):
    CorrectedInitializer(pde_type="memristor", epsilon=0.01, sigma=0.01, scale=0.1)(m)


def train_curve(init_fn, X, y, seed, epochs=EPOCHS, lr=1e-3, clip=1.0):
    torch.manual_seed(seed)
    m = base.SeqMemristor().to(base.DEV)
    init_fn(m)
    opt = torch.optim.Adam(m.parameters(), lr=lr); crit = nn.MSELoss()
    curve = []
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = crit(m(X), y)
        if not torch.isfinite(loss):
            curve.append(float('inf')); continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), clip); opt.step()
        curve.append(loss.item())
    return np.asarray(curve)


def main():
    V, t, I = base.load_memristor()
    scale = float(np.std(I)) or 1.0
    X = torch.tensor(np.stack([V, t], 1), device=base.DEV)
    y = torch.tensor((I / scale).reshape(-1, 1), device=base.DEV)
    print(f"Path B corrected-init test | SEQUENCE MODE | stride={base.STRIDE} epochs={EPOCHS} "
          f"seeds={SEEDS} N={len(V)}")
    print(f"{'seed':>5} {'random ep':>10} {'orig ep':>9} {'corr ep':>9} "
          f"{'orig x':>8} {'corr x':>8}")
    print("-" * 56)
    ro, rc = [], []
    for seed in SEEDS:
        c_r = train_curve(init_random, X, y, seed)
        c_o = train_curve(init_original, X, y, seed)
        c_c = train_curve(init_corrected, X, y, seed)
        target = float(np.min(c_r))               # shared target = slower(random) min loss
        er = base.epochs_to_target(c_r, target)
        eo = base.epochs_to_target(c_o, target)
        ec = base.epochs_to_target(c_c, target)
        rat_o = er / eo if eo > 0 else float('inf')
        rat_c = er / ec if ec > 0 else float('inf')
        ro.append(rat_o); rc.append(rat_c)
        print(f"{seed:>5} {er:>10} {eo:>9} {ec:>9} {rat_o:>7.2f}x {rat_c:>7.2f}x", flush=True)
    ro, rc = np.asarray(ro), np.asarray(rc)
    print("-" * 56)
    print(f"original  speedup (seq mode): {ro.mean():.2f}x +/- {ro.std():.2f}  (length-1 claim 6.74x)")
    print(f"corrected speedup (seq mode): {rc.mean():.2f}x +/- {rc.std():.2f}")
    # PRE-COMMITTED rule: succeed iff mean>=2 AND interval (mean-std) excludes 1
    success = (rc.mean() >= 2.0) and (rc.mean() - rc.std() > 1.0)
    verdict = ("B SUCCEEDS -> corrected-init narrative + rebuilt Case Study I tables"
               if success else
               "B FAILS -> path A (restructured intro), NO further init variants")
    print("=" * 56)
    print("PRE-COMMITTED RULE: corrected >= 2x AND (mean-std) > 1x")
    print(f"VERDICT: {verdict}")
    print("=" * 56)
    pd.DataFrame(dict(seed=SEEDS, ratio_original=ro, ratio_corrected=rc)).to_csv(
        "caseI_corrected_init.csv", index=False)


if __name__ == "__main__":
    main()
