"""
Sound-protocol recovery for the supplement (item 2) — self-contained port of the
validated parent-dir pipeline (psi_sound_recovery.py + p3b_recovery_comparison.py) into
the supplement repo, which has its own core_psi_xlstm / core_physics_init.

Replaces the supplement's length-1 + through-origin + scale-invariant-latent_corr recovery
with: true-sequence processing, free-intercept slope estimator, multi-realization training,
held-out-driver validation, and the amplitude-sensitive `latent_slope` metric.

This module provides the shared core + run_p3_sound (method comparison). P7/P8/P9 sound
versions build on the same core (run_p7_sound/run_p8_sound/run_p9_sound — next pass).
"""
import os
import numpy as np
import torch
import torch.nn as nn

import supplementary_experiments as SE
from core_psi_xlstm import mLSTMBlock, sLSTMBlock
try:
    from core_physics_init import apply_psi_vortex_init
    _HAS_PHYS = True
except Exception:
    _HAS_PHYS = False

DEV = SE.DEVICE
T_AMB = SE.T_AMB
V_READ = SE.V_READ
# FROZEN CONFIG (package §7b): stride-8, M=5 train + 3 held-out, R2>=0.8 gate.
# Defaults are the FINAL values; override via env for fast DEV runs.
STRIDE = int(os.environ.get("STRIDE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 350))
M = int(os.environ.get("M", 5))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 5))
R2_ACCEPT = 0.8

# Provenance guard (package §7b): every *_sound.csv carries its config; build_supplement.py
# calls assert_frozen() before rendering and HARD-FAILS on DEV-budget inputs, so a stale DEV
# CSV cannot silently feed a final table.
RUN_CONFIG = dict(stride=STRIDE, epochs=EPOCHS, M=M, seeds=os.environ.get("SEEDS", "0,1,2"))
FROZEN = dict(stride=8, epochs=350, M=5)   # final config (cfg_stride reflects the module-level
                                           # stride; the P8 tau-fast per-cell stride is a data column)


def _write_table(df, out_csv):
    for k, v in RUN_CONFIG.items():
        df[f"cfg_{k}"] = v
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}  (cfg: {RUN_CONFIG})")
    return df


def assert_frozen(csv_path):
    """Hard-fail unless the CSV was produced at the frozen final config. Call before rendering
    any final table/figure in build_supplement.py."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    for k, want in FROZEN.items():
        col = f"cfg_{k}"
        if col not in d.columns:
            raise SystemExit(f"PROVENANCE GUARD: {csv_path} missing {col} (un-stamped/old output).")
        got = int(d[col].iloc[0])
        if got != want:
            raise SystemExit(f"PROVENANCE GUARD: {csv_path} {col}={got} != frozen {want} "
                             f"(DEV-budget output? refusing to render final tables).")


# --------------------------- estimator + metric ---------------------------- #
def free_intercept_alpha(I_pred, dT, g_base, dT_band=0.5):
    """Free-intercept slope of log(I/(V_read*g_base)) vs dT (with R^2). A constant
    prediction yields slope 0 (unlike the through-origin estimator)."""
    I = np.asarray(I_pred, np.float64).flatten()
    dT = np.asarray(dT, np.float64).flatten()[:len(I)]
    lr = np.log(np.where(I > 1e-30, I, 1e-30) / (V_READ * g_base))
    mask = (np.abs(dT) > dT_band) & np.isfinite(lr) & (np.abs(lr) < 8.0)
    if mask.sum() < 10:
        mask = np.isfinite(lr) & (np.abs(lr) < 8.0)
    if mask.sum() < 5 or np.std(dT[mask]) < 1e-12 or np.std(lr[mask]) < 1e-12:
        return 0.0, 0.0
    slope, _ = np.polyfit(dT[mask], lr[mask], 1)
    r = np.corrcoef(dT[mask], lr[mask])[0, 1]
    return float(slope), (float(r * r) if np.isfinite(r) else 0.0)


def latent_slope(latent, dT):
    """Amplitude-SENSITIVE replacement for scale-invariant |Pearson r| latent_corr:
    regression slope of a 1-D latent summary vs dT. Report Pearson r separately, only as
    an explicit coupling-*timescale* (tau-shape) diagnostic."""
    latent = np.asarray(latent, np.float64).flatten()
    dT = np.asarray(dT, np.float64).flatten()[:len(latent)]
    if np.std(latent) < 1e-12 or np.std(dT) < 1e-12:
        return 0.0
    return float(np.polyfit(dT, latent, 1)[0])


# ------------------------------- models ------------------------------------ #
class SeqRecurrent(nn.Module):
    """Paper's mLSTM/sLSTM consumed as ONE sequence (recurrence integrates history)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.mlstm = mLSTMBlock(1, hidden, memory_size=max(8, hidden // 2))
        self.slstm = sLSTMBlock(hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, X, return_fused=False):
        h1, _, _ = self.mlstm(X.unsqueeze(0))
        h2, _, _ = self.slstm(h1)
        fused = h2.squeeze(0)
        out = self.fc(fused)
        return (out, fused) if return_fused else out


class SeqMLP(nn.Module):
    """Pointwise feed-forward (no temporal integration) — the architecture baseline."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.fc = nn.Linear(hidden, 1)

    def forward(self, X):
        return self.fc(self.net(X))


def _build(kind):
    return SeqMLP() if kind == "mlp" else SeqRecurrent()


def _apply_init(m, mode):
    if mode == "physics" and _HAS_PHYS:
        try:
            apply_psi_vortex_init(m, pde_type="thermal")
        except Exception:
            pass
    elif mode == "orthogonal":
        for n, p in m.named_parameters():
            if p.dim() == 2 and ("R_" in n or "W_" in n):
                nn.init.orthogonal_(p)
    # random -> default


# --------------------------- data realizations ----------------------------- #
def realization(alpha, seed, stride=None, **gen_kw):
    stride = STRIDE if stride is None else stride
    ds = SE.generate_thermal_data(alpha, seed=seed, **gen_kw)
    gb = ds["g_base"]
    n = ds["n_train"]
    V = ds["full_V"][:n][::stride]
    I = ds["full_I"][:n][::stride]
    dT = ds["dT"][:n][::stride]
    Vt = torch.tensor(V, dtype=torch.float32, device=DEV).view(-1, 1)
    y = torch.tensor(np.log(np.maximum(I, 1e-30) / (V_READ * gb)),
                     dtype=torch.float32, device=DEV).view(-1, 1)
    return (Vt, y), (Vt, dT), gb


def _train_psi(train, epochs=EPOCHS, lr=8e-4, clip=0.5, seed=0):
    """Train one physics-init recurrent model (seq, log-space); training-loss checkpoint."""
    torch.manual_seed(seed)
    m = _build("psi").to(DEV); _apply_init(m, "physics")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); crit = nn.MSELoss()
    best, bl = None, float("inf")
    for _ in range(epochs):
        m.train(); ls = []
        for Vt, y in train:
            opt.zero_grad(); loss = crit(m(Vt), y)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), clip); opt.step()
            ls.append(loss.item())
        sched.step()
        if ls and np.mean(ls) < bl:
            bl = float(np.mean(ls)); best = {k: v.detach().clone() for k, v in m.state_dict().items()}
    if best:
        m.load_state_dict(best)
    return m


def _latent_slope_heldout(model, held):
    """Mean latent_slope over held-out drivers: slope of the fused state's 1-D summary
    (first principal component) vs ΔT — amplitude-sensitive (replaces scale-invariant corr)."""
    model.eval(); out = []
    for item in held:
        Vt, dT = item[0], item[1]
        with torch.no_grad():
            _, fused = model(Vt, return_fused=True)
        F = fused.cpu().numpy()
        F = F - F.mean(0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(F, full_matrices=False)
            latent = U[:, 0] * S[0]
        except Exception:
            latent = F[:, 0]
        out.append(latent_slope(latent, dT))
    return float(np.mean(out))


# ------------------------------ training ----------------------------------- #
def _recover(model, held, gb):
    model.eval()
    al, r2 = [], []
    for Vt, dT in held:
        with torch.no_grad():
            yp = model(Vt).cpu().numpy().flatten()
        a, r = free_intercept_alpha(np.exp(yp) * (V_READ * gb), dT, gb)
        al.append(a); r2.append(r)
    return float(np.mean(al)), float(np.mean(r2))


def _train_track(kind, mode, train, held, gb, epochs=EPOCHS, lr=8e-4, clip=0.5, seed=0):
    """Train (seq, log-space); return (held-out alpha, held-out R2, epochs->identifiable).
    Checkpoint uses TRAINING loss ONLY (blind to held-out — no test leakage)."""
    torch.manual_seed(seed)
    m = _build(kind).to(DEV); _apply_init(m, mode)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best, best_loss, traj = None, float("inf"), []
    for ep in range(epochs):
        m.train(); losses = []
        for Vt, y in train:
            opt.zero_grad(); loss = crit(m(Vt), y)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), clip); opt.step()
            losses.append(loss.item())
        sched.step()
        if losses and np.mean(losses) < best_loss:
            best_loss = float(np.mean(losses))
            best = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % EVAL_EVERY == 0:
            traj.append((ep + 1, _recover(m, held, gb)[1]))
    # sustained onset: first eval epoch from which R2 stays >= accept for the remainder
    e2i = next((ep for i, (ep, _r) in enumerate(traj)
                if all(r >= R2_ACCEPT for _e, r in traj[i:])), None)
    if best is not None:
        m.load_state_dict(best)
    a, r2 = _recover(m, held, gb)
    return a, r2, e2i


# ------------------------------ P3 (sound) --------------------------------- #
def run_p3_sound(out_csv="supplementary_experiments_output/p3_alpha_sweep_summary_sound.csv",
                 alphas=(0.05, 0.08, 0.10, 0.15, 0.20), seeds=(0, 1, 2), only_methods=None):
    import pandas as pd
    methods = [("MLP", "mlp", "random"), ("Vanilla xLSTM", "psi", "random"),
               ("Manual Psi-xLSTM", "psi", "orthogonal"), ("Psi-Vortex", "psi", "physics")]
    if only_methods:                                    # e.g. Table V = Psi-Vortex only (fast)
        methods = [m for m in methods if m[0] in only_methods]
    train_seeds = [42, 123, 456, 7, 99][:M]
    held_seeds = [777, 888, 555]
    rows = []
    for ag in alphas:
        train = [realization(ag, s)[0] for s in train_seeds]
        held = [realization(ag, s)[1] for s in held_seeds]
        gb = realization(ag, train_seeds[0])[2]
        for name, kind, mode in methods:
            a_s, r2_s, e2i_s = [], [], []
            for sd in seeds:
                a, r2, e2i = _train_track(kind, mode, train, held, gb, seed=sd)
                a_s.append(a); r2_s.append(r2); e2i_s.append(e2i)
            a_s, r2_s = np.array(a_s), np.array(r2_s)
            e2i_vals = [e for e in e2i_s if e is not None]
            rows.append(dict(
                alpha_gt=ag, method=name,
                heldout_alpha_mean=float(a_s.mean()), heldout_alpha_std=float(a_s.std()),
                heldout_r2_mean=float(r2_s.mean()),
                err_pct=(abs(a_s.mean() - ag) / ag * 100 if ag > 0 else float("nan")),
                epochs_to_ident_mean=(float(np.mean(e2i_vals)) if e2i_vals else float("nan")),
                n_identifiable=len(e2i_vals), n_seeds=len(seeds),
                recovers=bool(r2_s.mean() >= R2_ACCEPT)))
            print(f"  a={ag:.2f} {name:16} R2={r2_s.mean():.2f} err={rows[-1]['err_pct']:.1f}% "
                  f"e2i={rows[-1]['epochs_to_ident_mean']}", flush=True)
    return _write_table(pd.DataFrame(rows), out_csv)


# ------------------------------ P7 (sound) --------------------------------- #
# Control realization builders. Each returns ((Vt, y_train), (Vt, dT, y_heldout), gb).
# The held-out pair carries y so we can measure FIT QUALITY (relMSE) — the discriminator
# between the two null failure modes (α=0/fake-drift fit but slope≈0; shuffled/victim-only
# fail to fit held-out at all).
def _pack(V, I, dT, gb):
    Vt = torch.tensor(V, dtype=torch.float32, device=DEV).view(-1, 1)
    y = torch.tensor(np.log(np.maximum(I, 1e-30) / (V_READ * gb)),
                     dtype=torch.float32, device=DEV).view(-1, 1)
    return Vt, y, dT


def _ctrl_realization(control, alpha, seed):
    ds = SE.generate_thermal_data(0.0 if control in ("alpha0", "fakedrift") else alpha, seed=seed)
    gb = ds["g_base"]; n = ds["n_train"]
    V = ds["full_V"][:n][::STRIDE].copy()
    I = ds["full_I"][:n][::STRIDE].copy()
    dT = ds["dT"][:n][::STRIDE]
    if control == "fakedrift":
        I = I + np.linspace(0, 3e-6, len(I))            # unrelated slow ramp, no coupling
    if control == "shuffled":
        I_for_y = I.copy()                              # current still from true T
        V = V[np.random.RandomState(seed).permutation(len(V))]  # break causal link in input
        I = I_for_y
    if control == "victim":
        V = np.zeros_like(V)                            # remove driver input
    Vt, y, dT = _pack(V, I, dT, gb)
    return (Vt, y), (Vt, dT, y), gb


def _recover_fit(model, held, gb, ref_var):
    """Held-out recovery slope, R², and relMSE (fit quality). relMSE is normalized by a
    FIXED yardstick `ref_var` (the genuine-signal variance) so it is comparable across
    controls — normalizing by each control's own variance breaks for α=0 (var→0)."""
    model.eval(); al, r2, rel = [], [], []
    crit = nn.MSELoss()
    for Vt, dT, y in held:
        with torch.no_grad():
            yp = model(Vt)
        rel.append(float(crit(yp, y).item()) / (ref_var + 1e-12))
        a, r = free_intercept_alpha(np.exp(yp.cpu().numpy().flatten()) * (V_READ * gb), dT, gb)
        al.append(a); r2.append(r)
    return float(np.mean(al)), float(np.mean(r2)), float(np.mean(rel))


def run_p7_sound(out_csv="supplementary_experiments_output/p7_negative_controls_sound.csv",
                 alpha=0.08, seeds=(0, 1, 2)):
    import pandas as pd
    # (label, control-key, expected-signature). Held-out uses GENUINE drivers for the
    # driver-level nulls (shuffled/victim) so cross-realization breaks any positional backdoor.
    controls = [("genuine a=0.08", "genuine", "fits; R2>=0.8; slope~alpha"),
                ("alpha=0", "alpha0", "fits; slope~0"),
                ("fake slow drift", "fakedrift", "fits; slope~0"),
                ("shuffled driver", "shuffled", "FAILS fit (high relMSE), R2<0.8"),
                ("victim-only", "victim", "FAILS fit (high relMSE), R2<0.8")]
    train_seeds = [42, 123, 456, 7, 99][:M]
    held_seeds = [777, 888, 555]
    # fixed yardstick for relMSE: variance of the genuine held-out target
    gen_held = [_ctrl_realization("genuine", alpha, s)[1] for s in held_seeds]
    ref_var = float(np.mean([float(y.var().item()) for _Vt, _dT, y in gen_held]))
    rows = []
    for label, key, sig in controls:
        # held-out: genuine drivers for shuffled/victim (test generalization to real V->thermal);
        # same-type for the others.
        held_key = "genuine" if key in ("shuffled", "victim") else key
        held = [_ctrl_realization(held_key, alpha, s)[1] for s in held_seeds]
        for sd in seeds:
            train = [_ctrl_realization(key, alpha, s)[0] for s in train_seeds]
            gb = _ctrl_realization(key, alpha, train_seeds[0])[2]
            # Use the SAME validated trainer as Table V/P3 (_train_psi seeds BEFORE init), so the
            # genuine control is identical to the Table V alpha=0.08 Psi-Vortex recovery by
            # construction. (The previous inline loop seeded AFTER init -> divergent, undertrained
            # genuine control; fixed 2026-06-11.)
            m = _train_psi(train, seed=sd)
            a, r2, rel = _recover_fit(m, held, gb, ref_var)
            rows.append(dict(control=label, seed=sd, slope=a, heldout_r2=r2, heldout_relmse=rel,
                             expected=sig))
        sub = [r for r in rows if r["control"] == label]
        print(f"  {label:18} slope={np.mean([r['slope'] for r in sub]):+.4f} "
              f"R2={np.mean([r['heldout_r2'] for r in sub]):.2f} "
              f"relMSE={np.mean([r['heldout_relmse'] for r in sub]):.2f}  [{sig}]", flush=True)
    return _write_table(pd.DataFrame(rows), out_csv)


# ------------------------------ P8 (sound) --------------------------------- #
def run_p8_sound(out_csv="supplementary_experiments_output/p8_cross_geometry_sound.csv",
                 alpha=0.08, seed=0):
    import pandas as pd
    # (name, generate_thermal_data kwargs, cell stride). tau-fast at stride-4 (FROZEN-CONFIG
    # exception, package §7b): tau=0.02 shrinks the time constant 2.5x, under-resolved at
    # stride-8; flag inline in caption.
    configs = [
        ("spacing 50um (strong)", dict(tau_th=0.05, heat_coeff=1200.0, n_victims=1), 8),
        ("spacing 100um (nominal)", dict(tau_th=0.05, heat_coeff=800.0, n_victims=1), 8),
        ("spacing 200um (weak)", dict(tau_th=0.05, heat_coeff=450.0, n_victims=1), 8),
        ("vertical R x2", dict(tau_th=0.05, heat_coeff=400.0, n_victims=1), 8),
        ("vertical R x5", dict(tau_th=0.05, heat_coeff=160.0, n_victims=1), 8),
        ("tau fast", dict(tau_th=0.02, heat_coeff=800.0, n_victims=1), 4),   # stride exception
        ("tau slow", dict(tau_th=0.10, heat_coeff=800.0, n_victims=1), 8),
        ("2 victim layers", dict(tau_th=0.05, heat_coeff=800.0, n_victims=2), 8),
        ("4 victim layers", dict(tau_th=0.05, heat_coeff=800.0, n_victims=4), 8),
    ]
    train_seeds = [42, 123, 456, 7, 99][:M]; held_seeds = [777, 888, 555]
    rows = []
    for name, kw, st in configs:
        train = [realization(alpha, s, stride=st, **kw)[0] for s in train_seeds]
        held = [realization(alpha, s, stride=st, **kw)[1] for s in held_seeds]
        gb = realization(alpha, train_seeds[0], stride=st, **kw)[2]
        m = _train_psi(train, seed=seed)
        a, r2 = _recover(m, held, gb)
        ls = _latent_slope_heldout(m, held)
        rows.append(dict(geometry=name, **kw, stride=st, heldout_alpha=a, heldout_r2=r2,
                         err_pct=abs(a - alpha) / alpha * 100, latent_slope=ls,
                         stride_exception=(st != 8)))
        print(f"  {name:22} st={st} R2={r2:.2f} a={a:.4f} err={rows[-1]['err_pct']:.1f}% "
              f"lslope={ls:.3e}", flush=True)
    return _write_table(pd.DataFrame(rows), out_csv)


# ------------------------------ P9 (sound) --------------------------------- #
def _apply_artifact(I, artifact, seed):
    rng = np.random.RandomState(seed + 777)
    n = len(I); I = I.copy()
    if artifact == "none":
        return I
    if artifact == "read_noise":
        return I * (1 + 0.02 * rng.randn(n))
    if artifact == "amp_jitter":
        return I * (1 + 0.03 * rng.randn(n))
    if artifact == "baseline_drift":
        return I + np.linspace(0, 0.05 * float(np.mean(I)), n)
    if artifact == "missing_samples":
        idx = rng.choice(n, max(1, n // 20), replace=False)
        I[idx] = I[(idx - 1) % n]                       # hold-last-value gaps
        return I
    return I


def _artifact_realization(artifact, alpha, seed, stride=None):
    stride = STRIDE if stride is None else stride
    ds = SE.generate_thermal_data(alpha, seed=seed)
    gb = ds["g_base"]; n = ds["n_train"]
    V = ds["full_V"][:n][::stride]
    I = _apply_artifact(ds["full_I"][:n][::stride], artifact, seed)
    dT = ds["dT"][:n][::stride]
    Vt = torch.tensor(V, dtype=torch.float32, device=DEV).view(-1, 1)
    y = torch.tensor(np.log(np.maximum(I, 1e-30) / (V_READ * gb)),
                     dtype=torch.float32, device=DEV).view(-1, 1)
    return (Vt, y), (Vt, dT), gb


def run_p9_sound(out_csv="supplementary_experiments_output/p9_measurement_artifacts_sound.csv",
                 alpha=0.08, seed=0):
    import pandas as pd
    artifacts = ["none", "read_noise", "amp_jitter", "baseline_drift", "missing_samples"]
    train_seeds = [42, 123, 456, 7, 99][:M]; held_seeds = [777, 888, 555]
    rows = []; clean_err = None
    for art in artifacts:
        # corrupt BOTH train and held-out (package §7b: realistic measurement scenario)
        train = [_artifact_realization(art, alpha, s)[0] for s in train_seeds]
        held = [_artifact_realization(art, alpha, s)[1] for s in held_seeds]
        gb = _artifact_realization(art, alpha, train_seeds[0])[2]
        m = _train_psi(train, seed=seed)
        a, r2 = _recover(m, held, gb)
        ls = _latent_slope_heldout(m, held)
        err = abs(a - alpha) / alpha * 100
        if art == "none":
            clean_err = err
        rows.append(dict(artifact=art, heldout_alpha=a, heldout_r2=r2, err_pct=err,
                         degradation_pp=(err - clean_err if clean_err is not None else 0.0),
                         latent_slope=ls, corrupts="train+heldout"))
        print(f"  {art:16} R2={r2:.2f} a={a:.4f} err={err:.1f}% "
              f"degr={rows[-1]['degradation_pp']:+.1f}pp lslope={ls:.3e}", flush=True)
    return _write_table(pd.DataFrame(rows), out_csv)


if __name__ == "__main__":
    which = os.environ.get("RUN", "p3")
    sds = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
    if which == "tablev":   # Table V = Psi-Vortex held-out recovery sweep (fast headline path)
        run_p3_sound(out_csv="supplementary_experiments_output/tableV_recovery_sound.csv",
                     seeds=sds, only_methods={"Psi-Vortex"})
    elif which == "p7":
        run_p7_sound(seeds=sds)
    elif which == "p8":
        run_p8_sound()
    elif which == "p9":
        run_p9_sound()
    else:
        run_p3_sound(alphas=[float(a) for a in os.environ.get("ALPHAS", "0.08").split(",")], seeds=sds)
