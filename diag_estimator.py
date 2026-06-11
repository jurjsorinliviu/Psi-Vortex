"""
Isolate the alpha-recovery ESTIMATOR from any trained model.

alpha_fit() regresses log(I_pred / baseline) onto the ground-truth dT, through the
origin. We feed it hand-built "predictions" to see what alpha it reports when the
prediction contains, by construction, NO recovered coupling:

  perfect    : I_pred = true victim current      -> should report ~alpha_gt
  const_mean : I_pred = mean(true I) everywhere  -> contains NO dT structure -> want 0
  const_base : I_pred = V_read*g_base            -> log_ratio = 0            -> want 0
  noisy_mean : const_mean + small gaussian       -> want ~0

A faithful estimator reports ~alpha_gt only for 'perfect' and ~0 for the constants.
If the constants report a sizeable alpha, the estimator itself manufactures
coupling and the negative-control failure is an ESTIMATOR artifact, not a model one.
"""
import numpy as np
import torch

import supplementary_experiments as S


class FixedPred(torch.nn.Module):
    """A fake 'model' whose forward() ignores inputs and returns a preset vector."""
    def __init__(self, vec):
        super().__init__()
        self.vec = torch.as_tensor(vec, dtype=torch.float32).reshape(-1, 1)

    def forward(self, V, t):
        n = len(V)
        return self.vec[:n], {"fused": torch.zeros(n, 1)}


def alpha_fit_free_intercept(model, ds):
    """Same as S.alpha_fit but with a FREE intercept (ordinary deg-1 OLS).
    A constant prediction then yields slope ~ 0, as a sound estimator must."""
    model.eval()
    with torch.no_grad():
        pred, _ = model(ds["train"]["V"], ds["train"]["t"])
    I = pred.cpu().numpy().flatten()
    dT = ds["dT"][:len(I)]
    base = S.V_READ * ds["g_base"]
    eps = 1e-30
    log_ratio = np.log(np.where(I > eps, I, eps) / base)
    mask = (np.abs(dT) > 0.5) & (I > 0) & (np.abs(log_ratio) < 8.0)
    if mask.sum() < 10:
        mask = (I > 0) & (np.abs(log_ratio) < 8.0)
    if mask.sum() < 5:
        return 0.0, 0.0
    dTm, lrm = dT[mask], log_ratio[mask]
    if np.std(dTm) < 1e-12:
        return 0.0, 0.0
    slope, intercept = np.polyfit(dTm, lrm, 1)
    r = np.corrcoef(dTm, lrm)[0, 1]
    return float(slope), float(r * r) if np.isfinite(r) else 0.0


def probe(seed=42, alpha_gt=0.08):
    ds = S.generate_thermal_data(alpha_gt, seed=seed)
    I_true = ds["train"]["I"].cpu().numpy().flatten()
    base = S.V_READ * ds["g_base"]
    rng = np.random.RandomState(0)

    preds = {
        "perfect (I_pred=I_true)":      I_true,
        "const_mean (=mean I)":         np.full_like(I_true, I_true.mean()),
        "const_base (=V_read*g_base)":  np.full_like(I_true, base),
        "noisy_mean (mean + 1% noise)": I_true.mean() * (1 + 0.01 * rng.randn(len(I_true))),
        "linear ramp in I":             np.linspace(I_true.min(), I_true.max(), len(I_true)),
    }
    print(f"\nseed={seed}  alpha_gt={alpha_gt}   baseline=V_read*g_base={base:.3e}")
    print(f"  mean(I_true)={I_true.mean():.6e}  std(I_true)={I_true.std():.3e}")
    print(f"  {'synthetic prediction':32s} {'thru-origin':>12s} {'free-intcpt':>12s} {'R2':>8s}")
    print("  " + "-" * 68)
    for name, vec in preds.items():
        m = FixedPred(vec)
        a0, r2 = S.alpha_fit(m, ds)
        a1, _ = alpha_fit_free_intercept(m, ds)
        print(f"  {name:32s} {a0:12.4f} {a1:12.4f} {r2:8.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ESTIMATOR PROBE: what alpha does alpha_fit() report for")
    print("predictions that contain NO recovered coupling?")
    print("=" * 60)
    for a_gt in (0.08, 0.0):
        probe(seed=42, alpha_gt=a_gt)
