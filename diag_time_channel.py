"""
Decisive confound diagnostic: does the model need the electrical driver V, or can
it recover alpha from the time index t alone (curve-fitting a single trajectory)?

Trains three input regimes on the GENUINE alpha=0.08 thermal benchmark:
    [V,t]  : full input (as in the paper)
    V-only : t channel zeroed   -> isolates information carried by the driver
    t-only : V channel zeroed   -> isolates the time-index backdoor

For each: final val MSE and post-hoc OLS-recovered alpha (mean +/- std over seeds).

Interpretation:
  * If t-only reaches ~1e-10 MSE and recovers alpha ~0.05-0.06  -> CONFOUND:
    the benchmark is memorisable from t; driver-level nulls fail for this reason.
  * If t-only fails (high MSE, alpha far off) -> V carries necessary information;
    shuffled-driver/victim-only become contained, explainable artifacts.
"""
import numpy as np
import torch

import supplementary_experiments as S


def run_regime(label, zero_V=False, zero_t=False, seeds=(42, 123, 456), epochs=120):
    mses, alphas, r2s, corrs = [], [], [], []
    for seed in seeds:
        ds = S.generate_thermal_data(0.08, seed=seed)
        # clone the splits so we don't mutate the shared dict, then ablate channels
        for split in ("train", "val", "test"):
            ds[split] = dict(ds[split])
            if zero_V:
                ds[split]["V"] = torch.zeros_like(ds[split]["V"])
            if zero_t:
                ds[split]["t"] = torch.zeros_like(ds[split]["t"])
        S.set_seed(seed)
        m = S.build_model("psi", 32)
        S.init_model(m, "physics")
        h = S.train_supervised(m, ds, epochs)
        a, r2 = S.alpha_fit(m, ds)
        mses.append(h["val_mse"])
        alphas.append(a)
        r2s.append(r2)
        corrs.append(S.latent_thermal_correlation(m, ds))
    mses, alphas, r2s, corrs = map(np.asarray, (mses, alphas, r2s, corrs))
    print(f"\n{label}")
    print(f"  val_MSE     : mean={mses.mean():.3e}  per-seed={np.array2string(mses, precision=2)}")
    print(f"  alpha_rec   : mean={alphas.mean():.4f} +/- {alphas.std():.4f}  "
          f"(err vs 0.08 = {abs(alphas.mean()-0.08)/0.08*100:.1f}%)")
    print(f"  R2 (OLS fit): mean={r2s.mean():.3f}  per-seed={np.array2string(r2s, precision=3)}")
    print(f"  latent_corr : mean={corrs.mean():.4f}")
    return dict(label=label, mse=mses.mean(), alpha=alphas.mean(),
                alpha_std=alphas.std(), r2=r2s.mean(), corr=corrs.mean())


if __name__ == "__main__":
    print("=" * 70)
    print("TIME-CHANNEL CONFOUND DIAGNOSTIC  (genuine alpha=0.08, psi-vortex)")
    print("=" * 70)
    res = []
    res.append(run_regime("[V, t]  full input (paper config)"))
    res.append(run_regime("V-only  (t channel zeroed)", zero_t=True))
    res.append(run_regime("t-only  (V channel zeroed)  <-- the decisive test", zero_V=True))

    print("\n" + "=" * 70)
    print("VERDICT")
    t_only = res[2]
    if t_only["mse"] < 5e-9 and abs(t_only["alpha"] - 0.08) / 0.08 < 0.5:
        print("  CONFOUND CONFIRMED: t-only reaches low MSE and recovers alpha.")
        print("  The benchmark is curve-fittable from the time index; driver-level")
        print("  nulls fail because V is not necessary. -> must FIX (drop t / rerun),")
        print("  not merely reframe.")
    else:
        print("  V IS NECESSARY: t-only fails to fit / recover alpha.")
        print("  Shuffled-driver & victim-only are contained, explainable artifacts.")
    print("=" * 70)
