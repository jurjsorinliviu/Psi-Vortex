"""
Fairness check: is the estimator confound a WEAK-SIGNAL effect, or does it void
recovery at all coupling strengths?

Train the genuine psi-vortex model across alpha_gt and recover with both
estimators. If free-intercept tracks alpha_gt at strong coupling but collapses at
weak coupling, the method genuinely works where the thermal signal is strong; the
through-origin estimator merely inflates the weak-coupling numbers (and the nulls).
"""
import numpy as np

import supplementary_experiments as S
from diag_estimator import alpha_fit_free_intercept


def main(seeds=(42, 123, 456), epochs=120):
    alphas = [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    print(f"\n{'alpha_gt':>8s} {'thru-origin':>22s} {'free-intercept':>22s} {'R2':>7s}")
    print("-" * 64)
    for ag in alphas:
        to, fi, r2s = [], [], []
        for seed in seeds:
            ds = S.generate_thermal_data(ag, seed=seed)
            S.set_seed(seed)
            m = S.build_model("psi", 32); S.init_model(m, "physics")
            S.train_supervised(m, ds, epochs)
            a0, r2 = S.alpha_fit(m, ds)
            a1, _ = alpha_fit_free_intercept(m, ds)
            to.append(a0); fi.append(a1); r2s.append(r2)
        to, fi = np.array(to), np.array(fi)
        print(f"{ag:8.2f} {to.mean():9.4f} +/- {to.std():6.4f}    "
              f"{fi.mean():9.4f} +/- {fi.std():6.4f}   {np.mean(r2s):6.3f}")
    print("-" * 64)
    print("read: 'free-intercept' ~ alpha_gt => genuine recovery; ~0 => artifact only")


if __name__ == "__main__":
    main()
