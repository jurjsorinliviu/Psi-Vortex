"""
Decisive end-to-end test of the ESTIMATOR fix.

Re-run the five P7 controls with the paper's [V,t] inputs (no input change), but
recover alpha with BOTH estimators:
    thru-origin   : the current alpha_fit (zero intercept)  -> manufactures alpha
    free-intercept: ordinary deg-1 OLS                      -> proposed fix

A sound estimator should give the genuine positive control a clearly non-zero
alpha while pushing ALL FOUR nulls (alpha=0, shuffled driver, victim-only,
fake drift) toward ~0.
"""
import numpy as np
import torch

import supplementary_experiments as S
from diag_estimator import alpha_fit_free_intercept
from diag_nulls_no_t import make_controls


def main(seeds=(42, 123, 456), epochs=120):
    order = ["REF genuine a=0.08", "alpha=0", "shuffled driver", "victim-only", "fake slow drift"]
    null = {"alpha=0", "shuffled driver", "victim-only", "fake slow drift"}
    to, fi = {}, {}
    for seed in seeds:
        for name, ds in make_controls(seed).items():
            S.set_seed(seed)
            m = S.build_model("psi", 32); S.init_model(m, "physics")
            S.train_supervised(m, ds, epochs)
            to.setdefault(name, []).append(S.alpha_fit(m, ds)[0])
            fi.setdefault(name, []).append(alpha_fit_free_intercept(m, ds)[0])

    print("\n" + "=" * 78)
    print(f"{'control':22s} {'thru-origin (current)':>24s} {'free-intercept (fix)':>24s}")
    print("-" * 78)
    for k in order:
        a0 = np.array(to[k]); a1 = np.array(fi[k])
        tag = "null" if k in null else "GENUINE"
        print(f"{k:22s} {a0.mean():8.4f} +/- {a0.std():6.4f}      "
              f"{a1.mean():8.4f} +/- {a1.std():6.4f}    {tag}")
    print("=" * 78)

    g = np.mean(fi["REF genuine a=0.08"])
    worst_null = max(abs(np.mean(fi[k])) for k in null)
    print(f"\nfree-intercept: genuine={g:.4f}  worst null={worst_null:.4f}")
    if g > 2 * worst_null and g > 0.01:
        print("  => ESTIMATOR FIX WORKS: genuine separates cleanly from all nulls.")
    else:
        print("  => genuine does NOT separate from nulls even with the fix -> the")
        print("     model's thermal-shape fit is too weak on this benchmark (deeper issue).")


if __name__ == "__main__":
    main()
