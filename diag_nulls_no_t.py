"""
Proof-of-concept for FIX #1: re-run the P7 negative controls with the time index
dropped (t channel zeroed), i.e. a V-only model. If V truly carries the coupling
information (shown by diag_time_channel.py: V-only recovers alpha), then removing
the t backdoor should make the DRIVER-LEVEL nulls correctly collapse to alpha ~ 0,
while the genuine positive control still recovers alpha.

Compares each control under [V,t] (current paper) vs V-only (proposed fix).
"""
import numpy as np
import torch

import supplementary_experiments as S


def _zero_t(ds):
    ds = dict(ds)
    for split in ("train", "val", "test"):
        if split in ds and isinstance(ds[split], dict):
            ds[split] = dict(ds[split])
            ds[split]["t"] = torch.zeros_like(ds[split]["t"])
    return ds


def make_controls(seed):
    """Return {name: dataset} replicating run_p7's five conditions."""
    out = {}
    ds = S.generate_thermal_data(0.08, seed=seed)
    out["REF genuine a=0.08"] = ds

    out["alpha=0"] = S.generate_thermal_data(0.0, seed=seed)

    out["shuffled driver"] = S._shuffle_driver(ds, seed=seed)

    dvo = S.generate_thermal_data(0.08, seed=seed)
    dvo["train"] = dict(dvo["train"]); dvo["train"]["V"] = torch.zeros_like(dvo["train"]["V"])
    dvo["val"] = dict(dvo["val"]);     dvo["val"]["V"] = torch.zeros_like(dvo["val"]["V"])
    out["victim-only"] = dvo

    dd = S.generate_thermal_data(0.0, seed=seed)
    drift = np.linspace(0, 3e-6, len(dd["full_I"]))
    i_tr, i_va = dd["n_train"], int(0.833 * len(dd["full_I"]))
    dd["train"] = dict(dd["train"]); dd["train"]["I"] = S.to_col(dd["full_I"][:i_tr] + drift[:i_tr])
    dd["val"] = dict(dd["val"]);     dd["val"]["I"] = S.to_col(dd["full_I"][i_tr:i_va] + drift[i_tr:i_va])
    out["fake slow drift"] = dd
    return out


def eval_regime(zero_t, seeds=(42, 123, 456), epochs=120):
    acc = {}
    for seed in seeds:
        for name, ds in make_controls(seed).items():
            d = _zero_t(ds) if zero_t else ds
            S.set_seed(seed)
            m = S.build_model("psi", 32); S.init_model(m, "physics")
            S.train_supervised(m, d, epochs)
            a = S.recover_alpha_ols(m, d)
            acc.setdefault(name, []).append(a)
    return {k: (np.mean(v), np.std(v)) for k, v in acc.items()}


if __name__ == "__main__":
    order = ["REF genuine a=0.08", "alpha=0", "shuffled driver", "victim-only", "fake slow drift"]
    null = {"alpha=0", "shuffled driver", "victim-only", "fake slow drift"}

    print("Training [V,t] baseline ...")
    base = eval_regime(zero_t=False)
    print("Training V-only (t dropped) ...")
    fix = eval_regime(zero_t=True)

    print("\n" + "=" * 74)
    print(f"{'control':22s} {'[V,t] alpha_rec':>20s} {'V-only alpha_rec':>20s}")
    print("-" * 74)
    for k in order:
        bm, bs = base[k]; fm, fs = fix[k]
        tag = "(null->want ~0)" if k in null else "(want ~0.08)"
        print(f"{k:22s} {bm:8.4f} +/- {bs:5.4f}    {fm:8.4f} +/- {fs:5.4f}   {tag}")
    print("=" * 74)

    # automated read of whether the fix works
    genuine = fix["REF genuine a=0.08"][0]
    drivers = [fix["shuffled driver"][0], fix["victim-only"][0]]
    print("\nV-only outcome:")
    print(f"  genuine recovers alpha = {genuine:.4f}  (want clearly > 0)")
    print(f"  shuffled-driver        = {fix['shuffled driver'][0]:.4f}  (want ~0)")
    print(f"  victim-only            = {fix['victim-only'][0]:.4f}  (want ~0)")
    if genuine > 0.03 and max(abs(d) for d in drivers) < 0.5 * genuine:
        print("  => FIX WORKS: dropping t separates genuine coupling from driver-level nulls.")
    else:
        print("  => inconclusive; inspect numbers above.")
