"""
Permanent regression tests for the corrected recovery estimator.

The kill-shot: a CONSTANT predictor (= mean current) recovers alpha_hat = 0 under
the corrected free-intercept estimator, whereas it reproduced the entire apparent
recovery curve under the original through-origin estimator. This file pins that
property so the defect cannot silently return.

Run:  python test_psi_sound_recovery.py   (exits non-zero on failure)
"""
import importlib.util
import numpy as np

import psi_sound_recovery as R

_spec = importlib.util.spec_from_file_location("ar", "14_alpha_recovery_experiment.py")
ar = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ar)
T_AMB = 298.0


def _data(alpha, seed=42):
    ds, T, gb = ar.generate_thermal_data_with_alpha(alpha, seed=seed)
    I = ds["train"]["I"].cpu().numpy().flatten()
    dT = T[: len(I)] - T_AMB
    return I, dT, gb


def test_perfect_recovers_alpha():
    for ag in (0.05, 0.08, 0.10, 0.20):
        I, dT, gb = _data(ag)
        a, r2 = R.free_intercept_alpha(I, dT, gb)
        assert abs(a - ag) < 1e-3, f"perfect alpha={ag}: got {a}"
        assert r2 > 0.999, f"perfect R2 too low: {r2}"


def test_constant_recovers_zero():
    """The original defect: a constant prediction must NOT manufacture coupling."""
    for ag in (0.05, 0.08, 0.10, 0.20):
        I, dT, gb = _data(ag)
        a, _ = R.free_intercept_alpha(np.full_like(I, I.mean()), dT, gb)
        assert abs(a) < 1e-6, f"constant alpha={ag}: got {a} (defect returned!)"


def test_baseline_constant_zero():
    I, dT, gb = _data(0.08)
    a, _ = R.free_intercept_alpha(np.full_like(I, R.V_READ_DEFAULT * gb), dT, gb)
    assert abs(a) < 1e-6, f"baseline constant: got {a}"


def test_band_robustness():
    """Verdict (perfect ~ alpha_gt) is stable across the masking band [0.25, 1.0]."""
    I, dT, gb = _data(0.08)
    for band in (0.25, 0.5, 1.0):
        a, _ = R.free_intercept_alpha(I, dT, gb, dT_band=band)
        assert abs(a - 0.08) < 2e-3, f"band={band}: got {a}"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    raise SystemExit(fails)
