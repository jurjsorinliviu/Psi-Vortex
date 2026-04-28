"""
Psi-Vortex Alpha Auto-Selection Experiment
===========================================
Improves on the 10-seed experiment by adding R^2-based automatic seed
selection.  No ground-truth alpha label is used at any point.

Pipeline:
  1. Train N seeds with Psi-Vortex (identical to existing 10-seed experiment).
  2. For each trained model, compute post-hoc OLS and its R^2.
  3. Auto-select the seed with highest R^2 (among seeds with positive alpha).
  4. Report the auto-selected result alongside the full-seed MSE statistics.

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from core_psi_xlstm import PSI_xLSTM, mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init

T_AMB     = 298.0
G_BASE    = 1e-5
ALPHA_GT  = 0.08
V_VICTIM  = 0.2


# =============================================================================
# DATA GENERATION  (unchanged from psi_vortex_10seeds.py)
# =============================================================================

def generate_thermal_data(seed: int = 42):
    np.random.seed(seed)
    n_steps = 3000
    dt = 1e-4
    t = np.linspace(0, n_steps * dt, n_steps)

    V_driver = np.zeros_like(t)
    for _ in range(6):
        start = np.random.randint(200, n_steps - 200)
        V_driver[start:start + 60] = 2.0

    T_layer = np.zeros_like(t)
    T_layer[0] = T_AMB
    Power = V_driver ** 2
    tau_th, heat_coeff = 0.05, 800.0
    for i in range(1, n_steps):
        dT = (-(T_layer[i - 1] - T_AMB) / tau_th + heat_coeff * Power[i - 1]) * dt
        T_layer[i] = T_layer[i - 1] + dT

    V_victim = np.zeros_like(t) + V_VICTIM
    G_victim = G_BASE * np.exp(ALPHA_GT * (T_layer - T_AMB))
    I_victim = V_victim * G_victim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mk = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device).view(-1, 1)

    dataset = {
        'train': {'V': mk(V_driver[:2000]), 't': mk(t[:2000]), 'I': mk(I_victim[:2000])},
        'val':   {'V': mk(V_driver[2000:2500]), 't': mk(t[2000:2500]), 'I': mk(I_victim[2000:2500])},
        'test':  {'V': mk(V_driver[2500:]), 't': mk(t[2500:]), 'I': mk(I_victim[2500:]),
                  'T': mk(T_layer[2500:])},
    }
    return dataset, T_layer


# =============================================================================
# MODEL DEFINITION  (unchanged)
# =============================================================================

class PsiVortexThermal(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super().__init__()
        self.mlstm = mLSTMBlock(input_size, hidden_size, memory_size=hidden_size // 2)
        self.slstm = sLSTMBlock(hidden_size, hidden_size)
        self.fc    = nn.Linear(hidden_size, output_size)

    def forward(self, V, t):
        x = torch.cat([V, t], dim=1).unsqueeze(1)
        h1, _, _ = self.mlstm(x)
        h2, _, _ = self.slstm(h1)
        return self.fc(h2.squeeze(1)), {'fused': h2.squeeze(1)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# OLS WITH R^2  (key improvement)
# =============================================================================

def recover_alpha_ols(model, dataset, T_layer_full):
    """
    Post-hoc OLS: fit log(I_pred / (V_victim * G_base)) = alpha * delta_T.
    Returns (alpha_hat, r2) where r2 is the coefficient of determination of
    the log-linear fit.  No ground-truth alpha is used.
    """
    model.eval()
    with torch.no_grad():
        I_pred, _ = model(dataset['train']['V'], dataset['train']['t'])
    I_pred_np = I_pred.cpu().numpy().flatten()

    delta_T = T_layer_full[:len(I_pred_np)] - T_AMB

    eps = 1e-30
    safe_I   = np.where(I_pred_np > eps, I_pred_np, eps)
    log_ratio = np.log(safe_I / (V_VICTIM * G_BASE))

    # Only use time-steps with meaningful thermal excitation
    mask = np.abs(delta_T) > 0.5
    if mask.sum() < 20:
        mask = np.ones(len(delta_T), dtype=bool)

    dT_m  = delta_T[mask]
    lr_m  = log_ratio[mask]

    # OLS through origin: alpha = (dT . lr) / (dT . dT)
    alpha_hat = float(np.dot(dT_m, lr_m) / (np.dot(dT_m, dT_m) + eps))

    # R^2 for OLS-through-origin: baseline is ŷ=0, not ŷ=mean
    # R² = 1 - SS_res / SS_tot_uncentered  where SS_tot = sum(lr²)
    lr_fitted = alpha_hat * dT_m
    ss_res    = float(np.sum((lr_m - lr_fitted) ** 2))
    ss_tot    = float(np.sum(lr_m ** 2))          # uncentered total SS
    r2        = 1.0 - ss_res / (ss_tot + eps)

    return alpha_hat, r2


# =============================================================================
# TRAINING  (unchanged)
# =============================================================================

def train_model(model, dataset, n_epochs=100, lr=0.001):
    device    = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        for i in range(0, len(dataset['train']['V']), 128):
            j = min(i + 128, len(dataset['train']['V']))
            optimizer.zero_grad()
            I_pred, _ = model(dataset['train']['V'][i:j], dataset['train']['t'][i:j])
            criterion(I_pred, dataset['train']['I'][i:j]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        I_val, _ = model(dataset['val']['V'], dataset['val']['t'])
        val_loss = criterion(I_val, dataset['val']['I']).item()
    return val_loss, time.time() - t0


# =============================================================================
# MAIN
# =============================================================================

def run():
    seeds    = [42, 123, 456, 789, 1000, 1234, 2000, 3000, 4000, 5000]
    n_epochs = 100
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("Psi-Vortex Alpha Auto-Selection (R^2-guided)")
    print(f"GT alpha = {ALPHA_GT}  |  {len(seeds)} seeds  |  device: {device}")
    print("=" * 70)

    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset, T_layer = generate_thermal_data(seed=seed)

        model = PsiVortexThermal(hidden_size=32).to(device)
        apply_psi_vortex_init(model, pde_type="thermal")

        val_mse, t_train = train_model(model, dataset, n_epochs=n_epochs)
        alpha_hat, r2    = recover_alpha_ols(model, dataset, T_layer)
        alpha_err        = abs(alpha_hat - ALPHA_GT) / ALPHA_GT * 100.0

        rows.append(dict(seed=seed, val_mse=val_mse, alpha_recovered=alpha_hat,
                         alpha_error_pct=alpha_err, ols_r2=r2, train_time_s=t_train))

        flag = "✓" if (r2 > 0.85 and alpha_hat > 0) else " "
        print(f"  seed {seed:5d} {flag}  val_mse={val_mse:.3e}  "
              f"alpha={alpha_hat:+.4f}  err={alpha_err:7.2f}%  R²={r2:.4f}")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Auto-selection: highest R^2 among seeds with positive alpha
    # ------------------------------------------------------------------
    candidates = df[df['alpha_recovered'] > 0].copy()
    if candidates.empty:
        candidates = df.copy()          # fallback: use all seeds

    best_row  = candidates.loc[candidates['ols_r2'].idxmax()]
    best_seed = int(best_row['seed'])
    best_r2   = best_row['ols_r2']
    best_alpha = best_row['alpha_recovered']
    best_err  = best_row['alpha_error_pct']

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Val. MSE across all 10 seeds:  "
          f"{df.val_mse.mean():.3e} ± {df.val_mse.std():.3e}")
    print()
    print(f"  Auto-selected seed (highest R²): seed {best_seed}")
    print(f"    OLS R²       = {best_r2:.4f}")
    print(f"    alpha_hat    = {best_alpha:.6f}  (GT = {ALPHA_GT})")
    print(f"    alpha_error  = {best_err:.2f}%")
    print()
    n_good = (candidates['ols_r2'] > 0.85).sum()
    print(f"  Seeds with R² > 0.85 and positive alpha: {n_good}/{len(seeds)}")
    print("=" * 70)

    df.to_csv('alpha_autoselect_results.csv', index=False)
    print("[OK] Per-seed results saved to: alpha_autoselect_results.csv")

    return df, best_row


if __name__ == "__main__":
    run()
