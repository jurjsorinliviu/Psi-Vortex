"""
Psi-Vortex 10-Seed Statistical Rigor Experiment
================================================
Addresses reviewer concern about statistical rigor for the headline result.
Runs Psi-Vortex only with 10 seeds instead of 5.

Seeds: [42, 123, 456, 789, 1000, 1234, 2000, 3000, 4000, 5000]
100 epochs, ALPHA_GT=0.08, thermal init

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from core_psi_xlstm import PSI_xLSTM, mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init


T_AMB = 298.0
G_BASE = 1e-5
ALPHA_GT = 0.08


# =============================================================================
# DATA GENERATION (identical to 15_thermal_baselines_experiment.py)
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

    V_victim = np.zeros_like(t) + 0.2
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
# MODEL DEFINITION
# =============================================================================

class PsiVortexThermal(nn.Module):
    """Psi-Vortex full framework (same as 15_thermal_baselines_experiment.py)."""
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super().__init__()
        self.mlstm = mLSTMBlock(input_size, hidden_size, memory_size=hidden_size // 2)
        self.slstm = sLSTMBlock(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, V, t):
        x = torch.cat([V, t], dim=1).unsqueeze(1)
        h1, h_final1, C_final = self.mlstm(x)
        h2, h_final2, c_final = self.slstm(h1)
        output = self.fc(h2.squeeze(1))
        return output, {'fused': h2.squeeze(1)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# ALPHA RECOVERY (identical to 15_thermal_baselines_experiment.py)
# =============================================================================

def recover_alpha_ols(model, dataset: dict, T_layer_full: np.ndarray):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        I_pred, _ = model(dataset['train']['V'], dataset['train']['t'])

    I_pred_np = I_pred.cpu().numpy().flatten()
    delta_T = T_layer_full[:len(I_pred_np)] - T_AMB

    eps = 1e-30
    safe_I = np.where(I_pred_np > eps, I_pred_np, eps)
    log_ratio = np.log(safe_I / (0.2 * G_BASE))

    mask = np.abs(delta_T) > 0.5
    if mask.sum() < 10:
        mask = np.ones(len(delta_T), dtype=bool)

    dT_m = delta_T[mask]
    lr_m = log_ratio[mask]
    alpha_recovered = np.dot(dT_m, lr_m) / (np.dot(dT_m, dT_m) + eps)
    return float(alpha_recovered)


# =============================================================================
# TRAINING LOOP (identical to 15_thermal_baselines_experiment.py)
# =============================================================================

def train_model(model, dataset, n_epochs=100, lr=0.001):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start = time.time()
    for epoch in range(n_epochs):
        model.train()
        batch_size = 128
        for i in range(0, len(dataset['train']['V']), batch_size):
            end_idx = min(i + batch_size, len(dataset['train']['V']))
            V_b = dataset['train']['V'][i:end_idx]
            t_b = dataset['train']['t'][i:end_idx]
            I_b = dataset['train']['I'][i:end_idx]
            optimizer.zero_grad()
            I_pred, _ = model(V_b, t_b)
            loss = criterion(I_pred, I_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        I_val, _ = model(dataset['val']['V'], dataset['val']['t'])
        val_loss = criterion(I_val, dataset['val']['I']).item()

    return val_loss, train_time


# =============================================================================
# MAIN
# =============================================================================

def run_psi_vortex_10seeds():
    print("=" * 70)
    print("Psi-Vortex: 10-Seed Statistical Rigor Experiment")
    print(f"Ground-truth alpha = {ALPHA_GT}")
    print("Thermal init (apply_psi_vortex_init, pde_type='thermal')")
    print("100 epochs per seed")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 1000, 1234, 2000, 3000, 4000, 5000]
    n_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset, T_layer_full = generate_thermal_data(seed=seed)

        model = PsiVortexThermal(hidden_size=32).to(device)
        apply_psi_vortex_init(model, pde_type="thermal")

        val_loss, train_time = train_model(model, dataset, n_epochs=n_epochs)
        alpha_rec = recover_alpha_ols(model, dataset, T_layer_full)
        alpha_error_pct = abs(alpha_rec - ALPHA_GT) / ALPHA_GT * 100

        results.append({
            'seed': seed,
            'val_mse': val_loss,
            'alpha_recovered': alpha_rec,
            'alpha_error_pct': alpha_error_pct,
            'train_time_s': train_time,
        })

        print(f"Seed {seed:5d}: val_mse={val_loss:.4e}  alpha_recovered={alpha_rec:.6f}  "
              f"alpha_error={alpha_error_pct:.2f}%  time={train_time:.2f}s")

    df = pd.DataFrame(results)

    # Summary statistics
    val_mse_mean = df['val_mse'].mean()
    val_mse_std  = df['val_mse'].std()
    alpha_mean   = df['alpha_recovered'].mean()
    alpha_std    = df['alpha_recovered'].std()
    alpha_err_mean = df['alpha_error_pct'].mean()
    alpha_err_std  = df['alpha_error_pct'].std()

    print("\n" + "=" * 70)
    print("SUMMARY over 10 seeds (mean +/- std)")
    print("=" * 70)
    print(f"  val_mse        : {val_mse_mean:.4e} +/- {val_mse_std:.4e}")
    print(f"  alpha_recovered: {alpha_mean:.6f} +/- {alpha_std:.6f}  (GT={ALPHA_GT})")
    print(f"  alpha_error_pct: {alpha_err_mean:.2f}% +/- {alpha_err_std:.2f}%")
    print("=" * 70)

    # Save CSV
    df.to_csv('psi_vortex_10seeds_results.csv', index=False)
    print("\n[OK] Per-seed results saved to: psi_vortex_10seeds_results.csv")

    # Append a summary row
    summary_df = pd.DataFrame([{
        'seed': 'MEAN',
        'val_mse': val_mse_mean,
        'alpha_recovered': alpha_mean,
        'alpha_error_pct': alpha_err_mean,
        'train_time_s': df['train_time_s'].mean(),
    }, {
        'seed': 'STD',
        'val_mse': val_mse_std,
        'alpha_recovered': alpha_std,
        'alpha_error_pct': alpha_err_std,
        'train_time_s': df['train_time_s'].std(),
    }])
    df_full = pd.concat([df, summary_df], ignore_index=True)
    df_full.to_csv('psi_vortex_10seeds_results.csv', index=False)
    print("[OK] Summary rows (MEAN/STD) appended to CSV.")

    return df


if __name__ == "__main__":
    run_psi_vortex_10seeds()
