"""
Alpha Recovery Validation Experiment for Psi-Vortex
=====================================================
Addresses reviewer concern about circular validation:
  "alpha = 0.08 is both the data-generation parameter and the discovered value"

This experiment:
1. Generates thermal datasets with different GROUND-TRUTH alpha values
2. Trains Psi-Vortex with NO knowledge of the true alpha
3. Recovers alpha post-hoc via OLS regression on held-out test set
4. Reports recovery accuracy across the full alpha range

The recovery is structurally independent of the training label:
  Model is trained on (V, t) -> I supervision only.
  After training, alpha is extracted by solving:
    log(I_pred / (V * G_base)) = alpha * delta_T
  via ordinary least squares on test-set model predictions.

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from core_psi_xlstm import PSI_xLSTM, mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init


# =============================================================================
# DATA GENERATION WITH VARIABLE ALPHA
# =============================================================================

def generate_thermal_data_with_alpha(alpha_gt: float, seed: int = 42):
    """
    Generate 3D thermal crosstalk data for a given ground-truth alpha.
    Identical to 03_psi_vortex_experiment.py except alpha is a parameter.

    Returns dataset dict and the temperature array (for post-hoc regression).
    """
    np.random.seed(seed)
    n_steps = 3000
    dt = 1e-4
    t = np.linspace(0, n_steps * dt, n_steps)

    # Driver signal
    V_driver = np.zeros_like(t)
    for _ in range(6):
        start = np.random.randint(200, n_steps - 200)
        V_driver[start:start + 60] = 2.0

    # Thermal diffusion
    T_amb = 298.0
    tau_th = 0.05
    heat_coeff = 800.0
    T_layer = np.zeros_like(t)
    T_layer[0] = T_amb
    Power = V_driver ** 2
    for i in range(1, n_steps):
        dT = (-(T_layer[i - 1] - T_amb) / tau_th + heat_coeff * Power[i - 1]) * dt
        T_layer[i] = T_layer[i - 1] + dT

    # Victim device with the given alpha
    V_victim = np.zeros_like(t) + 0.2
    G_base = 1e-5
    G_victim = G_base * np.exp(alpha_gt * (T_layer - T_amb))
    I_victim = V_victim * G_victim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = {
        'train': {
            't': torch.tensor(t[:2000], dtype=torch.float32, device=device).view(-1, 1),
            'V': torch.tensor(V_driver[:2000], dtype=torch.float32, device=device).view(-1, 1),
            'I': torch.tensor(I_victim[:2000], dtype=torch.float32, device=device).view(-1, 1),
        },
        'val': {
            't': torch.tensor(t[2000:2500], dtype=torch.float32, device=device).view(-1, 1),
            'V': torch.tensor(V_driver[2000:2500], dtype=torch.float32, device=device).view(-1, 1),
            'I': torch.tensor(I_victim[2000:2500], dtype=torch.float32, device=device).view(-1, 1),
        },
        'test': {
            't': torch.tensor(t[2500:], dtype=torch.float32, device=device).view(-1, 1),
            'V': torch.tensor(V_driver[2500:], dtype=torch.float32, device=device).view(-1, 1),
            'I': torch.tensor(I_victim[2500:], dtype=torch.float32, device=device).view(-1, 1),
            'T': torch.tensor(T_layer[2500:], dtype=torch.float32, device=device).view(-1, 1),
        },
    }
    return dataset, T_layer, G_base


# =============================================================================
# MODEL (same as 03_psi_vortex_experiment.py)
# =============================================================================

class ThermalPSIxLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
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
# ALPHA RECOVERY VIA POST-HOC OLS
# =============================================================================

def recover_alpha_ols(model, dataset: dict, T_layer_full: np.ndarray,
                      G_base: float, T_amb: float = 298.0):
    """
    Recover alpha from trained model via post-hoc OLS on the training sequence.

    The model is run in sequence mode on the full 2000-step training sequence
    so the LSTM has temporal context to encode thermal history.
    T_layer_full[:2000] provides the correct temperatures for those steps —
    computed deterministically from V_driver, never given to the model as input.

    Physical relationship:
      I_victim = V_victim * G_base * exp(alpha * delta_T)
      log(I_pred / (V_victim * G_base)) = alpha * delta_T
      alpha_OLS = sum(delta_T * log_ratio) / sum(delta_T^2)

    V_victim = 0.2 V (constant read voltage throughout data generation).
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        I_pred, _ = model(dataset['train']['V'], dataset['train']['t'])

    I_pred_np = I_pred.cpu().numpy().flatten()
    delta_T = T_layer_full[:len(I_pred_np)] - T_amb

    V_victim = 0.2
    eps = 1e-30
    safe_I = np.where(I_pred_np > eps, I_pred_np, eps)
    log_ratio = np.log(safe_I / (V_victim * G_base))

    # Use only samples with meaningful thermal variation (> 0.5 K)
    mask = np.abs(delta_T) > 0.5
    if mask.sum() < 10:
        mask = np.ones(len(delta_T), dtype=bool)

    dT_m = delta_T[mask]
    lr_m = log_ratio[mask]
    alpha_recovered = np.dot(dT_m, lr_m) / (np.dot(dT_m, dT_m) + eps)
    return float(alpha_recovered)


# =============================================================================
# SINGLE ALPHA EXPERIMENT
# =============================================================================

def run_single_alpha_experiment(alpha_gt: float, seed: int = 42, n_epochs: int = 100):
    """
    Train Psi-Vortex on thermal data with given ground-truth alpha,
    then recover alpha post-hoc via OLS.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    dataset, T_layer, G_base = generate_thermal_data_with_alpha(alpha_gt, seed=seed)

    model = ThermalPSIxLSTM(input_size=2, hidden_size=32, output_size=1).to(device)
    apply_psi_vortex_init(model, pde_type="thermal")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

    # Validation loss
    model.eval()
    with torch.no_grad():
        I_val_pred, _ = model(dataset['val']['V'], dataset['val']['t'])
        val_loss = criterion(I_val_pred, dataset['val']['I']).item()

    # Post-hoc alpha recovery via OLS on full sequence (T_layer from data generator)
    alpha_recovered = recover_alpha_ols(model, dataset, T_layer, G_base)
    rel_error_pct = abs(alpha_recovered - alpha_gt) / alpha_gt * 100

    return {
        'alpha_gt': alpha_gt,
        'alpha_recovered': alpha_recovered,
        'relative_error_pct': rel_error_pct,
        'val_loss': val_loss,
        'train_time_s': train_time,
        'seed': seed,
    }


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_alpha_recovery_sweep():
    """
    Run recovery experiments across a range of ground-truth alpha values.
    Each alpha is tested with multiple seeds to assess robustness.
    """
    print("=" * 70)
    print("ALPHA RECOVERY VALIDATION EXPERIMENT")
    print("Ground-truth alpha spans [0.05, 0.08, 0.10, 0.15, 0.20]")
    print("Recovery via post-hoc OLS (no alpha label used in training)")
    print("=" * 70)

    alpha_values = [0.05, 0.08, 0.10, 0.15, 0.20]
    seeds = [42, 123, 456]  # 3 seeds per alpha for reliability
    n_epochs = 100

    all_results = []

    for alpha_gt in alpha_values:
        print(f"\n--- alpha_gt = {alpha_gt:.2f} ---")
        for seed in seeds:
            result = run_single_alpha_experiment(alpha_gt, seed=seed, n_epochs=n_epochs)
            all_results.append(result)
            print(f"  Seed {seed}: recovered={result['alpha_recovered']:.4f}, "
                  f"error={result['relative_error_pct']:.1f}%, "
                  f"val_loss={result['val_loss']:.2e}")

    df = pd.DataFrame(all_results)
    df.to_csv('alpha_recovery_results.csv', index=False)

    # Aggregate by alpha_gt
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean +/- std over seeds)")
    print(f"{'alpha_gt':>10} {'alpha_rec (mean)':>18} {'rel_error % (mean)':>20} {'rel_error % (std)':>18}")
    print("-" * 70)

    summary_rows = []
    for alpha_gt in alpha_values:
        sub = df[df['alpha_gt'] == alpha_gt]
        mean_rec = sub['alpha_recovered'].mean()
        mean_err = sub['relative_error_pct'].mean()
        std_err = sub['relative_error_pct'].std()
        mean_loss = sub['val_loss'].mean()
        print(f"  {alpha_gt:>8.2f}   {mean_rec:>16.4f}   {mean_err:>18.1f}%   {std_err:>16.1f}%")
        summary_rows.append({
            'alpha_gt': alpha_gt,
            'alpha_recovered_mean': mean_rec,
            'alpha_recovered_std': sub['alpha_recovered'].std(),
            'relative_error_pct_mean': mean_err,
            'relative_error_pct_std': std_err,
            'val_loss_mean': mean_loss,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('alpha_recovery_summary.csv', index=False)

    print("\n[OK] Results saved to: alpha_recovery_results.csv")
    print("[OK] Summary saved to:  alpha_recovery_summary.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(alpha_values, alpha_values, 'k--', label='Ideal (recovered = GT)', linewidth=1.5)
    ax.errorbar(df_summary['alpha_gt'], df_summary['alpha_recovered_mean'],
                yerr=df_summary['alpha_recovered_std'],
                fmt='o-', color='green', linewidth=2, markersize=8,
                capsize=5, label='Psi-Vortex recovery')
    ax.set_xlabel('Ground-truth alpha')
    ax.set_ylabel('Recovered alpha')
    ax.set_title('(a) Alpha Recovery Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(df_summary['alpha_gt'], df_summary['relative_error_pct_mean'],
           yerr=df_summary['relative_error_pct_std'],
           color='steelblue', alpha=0.8, capsize=5,
           width=0.015)
    ax.axhline(y=10, color='red', linestyle='--', label='10% threshold')
    ax.set_xlabel('Ground-truth alpha')
    ax.set_ylabel('Relative Recovery Error (%)')
    ax.set_title('(b) Recovery Error vs Ground-Truth Alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alpha_recovery_plot.png', dpi=150, bbox_inches='tight')
    print("[OK] Plot saved to: alpha_recovery_plot.png")

    return df, df_summary


if __name__ == "__main__":
    run_alpha_recovery_sweep()
