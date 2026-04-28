"""
Vanilla xLSTM Learning Rate Sweep
==================================
Checks whether poor vanilla xLSTM performance in 15_thermal_baselines_experiment.py
is a hyperparameter artifact (wrong learning rate) rather than a genuine
architectural limitation.

Tests lr in {0.0001, 0.001, 0.01} x seeds {42, 123, 456}, 200 epochs each.

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import sys

# Force UTF-8 output (avoid encoding errors on Windows)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from core_psi_xlstm import mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init  # imported per spec, not used here

# ---------------------------------------------------------------------------
# Constants (identical to 15_thermal_baselines_experiment.py)
# ---------------------------------------------------------------------------
T_AMB = 298.0
G_BASE = 1e-5
ALPHA_GT = 0.08


# ---------------------------------------------------------------------------
# Data generation  (copied verbatim from 15_thermal_baselines_experiment.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model definition  (copied verbatim from 15_thermal_baselines_experiment.py)
# ---------------------------------------------------------------------------

class VanillaxLSTM(nn.Module):
    """Vanilla xLSTM: random init, no BIC, no structure extraction."""
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


# ---------------------------------------------------------------------------
# Training loop  (same as 15_thermal_baselines_experiment.py, 200 epochs)
# ---------------------------------------------------------------------------

def train_model(model, dataset, n_epochs=200, lr=0.001):
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
    criterion_eval = nn.MSELoss()
    with torch.no_grad():
        I_val, _ = model(dataset['val']['V'], dataset['val']['t'])
        val_loss = criterion_eval(I_val, dataset['val']['I']).item()

    return val_loss, train_time


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_lr_sweep():
    print("=" * 70)
    print("Vanilla xLSTM Learning Rate Sweep")
    print("lr in {0.0001, 0.001, 0.01}  x  seeds {42, 123, 456}")
    print("200 epochs  |  identical data splits to 15_thermal_baselines_experiment.py")
    print("=" * 70)

    learning_rates = [0.0001, 0.001, 0.01]
    seeds = [42, 123, 456]
    n_epochs = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    all_results = []

    for lr in learning_rates:
        print(f"--- lr = {lr} ---")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            dataset, T_layer_full = generate_thermal_data(seed=seed)

            model = VanillaxLSTM(hidden_size=32).to(device)
            # Random init only — no physics init (vanilla baseline)

            val_mse, train_time = train_model(model, dataset, n_epochs=n_epochs, lr=lr)

            result = {
                'lr': lr,
                'seed': seed,
                'val_mse': val_mse,
                'train_time_s': round(train_time, 2),
                'n_params': model.count_parameters(),
                'n_epochs': n_epochs,
            }
            all_results.append(result)
            print(f"  seed={seed}  val_mse={val_mse:.4e}  time={train_time:.1f}s")

        # Per-lr summary
        lr_results = [r for r in all_results if r['lr'] == lr]
        mses = [r['val_mse'] for r in lr_results]
        print(f"  => mean val_mse = {np.mean(mses):.4e}  std = {np.std(mses):.2e}\n")

    # Build DataFrame
    df = pd.DataFrame(all_results)
    out_path = 'vanilla_xlstm_lr_sweep.csv'
    df.to_csv(out_path, index=False)
    print(f"[OK] Results saved to: {out_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS TABLE  (val_mse per lr / seed)")
    print("=" * 70)
    print(f"{'lr':<10} {'seed':<8} {'val_mse':<14} {'train_time_s':<14}")
    print("-" * 46)
    for _, row in df.iterrows():
        print(f"  {row['lr']:<8} {int(row['seed']):<8} {row['val_mse']:.4e}    {row['train_time_s']:.1f}s")

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std over 3 seeds per lr)")
    print("=" * 70)
    print(f"{'lr':<12} {'mean_val_mse':<18} {'std_val_mse':<18} {'best_seed_mse':<16}")
    print("-" * 64)
    for lr in learning_rates:
        sub = df[df['lr'] == lr]
        m = sub['val_mse'].mean()
        s = sub['val_mse'].std()
        best = sub['val_mse'].min()
        print(f"  {lr:<10} {m:.4e}         {s:.2e}         {best:.4e}")

    # Reference: baseline experiment used lr=0.001 and 100 epochs
    print("\n  [Ref] 15_thermal_baselines_experiment.py used lr=0.001, 100 epochs")
    print("  [Ref] 200 epochs here gives a fairer comparison at each lr.\n")

    # Identify best lr
    summary = df.groupby('lr')['val_mse'].mean()
    best_lr = summary.idxmin()
    print(f"  Best mean val_mse achieved at lr = {best_lr}  ({summary[best_lr]:.4e})")

    if best_lr == 0.001:
        print("  => lr=0.001 IS the best lr: poor performance is NOT a hyperparameter artifact.")
        print("     The gap vs Psi-Vortex is genuine (architecture / physics-init matters).")
    else:
        print(f"  => lr={best_lr} outperforms lr=0.001: performance WAS partly a hyperparameter artifact.")
        print("     Recommend re-running 15_thermal_baselines_experiment.py with the better lr.")

    return df


if __name__ == "__main__":
    run_lr_sweep()
