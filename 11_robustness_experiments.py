"""
Robustness Experiments for Ψ-Vortex using PSI-xLSTM

This file addresses Reviewer Concerns:
- #2: Statistical rigor (multiple seeds, confidence intervals)
- #5: Hyperparameter sensitivity and search space dependence

Contains:
1. Noise sweep experiments using actual memristor data
2. Multi-seed runs with 95% confidence intervals
3. Convergence comparison: Ψ-Vortex init vs random init

Uses actual PSI-xLSTM architecture from core_psi_xlstm.py
Compares Ψ-Vortex physics-aware initialization vs random initialization
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import time
import pandas as pd

# Import PSI-xLSTM from core module
from core_psi_xlstm import PSI_xLSTM, apply_physics_init_xlstm


# =============================================================================
# SECTION 1: LOAD ACTUAL MEMRISTOR DATA
# =============================================================================

def load_memristor_data(snr_db: float = None, seed: int = 42):
    """
    Load actual printed memristor training data and optionally add noise.
    
    Returns: V, state, I tensors on appropriate device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load actual data
    df = pd.read_csv('printed_memristor_training_data.csv')
    
    # Use correct column names: voltage, state, current
    V = torch.tensor(df['voltage'].values, dtype=torch.float32).view(-1, 1)
    state = torch.tensor(df['state'].values, dtype=torch.float32).view(-1, 1)
    I = torch.tensor(df['current'].values, dtype=torch.float32).view(-1, 1)
    
    # Add noise if specified
    if snr_db is not None:
        torch.manual_seed(seed)
        signal_power = (I ** 2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(I) * noise_std
        I_noisy = I + noise
    else:
        I_noisy = I.clone()
    
    return V.to(device), state.to(device), I_noisy.to(device), I.to(device)


# =============================================================================
# SECTION 2: TRAIN FUNCTION FOR PSI-xLSTM
# =============================================================================

def train_psi_xlstm(V_train, t_train, I_train, V_val, t_val, I_val,
                    use_physics_init=True, n_epochs=200, seed=42, target_loss=1e-6):
    """
    Train PSI-xLSTM model and return convergence metrics.
    
    Args:
        use_physics_init: If True, use Ψ-Vortex initialization; else random
        target_loss: Loss threshold to consider converged
        
    Returns:
        final_loss, convergence_epoch, training_time, loss_history
    """
    device = V_train.device
    torch.manual_seed(seed)
    
    # Create model
    model = PSI_xLSTM(input_size=2, hidden_size=32, num_blocks=4, output_size=1).to(device)
    
    if use_physics_init:
        apply_physics_init_xlstm(model, pde_type="memristor")
    # else: use default PyTorch initialization
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = []
    best_val_loss = float('inf')
    convergence_epoch = n_epochs  # Default: didn't converge
    
    start_time = time.perf_counter()
    
    for epoch in range(n_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        pred, _ = model(V_train, t_train)
        train_loss = nn.functional.mse_loss(pred, I_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(V_val, t_val)
            val_loss = nn.functional.mse_loss(val_pred, I_val).item()
        
        loss_history.append(val_loss)
        
        # Check convergence (first time reaching target)
        if val_loss < target_loss and convergence_epoch == n_epochs:
            convergence_epoch = epoch + 1
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    training_time = time.perf_counter() - start_time
    
    return best_val_loss, convergence_epoch, training_time, loss_history


# =============================================================================
# SECTION 3: NOISE SWEEP EXPERIMENT
# =============================================================================

def run_noise_sweep_experiment(
    snr_levels: List[float] = [20, 30, 40, 50, 60],
    n_seeds: int = 5,
    n_epochs: int = 500
) -> Dict:
    """
    Compare Ψ-Vortex init vs random init across different noise levels.
    Measures both final MSE and convergence speed (epochs to target).
    """
    print("=" * 70)
    print("NOISE SWEEP EXPERIMENT")
    print("Comparing: Ψ-Vortex Init vs Random Init")
    print(f"Testing SNR levels: {snr_levels} dB")
    print(f"Seeds: {n_seeds}, Epochs: {n_epochs}")
    print("=" * 70)
    
    # Use a target loss that both can reach to measure convergence
    target_loss = 5e-7  # Between vortex and random final losses
    
    results = {
        'snr_levels': snr_levels,
        'vortex_mse': {snr: [] for snr in snr_levels},
        'random_mse': {snr: [] for snr in snr_levels},
        'vortex_epochs': {snr: [] for snr in snr_levels},
        'random_epochs': {snr: [] for snr in snr_levels},
        'vortex_time': {snr: [] for snr in snr_levels},
        'random_time': {snr: [] for snr in snr_levels}
    }
    
    for snr in snr_levels:
        print(f"\n--- SNR = {snr} dB ---")
        
        for seed in range(n_seeds):
            # Load data with noise
            V, state, I_noisy, I_clean = load_memristor_data(snr_db=snr, seed=seed)
            
            # Split data
            n_samples = len(V)
            n_train = int(0.8 * n_samples)
            V_train, state_train, I_train = V[:n_train], state[:n_train], I_noisy[:n_train]
            V_val, state_val, I_val = V[n_train:], state[n_train:], I_clean[n_train:]
            
            # Train with Ψ-Vortex init
            vortex_loss, vortex_epoch, vortex_time, _ = train_psi_xlstm(
                V_train, state_train, I_train, V_val, state_val, I_val,
                use_physics_init=True, n_epochs=n_epochs, seed=seed, target_loss=target_loss
            )
            results['vortex_mse'][snr].append(vortex_loss)
            results['vortex_epochs'][snr].append(vortex_epoch)
            results['vortex_time'][snr].append(vortex_time)
            
            # Train with random init
            random_loss, random_epoch, random_time, _ = train_psi_xlstm(
                V_train, state_train, I_train, V_val, state_val, I_val,
                use_physics_init=False, n_epochs=n_epochs, seed=seed, target_loss=target_loss
            )
            results['random_mse'][snr].append(random_loss)
            results['random_epochs'][snr].append(random_epoch)
            results['random_time'][snr].append(random_time)
            
            # Convergence speedup
            conv_speedup = random_epoch / vortex_epoch if vortex_epoch > 0 else float('inf')
            print(f"  Seed {seed}: Vortex MSE={vortex_loss:.2e} (conv@{vortex_epoch}), "
                  f"Random MSE={random_loss:.2e} (conv@{random_epoch}), Conv speedup={conv_speedup:.2f}×")
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("NOISE SWEEP RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 70)
    
    summary = []
    for snr in snr_levels:
        vortex_mses = results['vortex_mse'][snr]
        random_mses = results['random_mse'][snr]
        vortex_epochs = results['vortex_epochs'][snr]
        random_epochs = results['random_epochs'][snr]
        
        v_mean = np.mean(vortex_mses)
        v_std = np.std(vortex_mses, ddof=1)
        v_ci = stats.t.interval(0.95, len(vortex_mses)-1, loc=v_mean,
                                scale=v_std/np.sqrt(len(vortex_mses))) if v_std > 0 else (v_mean, v_mean)
        
        r_mean = np.mean(random_mses)
        mse_improvement = (r_mean - v_mean) / r_mean * 100 if r_mean > 0 else 0
        
        # Convergence speedup (epochs to reach target)
        v_epochs_mean = np.mean(vortex_epochs)
        r_epochs_mean = np.mean(random_epochs)
        conv_speedup = r_epochs_mean / v_epochs_mean if v_epochs_mean > 0 else 0
        
        summary.append({
            'snr': snr,
            'vortex_mean': v_mean,
            'vortex_ci_low': v_ci[0],
            'vortex_ci_high': v_ci[1],
            'random_mean': r_mean,
            'mse_improvement': mse_improvement,
            'vortex_epochs': v_epochs_mean,
            'random_epochs': r_epochs_mean,
            'conv_speedup': conv_speedup
        })
        
        print(f"SNR {snr:3d} dB: Vortex = {v_mean:.2e} [{v_ci[0]:.2e}, {v_ci[1]:.2e}], "
              f"MSE improvement = {mse_improvement:+.1f}%, Conv speedup = {conv_speedup:.2f}×")
    
    results['summary'] = summary

    # Save noise sweep results to CSV
    df_noise = pd.DataFrame([
        {
            'snr_db': s['snr'],
            'vortex_mse_mean': s['vortex_mean'],
            'vortex_ci_low': s['vortex_ci_low'],
            'vortex_ci_high': s['vortex_ci_high'],
            'random_mse_mean': s['random_mean'],
            'mse_improvement_pct': s['mse_improvement'],
            'vortex_epochs_mean': s['vortex_epochs'],
            'random_epochs_mean': s['random_epochs'],
            'conv_speedup': s['conv_speedup'],
        }
        for s in summary
    ])
    df_noise.to_csv('noise_sweep_results.csv', index=False)
    print("\n📁 Results saved to: noise_sweep_results.csv")

    return results


# =============================================================================
# SECTION 4: MULTI-SEED STATISTICAL SIGNIFICANCE
# =============================================================================

def run_statistical_significance_experiment(n_seeds: int = 20, n_epochs: int = 500) -> Dict:
    """
    Run many seeds to establish statistical significance of Ψ-Vortex init.
    Measures both MSE improvement and convergence speedup.
    """
    print("\n" + "=" * 70)
    print(f"STATISTICAL SIGNIFICANCE EXPERIMENT ({n_seeds} seeds)")
    print("Comparing: Ψ-Vortex Init vs Random Init")
    print("=" * 70)
    
    target_loss = 5e-7
    
    vortex_mses = []
    random_mses = []
    vortex_times = []
    random_times = []
    vortex_epochs = []
    random_epochs = []
    
    for seed in range(n_seeds):
        # Load clean data (no noise)
        V, state, I, _ = load_memristor_data(snr_db=None, seed=seed)
        
        # Split data
        n_samples = len(V)
        n_train = int(0.8 * n_samples)
        V_train, state_train, I_train = V[:n_train], state[:n_train], I[:n_train]
        V_val, state_val, I_val = V[n_train:], state[n_train:], I[n_train:]
        
        # Train with Ψ-Vortex init
        v_loss, v_epoch, v_time, _ = train_psi_xlstm(
            V_train, state_train, I_train, V_val, state_val, I_val,
            use_physics_init=True, n_epochs=n_epochs, seed=seed, target_loss=target_loss
        )
        vortex_mses.append(v_loss)
        vortex_times.append(v_time)
        vortex_epochs.append(v_epoch)
        
        # Train with random init
        r_loss, r_epoch, r_time, _ = train_psi_xlstm(
            V_train, state_train, I_train, V_val, state_val, I_val,
            use_physics_init=False, n_epochs=n_epochs, seed=seed, target_loss=target_loss
        )
        random_mses.append(r_loss)
        random_times.append(r_time)
        random_epochs.append(r_epoch)
        
        if (seed + 1) % 5 == 0:
            print(f"  Completed {seed + 1}/{n_seeds} seeds...")
    
    # Statistical tests
    v_mean = np.mean(vortex_mses)
    v_std = np.std(vortex_mses, ddof=1)
    r_mean = np.mean(random_mses)
    r_std = np.std(random_mses, ddof=1)
    
    # Paired t-test on MSE
    t_stat_mse, p_value_mse = stats.ttest_rel(vortex_mses, random_mses)
    
    # Convergence speedup statistics
    v_epochs_mean = np.mean(vortex_epochs)
    r_epochs_mean = np.mean(random_epochs)
    conv_speedup = r_epochs_mean / v_epochs_mean if v_epochs_mean > 0 else 0
    
    # Paired t-test on convergence epochs
    t_stat_epochs, p_value_epochs = stats.ttest_rel(random_epochs, vortex_epochs)
    
    # 95% CI for Ψ-Vortex
    ci = stats.t.interval(0.95, n_seeds-1, loc=v_mean, scale=v_std/np.sqrt(n_seeds)) if v_std > 0 else (v_mean, v_mean)
    
    # Effect size (Cohen's d) for MSE
    pooled_std = np.sqrt((v_std**2 + r_std**2) / 2) if (v_std + r_std) > 0 else 1
    cohens_d = (r_mean - v_mean) / pooled_std
    
    print("\n" + "=" * 70)
    print("STATISTICAL RESULTS")
    print("=" * 70)
    print(f"Ψ-Vortex Init MSE: {v_mean:.2e} ± {v_std:.2e}")
    print(f"Random Init MSE:   {r_mean:.2e} ± {r_std:.2e}")
    print(f"95% CI for Ψ-Vortex: [{ci[0]:.2e}, {ci[1]:.2e}]")
    print(f"MSE Improvement: {(r_mean - v_mean) / r_mean * 100:.1f}%")
    print(f"Paired t-test (MSE): t = {t_stat_mse:.3f}, p = {p_value_mse:.2e}")
    print(f"Cohen's d (effect size): {cohens_d:.2f}")
    print(f"\nConvergence (epochs to target loss {target_loss:.0e}):")
    print(f"  Ψ-Vortex: {v_epochs_mean:.1f} ± {np.std(vortex_epochs):.1f} epochs")
    print(f"  Random:   {r_epochs_mean:.1f} ± {np.std(random_epochs):.1f} epochs")
    print(f"  Convergence Speedup: {conv_speedup:.2f}×")
    print(f"  Paired t-test (Epochs): t = {t_stat_epochs:.3f}, p = {p_value_epochs:.2e}")
    
    if p_value_mse < 0.05:
        print("\n✅ MSE difference is statistically significant (p < 0.05)")
    if p_value_epochs < 0.05:
        print("✅ Convergence speedup is statistically significant (p < 0.05)")
    if abs(cohens_d) > 0.8:
        print("✅ Large effect size (|Cohen's d| > 0.8)")

    # Save per-seed raw results
    df_seeds = pd.DataFrame({
        'seed': list(range(n_seeds)),
        'vortex_mse': vortex_mses,
        'random_mse': random_mses,
        'vortex_epochs': vortex_epochs,
        'random_epochs': random_epochs,
        'vortex_time': vortex_times,
        'random_time': random_times,
    })
    df_seeds.to_csv('statistical_significance_per_seed.csv', index=False)

    # Save aggregate summary
    mse_reduction_pct = (r_mean - v_mean) / r_mean * 100 if r_mean > 0 else 0
    df_summary = pd.DataFrame([{
        'n_seeds': n_seeds,
        'vortex_mse_mean': v_mean,
        'vortex_mse_std': v_std,
        'random_mse_mean': r_mean,
        'random_mse_std': r_std,
        'mse_reduction_pct': mse_reduction_pct,
        'vortex_ci_low': ci[0],
        'vortex_ci_high': ci[1],
        'p_value_mse': p_value_mse,
        'p_value_epochs': p_value_epochs,
        'cohens_d': cohens_d,
        'vortex_epochs_mean': v_epochs_mean,
        'random_epochs_mean': r_epochs_mean,
        'conv_speedup': conv_speedup,
    }])
    df_summary.to_csv('statistical_significance_summary.csv', index=False)
    print("\n📁 Results saved to:")
    print("   - statistical_significance_per_seed.csv")
    print("   - statistical_significance_summary.csv")

    return {
        'vortex_mses': vortex_mses,
        'random_mses': random_mses,
        'vortex_epochs': vortex_epochs,
        'random_epochs': random_epochs,
        'vortex_mean': v_mean,
        'random_mean': r_mean,
        'vortex_std': v_std,
        'vortex_ci': ci,
        'p_value_mse': p_value_mse,
        'p_value_epochs': p_value_epochs,
        'cohens_d': cohens_d,
        'conv_speedup': conv_speedup
    }


# =============================================================================
# SECTION 5: CONVERGENCE COMPARISON
# =============================================================================

def run_convergence_comparison(n_epochs: int = 500, seed: int = 42) -> Dict:
    """
    Compare convergence curves: Ψ-Vortex init vs random init.
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE COMPARISON")
    print(f"Epochs: {n_epochs}")
    print("=" * 70)
    
    target_loss = 5e-7
    
    # Load clean data
    V, state, I, _ = load_memristor_data(snr_db=None, seed=seed)
    
    n_samples = len(V)
    n_train = int(0.8 * n_samples)
    V_train, state_train, I_train = V[:n_train], state[:n_train], I[:n_train]
    V_val, state_val, I_val = V[n_train:], state[n_train:], I[n_train:]
    
    # Train with Ψ-Vortex init
    print("Training with Ψ-Vortex initialization...")
    _, vortex_epoch, vortex_time, vortex_history = train_psi_xlstm(
        V_train, state_train, I_train, V_val, state_val, I_val,
        use_physics_init=True, n_epochs=n_epochs, seed=seed, target_loss=target_loss
    )
    
    # Train with random init
    print("Training with random initialization...")
    _, random_epoch, random_time, random_history = train_psi_xlstm(
        V_train, state_train, I_train, V_val, state_val, I_val,
        use_physics_init=False, n_epochs=n_epochs, seed=seed, target_loss=target_loss
    )
    
    conv_speedup = random_epoch / vortex_epoch if vortex_epoch > 0 else float('inf')
    
    print(f"\nΨ-Vortex: Final loss = {vortex_history[-1]:.2e}, Converged @ epoch {vortex_epoch}")
    print(f"Random:   Final loss = {random_history[-1]:.2e}, Converged @ epoch {random_epoch}")
    print(f"Convergence Speedup: {conv_speedup:.2f}×")
    
    return {
        'vortex_history': vortex_history,
        'random_history': random_history,
        'vortex_epoch': vortex_epoch,
        'random_epoch': random_epoch,
        'conv_speedup': conv_speedup,
        'n_epochs': n_epochs
    }


# =============================================================================
# SECTION 6: GENERATE PLOTS
# =============================================================================

def generate_robustness_plots(noise_results: Dict, stats_results: Dict, convergence_results: Dict):
    """Generate publication-quality plots for robustness experiments."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Noise sweep - MSE comparison
    ax = axes[0, 0]
    snr_levels = noise_results['snr_levels']
    summary = noise_results['summary']
    
    vortex_means = [s['vortex_mean'] for s in summary]
    vortex_lows = [s['vortex_ci_low'] for s in summary]
    vortex_highs = [s['vortex_ci_high'] for s in summary]
    random_means = [s['random_mean'] for s in summary]
    
    ax.semilogy(snr_levels, vortex_means, 'o-', label=r'$\Psi$-Vortex Init', linewidth=2, markersize=8, color='green')
    ax.fill_between(snr_levels, vortex_lows, vortex_highs, alpha=0.3, color='green')
    ax.semilogy(snr_levels, random_means, 's--', label='Random Init', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Validation MSE (log scale)')
    ax.set_title('(a) MSE vs Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Convergence speedup by SNR
    ax = axes[0, 1]
    speedups = [s['conv_speedup'] for s in summary]
    ax.bar(snr_levels, speedups, color='blue', alpha=0.7)
    ax.axhline(y=1, color='red', linestyle='--', label='No speedup')
    ax.axhline(y=3.63, color='green', linestyle='--', label='Paper claim (3.63×)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Convergence Speedup (×)')
    ax.set_title('(b) Convergence Speedup (epochs to target)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Convergence epochs distribution
    ax = axes[1, 0]
    ax.hist(stats_results['vortex_epochs'], bins=15, alpha=0.7, label=r'$\Psi$-Vortex Init', color='green')
    ax.hist(stats_results['random_epochs'], bins=15, alpha=0.7, label='Random Init', color='red')
    ax.axvline(np.mean(stats_results['vortex_epochs']), color='green', linestyle='--', linewidth=2)
    ax.axvline(np.mean(stats_results['random_epochs']), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Epochs to Convergence')
    ax.set_ylabel('Frequency')
    ax.set_title(f'(c) Convergence Distribution (n={len(stats_results["vortex_epochs"])} seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence curves
    ax = axes[1, 1]
    epochs = range(1, convergence_results['n_epochs'] + 1)
    ax.semilogy(epochs, convergence_results['vortex_history'], '-', label=r'$\Psi$-Vortex Init', linewidth=2, color='green')
    ax.semilogy(epochs, convergence_results['random_history'], '--', label='Random Init', linewidth=2, color='red')
    
    # Mark convergence points
    if convergence_results['vortex_epoch'] < convergence_results['n_epochs']:
        ax.axvline(convergence_results['vortex_epoch'], color='green', linestyle=':', alpha=0.7)
        ax.scatter([convergence_results['vortex_epoch']],
                   [convergence_results['vortex_history'][convergence_results['vortex_epoch']-1]],
                   color='green', s=100, zorder=5, marker='*')
    if convergence_results['random_epoch'] < convergence_results['n_epochs']:
        ax.axvline(convergence_results['random_epoch'], color='red', linestyle=':', alpha=0.7)
        ax.scatter([convergence_results['random_epoch']],
                   [convergence_results['random_history'][convergence_results['random_epoch']-1]],
                   color='red', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE (log scale)')
    ax.set_title(f'(d) Convergence Curves (speedup: {convergence_results["conv_speedup"]:.2f}×)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robustness_experiments.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved: robustness_experiments.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Ψ-VORTEX ROBUSTNESS EXPERIMENTS (PSI-xLSTM)")
    print("Comparing: Ψ-Vortex Physics-Aware Init vs Random Init")
    print("Addressing Reviewer Concerns #2 and #5")
    print("=" * 70)
    
    # Run experiments
    noise_results = run_noise_sweep_experiment(
        snr_levels=[20, 30, 40, 50, 60],
        n_seeds=5,
        n_epochs=500
    )
    
    stats_results = run_statistical_significance_experiment(n_seeds=20, n_epochs=500)
    
    convergence_results = run_convergence_comparison(n_epochs=500, seed=42)
    
    # Generate plots
    generate_robustness_plots(noise_results, stats_results, convergence_results)
    
    print("\n" + "=" * 70)
    print("ALL ROBUSTNESS EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("\nKey findings for reviewer response:")
    print(f"  1. Ψ-Vortex init vs random init tested across SNR 20-60 dB")
    print(f"  2. MSE Statistical significance: p = {stats_results['p_value_mse']:.2e}")
    print(f"  3. Convergence speedup significance: p = {stats_results['p_value_epochs']:.2e}")
    print(f"  4. Effect size (Cohen's d): {stats_results['cohens_d']:.2f}")
    print(f"  5. Convergence speedup: {stats_results['conv_speedup']:.2f}×")
    print(f"  6. 95% CI for Ψ-Vortex MSE: [{stats_results['vortex_ci'][0]:.2e}, {stats_results['vortex_ci'][1]:.2e}]")