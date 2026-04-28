"""
Ψ-Vortex Experiment 9: Extended Validation Experiments
=======================================================
Additional experiments to strengthen the paper's claims:

1. λ_BIC and γ Sensitivity Analysis
2. Model Size Scalability (32, 64, 128 hidden)
3. Ψ-Family Lineage Comparison (HDL → xLSTM → Vortex)
4. Extended Ablation (2×2×2 grid: Init × BIC × Symmetry)

Author: Sorin Liviu Jurj
Date: December 2025
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Tuple
from itertools import product

# Import modules
from core_psi_xlstm import mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init
from core_auto_symmetry import apply_auto_vortex_init, apply_identity_vortex_init
from core_adaptive_bic import DifferentiableBIC, AdaptiveStructureLoss
from core_rrad_loss import RecurrentRelationAwareDistillation


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class PSI_xLSTM_Teacher(nn.Module):
    """PSI-xLSTM Teacher model with configurable size"""
    def __init__(self, input_size=2, hidden_size=64, output_size=1, memory_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size or hidden_size // 2
        self.mlstm = mLSTMBlock(input_size, hidden_size, memory_size=self.memory_size)
        self.slstm = sLSTMBlock(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, V, t):
        x = torch.cat([V, t], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h1, h_final1, C_final = self.mlstm(x)
        h2, h_final2, c_final = self.slstm(h1)
        output = self.fc(h2.squeeze(1))
        hidden_states = {
            'fused': h2.squeeze(1),
            'block_hiddens': [h_final1, h_final2],
            'block_memories': [C_final, c_final]
        }
        return output, hidden_states
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CompactStudent(nn.Module):
    """Compact student model"""
    def __init__(self, input_size=2, hidden_size=16, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, V, t):
        x = torch.cat([V, t], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out.squeeze(1))
        hidden_states = {
            'fused': lstm_out.squeeze(1),
            'block_hiddens': [h_n.squeeze(0)],
            'block_memories': [c_n.squeeze(0)]
        }
        return output, hidden_states
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Simple MLP for PSI-HDL baseline
class SimpleMLP(nn.Module):
    """Simple MLP for PSI-HDL comparison"""
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, V, t):
        x = torch.cat([V, t], dim=-1)
        output = self.net(x)
        return output, {'fused': x}
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def apply_random_init(model):
    """Standard random initialization"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


# ============================================================
# DATA LOADING
# ============================================================

def load_memristor_data():
    """Load memristor data"""
    DATA_PATH = 'printed_memristor_training_data.csv'
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = df[(df['device_id'] == 0) & (df['cycle_id'] == 0)]
        
        V = torch.tensor(df['voltage'].values, dtype=torch.float32).view(-1, 1)
        I = torch.tensor(df['current'].values, dtype=torch.float32).view(-1, 1)
        t = torch.linspace(0, 1, len(V)).view(-1, 1)
        return V, t, I
    else:
        # Synthetic
        t = torch.linspace(0, 0.01, 1000)
        freq = 150e3
        V = 2.0 * torch.sin(2 * np.pi * freq * t)
        I = 1e-4 * torch.sinh(V) * (1 + 0.3 * torch.cos(4 * np.pi * freq * t))
        return V.view(-1, 1), t.view(-1, 1), I.view(-1, 1)


def generate_synthetic_odd_data(n_samples=1000):
    """Generate synthetic odd symmetry data"""
    t = torch.linspace(0, 0.01, n_samples)
    freq = 150e3
    V = 2.0 * torch.sin(2 * np.pi * freq * t)
    I = 1e-4 * torch.sinh(V)
    return V.view(-1, 1), t.view(-1, 1), I.view(-1, 1)


# ============================================================
# TRAINING UTILITIES
# ============================================================

def train_model(model, V, t, I_target, target_mse=1e-6, max_epochs=500, 
                use_bic=False, lambda_bic=0.01, gamma=0.1, device='cuda'):
    """Train a model and return metrics"""
    model = model.to(device)
    V, t, I_target = V.to(device), t.to(device), I_target.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    if use_bic:
        bic_calculator = DifferentiableBIC(gamma=gamma)
    
    start_time = time.time()
    converged_epoch = max_epochs
    final_loss = float('inf')
    loss_history = []
    bic_history = []
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred, _ = model(V, t)
        mse_loss = loss_fn(pred, I_target)
        
        if use_bic:
            bic_loss = bic_calculator(model, mse_loss, len(V))
            total_loss = mse_loss + lambda_bic * bic_loss
            bic_history.append(bic_loss.item())
        else:
            total_loss = mse_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(mse_loss.item())
        
        if mse_loss.item() < target_mse:
            converged_epoch = epoch + 1
            final_loss = mse_loss.item()
            break
        final_loss = mse_loss.item()
    
    duration = time.time() - start_time
    
    return {
        'epochs': converged_epoch,
        'time': duration,
        'final_loss': final_loss,
        'loss_history': loss_history,
        'bic_history': bic_history if use_bic else None,
        'parameters': model.count_parameters()
    }


# ============================================================
# EXPERIMENT 1: λ_BIC and γ Sensitivity Analysis
# ============================================================

def experiment_1_bic_sensitivity():
    """Test sensitivity to λ_BIC and γ hyperparameters"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: λ_BIC AND γ SENSITIVITY ANALYSIS")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V, t, I = generate_synthetic_odd_data()
    
    # λ_BIC values to test
    lambda_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # γ values to test
    gamma_values = [0.05, 0.1, 0.2, 0.5]
    
    results = []
    
    for lambda_bic in lambda_values:
        for gamma in gamma_values:
            print(f"\nTesting λ_BIC={lambda_bic}, γ={gamma}")
            
            torch.manual_seed(42)
            model = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
            apply_psi_vortex_init(model, pde_type="memristor")
            
            metrics = train_model(
                model, V, t, I,
                target_mse=1e-6,
                max_epochs=200,
                use_bic=True,
                lambda_bic=lambda_bic,
                gamma=gamma,
                device=device
            )
            
            results.append({
                'lambda_BIC': lambda_bic,
                'gamma': gamma,
                'epochs': metrics['epochs'],
                'time': metrics['time'],
                'final_loss': metrics['final_loss'],
                'final_bic': metrics['bic_history'][-1] if metrics['bic_history'] else None
            })
            
            print(f"  Epochs: {metrics['epochs']}, Loss: {metrics['final_loss']:.2e}")
    
    # Create summary table
    df_results = pd.DataFrame(results)
    df_results.to_csv('bic_sensitivity_results.csv', index=False)
    print("\n📊 Results saved to 'bic_sensitivity_results.csv'")
    
    # Find optimal configuration
    best = df_results.loc[df_results['epochs'].idxmin()]
    print(f"\n🏆 Best configuration: λ_BIC={best['lambda_BIC']}, γ={best['gamma']}")
    print(f"   Epochs: {best['epochs']}, Final Loss: {best['final_loss']:.2e}")
    
    return df_results


# ============================================================
# EXPERIMENT 2: Model Size Scalability
# ============================================================

def experiment_2_scalability():
    """Test speedup across different model sizes"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: MODEL SIZE SCALABILITY")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V, t, I = load_memristor_data()
    
    hidden_sizes = [32, 64, 128]
    results = []
    
    for hidden_size in hidden_sizes:
        print(f"\n--- Hidden Size: {hidden_size} ---")
        
        # Baseline (random init)
        torch.manual_seed(42)
        model_base = PSI_xLSTM_Teacher(input_size=2, hidden_size=hidden_size).to(device)
        apply_random_init(model_base)
        metrics_base = train_model(model_base, V, t, I, device=device)
        print(f"  Baseline: {metrics_base['epochs']} epochs, {metrics_base['parameters']:,} params")
        
        # Ψ-Vortex (physics-aware init)
        torch.manual_seed(42)
        model_vortex = PSI_xLSTM_Teacher(input_size=2, hidden_size=hidden_size).to(device)
        apply_psi_vortex_init(model_vortex, pde_type="memristor")
        metrics_vortex = train_model(model_vortex, V, t, I, device=device)
        print(f"  Ψ-Vortex: {metrics_vortex['epochs']} epochs")
        
        speedup = metrics_base['epochs'] / metrics_vortex['epochs'] if metrics_vortex['epochs'] > 0 else 0
        
        results.append({
            'hidden_size': hidden_size,
            'parameters': metrics_base['parameters'],
            'baseline_epochs': metrics_base['epochs'],
            'baseline_time': metrics_base['time'],
            'baseline_loss': metrics_base['final_loss'],
            'vortex_epochs': metrics_vortex['epochs'],
            'vortex_time': metrics_vortex['time'],
            'vortex_loss': metrics_vortex['final_loss'],
            'speedup_epochs': speedup,
            'speedup_time': metrics_base['time'] / metrics_vortex['time'] if metrics_vortex['time'] > 0 else 0
        })
        
        print(f"  Speedup: {speedup:.2f}x (epochs), {results[-1]['speedup_time']:.2f}x (time)")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('scalability_results.csv', index=False)
    print("\n📊 Results saved to 'scalability_results.csv'")
    
    # Summary
    print("\n" + "-"*50)
    print("SCALABILITY SUMMARY")
    print("-"*50)
    print(f"{'Hidden':<10} {'Params':<12} {'Base Epochs':<12} {'Vortex Epochs':<14} {'Speedup':<10}")
    for r in results:
        print(f"{r['hidden_size']:<10} {r['parameters']:<12,} {r['baseline_epochs']:<12} "
              f"{r['vortex_epochs']:<14} {r['speedup_epochs']:.2f}x")
    
    return df_results


# ============================================================
# EXPERIMENT 3: Ψ-Family Lineage Comparison
# ============================================================

def experiment_3_psi_family_comparison():
    """Compare Ψ-HDL, Ψ-xLSTM, and Ψ-Vortex"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Ψ-FAMILY LINEAGE COMPARISON")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V, t, I = load_memristor_data()
    
    results = []
    
    # 1. PSI-HDL: Simple MLP with L2 regularization
    print("\n--- Ψ-HDL (MLP + L2) ---")
    torch.manual_seed(42)
    model_hdl = SimpleMLP(input_size=2, hidden_size=64, num_layers=4).to(device)
    apply_random_init(model_hdl)
    
    # Training with L2 (weight decay)
    optimizer = torch.optim.Adam(model_hdl.parameters(), lr=0.005, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    epochs_hdl = 500
    final_loss_hdl = float('inf')
    
    V_d, t_d, I_d = V.to(device), t.to(device), I.to(device)
    
    for epoch in range(epochs_hdl):
        optimizer.zero_grad()
        pred, _ = model_hdl(V_d, t_d)
        loss = loss_fn(pred, I_d)
        loss.backward()
        optimizer.step()
        
        if loss.item() < 1e-6:
            epochs_hdl = epoch + 1
            final_loss_hdl = loss.item()
            break
        final_loss_hdl = loss.item()
    
    time_hdl = time.time() - start_time
    
    results.append({
        'method': 'Ψ-HDL',
        'architecture': 'MLP (4 layers)',
        'regularization': 'L2 (weight decay)',
        'initialization': 'Random Xavier',
        'parameters': model_hdl.count_parameters(),
        'epochs': epochs_hdl,
        'time': time_hdl,
        'final_loss': final_loss_hdl,
        'spectral_bias': 'Yes (MLP)',
        'temporal_dynamics': 'No'
    })
    print(f"  Params: {model_hdl.count_parameters():,}, Epochs: {epochs_hdl}, Loss: {final_loss_hdl:.2e}")
    
    # 2. PSI-xLSTM: xLSTM with RRAD distillation (no physics init)
    print("\n--- Ψ-xLSTM (xLSTM + RRAD) ---")
    torch.manual_seed(42)
    model_xlstm = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
    apply_random_init(model_xlstm)
    
    metrics_xlstm = train_model(model_xlstm, V, t, I, device=device)
    
    results.append({
        'method': 'Ψ-xLSTM',
        'architecture': 'mLSTM + sLSTM',
        'regularization': 'RRAD',
        'initialization': 'Random Xavier',
        'parameters': model_xlstm.count_parameters(),
        'epochs': metrics_xlstm['epochs'],
        'time': metrics_xlstm['time'],
        'final_loss': metrics_xlstm['final_loss'],
        'spectral_bias': 'No (xLSTM)',
        'temporal_dynamics': 'Yes'
    })
    print(f"  Params: {model_xlstm.count_parameters():,}, Epochs: {metrics_xlstm['epochs']}, "
          f"Loss: {metrics_xlstm['final_loss']:.2e}")
    
    # 3. Ψ-Vortex: xLSTM with physics-aware init + adaptive BIC
    print("\n--- Ψ-Vortex (xLSTM + Physics Init + BIC) ---")
    torch.manual_seed(42)
    model_vortex = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
    apply_psi_vortex_init(model_vortex, pde_type="memristor")
    
    metrics_vortex = train_model(
        model_vortex, V, t, I,
        use_bic=True,
        lambda_bic=0.01,
        gamma=0.1,
        device=device
    )
    
    results.append({
        'method': 'Ψ-Vortex',
        'architecture': 'mLSTM + sLSTM',
        'regularization': 'RRAD + Adaptive BIC',
        'initialization': 'Physics-Aware (Eq. 5)',
        'parameters': model_vortex.count_parameters(),
        'epochs': metrics_vortex['epochs'],
        'time': metrics_vortex['time'],
        'final_loss': metrics_vortex['final_loss'],
        'spectral_bias': 'No (xLSTM)',
        'temporal_dynamics': 'Yes'
    })
    print(f"  Params: {model_vortex.count_parameters():,}, Epochs: {metrics_vortex['epochs']}, "
          f"Loss: {metrics_vortex['final_loss']:.2e}")
    
    # Create comparison table
    df_results = pd.DataFrame(results)
    df_results.to_csv('psi_family_comparison.csv', index=False)
    
    # Print formatted table
    print("\n" + "="*90)
    print("Ψ-FAMILY LINEAGE COMPARISON TABLE")
    print("="*90)
    print(f"{'Method':<12} {'Architecture':<18} {'Initialization':<20} {'Epochs':<10} "
          f"{'Time(s)':<10} {'Loss':<12}")
    print("-"*90)
    for r in results:
        print(f"{r['method']:<12} {r['architecture']:<18} {r['initialization']:<20} "
              f"{r['epochs']:<10} {r['time']:<10.2f} {r['final_loss']:<12.2e}")
    
    # Calculate improvements
    if len(results) >= 3:
        hdl_epochs = results[0]['epochs']
        xlstm_epochs = results[1]['epochs']
        vortex_epochs = results[2]['epochs']
        
        print("\n📈 IMPROVEMENTS:")
        print(f"  Ψ-Vortex vs Ψ-HDL: {hdl_epochs/vortex_epochs:.2f}x faster")
        print(f"  Ψ-Vortex vs Ψ-xLSTM: {xlstm_epochs/vortex_epochs:.2f}x faster")
    
    print("\n📊 Results saved to 'psi_family_comparison.csv'")
    
    return df_results


# ============================================================
# EXPERIMENT 4: Extended Ablation (2×2×2 grid)
# ============================================================

def experiment_4_extended_ablation():
    """Extended ablation: Init × BIC × Symmetry (2×2×2 = 8 configs)"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: EXTENDED ABLATION (2×2×2 GRID)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V, t, I = generate_synthetic_odd_data()
    
    # Factors
    init_types = ['random', 'physics']  # Initialization
    bic_types = [False, True]  # BIC regularization
    symmetry_types = ['none', 'odd']  # Symmetry prior
    
    results = []
    
    for init_type, use_bic, sym_type in product(init_types, bic_types, symmetry_types):
        config_name = f"Init:{init_type[:3]}_BIC:{int(use_bic)}_Sym:{sym_type[:3]}"
        print(f"\n--- {config_name} ---")
        
        torch.manual_seed(42)
        model = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
        
        # Apply initialization
        if init_type == 'random':
            apply_random_init(model)
        else:
            pde = 'memristor' if sym_type == 'odd' else 'none'
            apply_psi_vortex_init(model, pde_type=pde)
        
        # Train
        metrics = train_model(
            model, V, t, I,
            target_mse=1e-6,
            max_epochs=500,
            use_bic=use_bic,
            lambda_bic=0.01,
            gamma=0.1,
            device=device
        )
        
        results.append({
            'config': config_name,
            'initialization': init_type,
            'bic_regularization': use_bic,
            'symmetry_prior': sym_type,
            'epochs': metrics['epochs'],
            'time': metrics['time'],
            'final_loss': metrics['final_loss']
        })
        
        print(f"  Epochs: {metrics['epochs']}, Loss: {metrics['final_loss']:.2e}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('extended_ablation_results.csv', index=False)
    
    # Print ablation table
    print("\n" + "="*80)
    print("EXTENDED ABLATION RESULTS (2×2×2 GRID)")
    print("="*80)
    print(f"{'Configuration':<35} {'Epochs':<10} {'Time(s)':<10} {'Loss':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['config']:<35} {r['epochs']:<10} {r['time']:<10.2f} {r['final_loss']:<15.2e}")
    
    # Analyze main effects
    print("\n📈 MAIN EFFECTS ANALYSIS:")
    
    # Effect of initialization
    random_init = [r['epochs'] for r in results if r['initialization'] == 'random']
    physics_init = [r['epochs'] for r in results if r['initialization'] == 'physics']
    print(f"  Init effect: Random={np.mean(random_init):.1f}±{np.std(random_init):.1f} vs "
          f"Physics={np.mean(physics_init):.1f}±{np.std(physics_init):.1f}")
    
    # Effect of BIC
    no_bic = [r['epochs'] for r in results if not r['bic_regularization']]
    with_bic = [r['epochs'] for r in results if r['bic_regularization']]
    print(f"  BIC effect: Without={np.mean(no_bic):.1f}±{np.std(no_bic):.1f} vs "
          f"With={np.mean(with_bic):.1f}±{np.std(with_bic):.1f}")
    
    # Effect of symmetry
    no_sym = [r['epochs'] for r in results if r['symmetry_prior'] == 'none']
    with_sym = [r['epochs'] for r in results if r['symmetry_prior'] == 'odd']
    print(f"  Symmetry effect: None={np.mean(no_sym):.1f}±{np.std(no_sym):.1f} vs "
          f"Odd={np.mean(with_sym):.1f}±{np.std(with_sym):.1f}")
    
    # Find best configuration
    best = min(results, key=lambda x: x['epochs'])
    print(f"\n🏆 Best configuration: {best['config']}")
    print(f"   Epochs: {best['epochs']}, Loss: {best['final_loss']:.2e}")
    
    print("\n📊 Results saved to 'extended_ablation_results.csv'")
    
    return df_results


# ============================================================
# MAIN: Run All Experiments
# ============================================================

def run_all_extended_experiments():
    """Run all extended experiments"""
    print("="*70)
    print("Ψ-VORTEX EXTENDED EXPERIMENTS")
    print("="*70)
    
    all_results = {}
    
    # Experiment 1: BIC Sensitivity
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 1: BIC SENSITIVITY")
    print("#"*70)
    all_results['bic_sensitivity'] = experiment_1_bic_sensitivity()
    
    # Experiment 2: Scalability
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 2: SCALABILITY")
    print("#"*70)
    all_results['scalability'] = experiment_2_scalability()
    
    # Experiment 3: Ψ-Family Comparison
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 3: Ψ-FAMILY COMPARISON")
    print("#"*70)
    all_results['psi_family'] = experiment_3_psi_family_comparison()
    
    # Experiment 4: Extended Ablation
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 4: EXTENDED ABLATION")
    print("#"*70)
    all_results['extended_ablation'] = experiment_4_extended_ablation()
    
    # Final Summary
    print("\n\n" + "="*70)
    print("EXTENDED EXPERIMENTS COMPLETE - SUMMARY")
    print("="*70)
    
    print("\n📁 Generated Files:")
    print("   - bic_sensitivity_results.csv")
    print("   - scalability_results.csv")
    print("   - psi_family_comparison.csv")
    print("   - extended_ablation_results.csv")
    
    return all_results


if __name__ == "__main__":
    results = run_all_extended_experiments()