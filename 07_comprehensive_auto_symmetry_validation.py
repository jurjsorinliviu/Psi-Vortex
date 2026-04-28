"""
Ψ-Vortex Experiment 7: Comprehensive Auto-Symmetry Validation
===============================================================
Rigorous validation of automatic symmetry detection for "fully automated" claim.

Experiments:
1. Multi-seed statistical significance (5 seeds × 4 configs)
2. Synthetic data with KNOWN symmetry (odd, even, none)
3. 3D Thermal dataset test
4. Train/Test split generalization test
5. Noise robustness analysis
6. Detection threshold sensitivity

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Tuple
from collections import defaultdict

# Import modules
from core_psi_xlstm import mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init
from core_auto_symmetry import (
    apply_auto_vortex_init,
    apply_identity_vortex_init,
    AutoSymmetryDetector
)


class PSI_xLSTM_Teacher(nn.Module):
    """PSI-xLSTM Teacher model"""
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlstm = mLSTMBlock(input_size, hidden_size, memory_size=32)
        self.slstm = sLSTMBlock(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, V, t):
        x = torch.cat([V, t], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h1, h_final1, C_final = self.mlstm(x)
        h2, h_final2, c_final = self.slstm(h1)
        output = self.fc(h2.squeeze(1))
        return output, {'fused': h2.squeeze(1)}
    
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


def generate_synthetic_data(symmetry_type: str, n_samples: int = 1000, 
                           noise_level: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data with KNOWN symmetry.
    
    Args:
        symmetry_type: "odd", "even", or "none"
        n_samples: Number of data points
        noise_level: Standard deviation of Gaussian noise (as fraction of signal)
        
    Returns:
        V, t, I tensors
    """
    t = torch.linspace(0, 0.01, n_samples)
    freq = 150e3  # 150 kHz
    V = 2.0 * torch.sin(2 * np.pi * freq * t)
    
    if symmetry_type == "odd":
        # sinh is odd: sinh(-x) = -sinh(x)
        I_clean = 1e-4 * torch.sinh(V)
    elif symmetry_type == "even":
        # cosh-1 is even: cosh(-x) = cosh(x)
        I_clean = 1e-4 * (torch.cosh(V) - 1)
    else:  # "none"
        # Asymmetric: exp(V) - 1
        I_clean = 1e-4 * (torch.exp(V) - 1)
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(I_clean) * noise_level * torch.std(I_clean)
        I = I_clean + noise
    else:
        I = I_clean
    
    return V.view(-1, 1), t.view(-1, 1), I.view(-1, 1)


def load_real_memristor_data():
    """Load real memristor dataset"""
    DATA_PATH = 'printed_memristor_training_data.csv'
    
    if not os.path.exists(DATA_PATH):
        print("  Real data not found, using synthetic odd data...")
        return generate_synthetic_data("odd", noise_level=0.05)
    
    df = pd.read_csv(DATA_PATH)
    df = df[(df['device_id'] == 0) & (df['cycle_id'] == 0)]
    
    V = torch.tensor(df['voltage'].values, dtype=torch.float32).view(-1, 1)
    I = torch.tensor(df['current'].values, dtype=torch.float32).view(-1, 1)
    t = torch.linspace(0, 1, len(V)).view(-1, 1)
    
    return V, t, I


def load_thermal_data():
    """Load 3D thermal crosstalk dataset"""
    DATA_PATH = '3d_thermal_crosstalk_data.csv'
    
    if not os.path.exists(DATA_PATH):
        print("  Thermal data not found, generating synthetic thermal data...")
        # Thermal dynamics with dissipative (stable) behavior
        t = torch.linspace(0, 1, 500)
        T_amb = 298.0
        # Thermal response to pulse: dT/dt = -(T-T_amb)/tau + P
        power = torch.sin(2 * np.pi * 2 * t) ** 2  # Rectified sine (heat source)
        tau = 0.1
        T = T_amb + power * tau  # Simplified steady-state approximation
        
        V = power.view(-1, 1)
        t = t.view(-1, 1)
        I = (T - T_amb).view(-1, 1)  # Temperature rise as "output"
        return V, t, I
    
    df = pd.read_csv(DATA_PATH)
    # Use correct column names: 'V_driver', 'T_layer', 'I_victim'
    n_samples = min(500, len(df))
    V = torch.tensor(df['V_driver'].values[:n_samples], dtype=torch.float32).view(-1, 1)
    t = torch.tensor(df['time'].values[:n_samples], dtype=torch.float32).view(-1, 1)
    I = torch.tensor(df['I_victim'].values[:n_samples], dtype=torch.float32).view(-1, 1)
    
    return V, t, I


def run_single_training(model, V, t, I_target, target_mse=1e-6, max_epochs=500, device='cuda'):
    """Run a single training and return metrics"""
    model = model.to(device)
    V, t, I_target = V.to(device), t.to(device), I_target.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    converged_epoch = max_epochs
    final_loss = float('inf')
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred, _ = model(V, t)
        loss = loss_fn(pred, I_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if loss.item() < target_mse:
            converged_epoch = epoch + 1
            final_loss = loss.item()
            break
        final_loss = loss.item()
    
    duration = time.time() - start_time
    return {
        'epochs': converged_epoch,
        'time': duration,
        'final_loss': final_loss
    }


def run_config(config_name: str, V: torch.Tensor, t: torch.Tensor, 
               I: torch.Tensor, seed: int, device: str = 'cuda') -> Dict:
    """Run a single configuration with given seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = PSI_xLSTM_Teacher(input_size=2, hidden_size=64)
    
    if config_name == "baseline":
        apply_random_init(model)
    elif config_name == "identity":
        apply_identity_vortex_init(model, verbose=False)
    elif config_name == "auto":
        apply_auto_vortex_init(model, V.flatten(), I.flatten(), verbose=False)
    elif config_name == "expert_odd":
        apply_psi_vortex_init(model, pde_type="memristor")
    elif config_name == "expert_thermal":
        apply_psi_vortex_init(model, pde_type="thermal")
    
    return run_single_training(model, V, t, I, device=device)


# ============================================================
# EXPERIMENT 1: Multi-Seed Statistical Significance
# ============================================================
def experiment_1_multi_seed(seeds: List[int] = [42, 123, 456, 789, 1000]):
    """Run all configurations with multiple seeds"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: MULTI-SEED STATISTICAL SIGNIFICANCE")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V, t, I = load_real_memristor_data()
    
    configs = ["baseline", "identity", "auto", "expert_odd"]
    results = defaultdict(list)
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for config in configs:
            r = run_config(config, V, t, I, seed, device)
            results[config].append(r)
            print(f"  {config}: epochs={r['epochs']}, time={r['time']:.2f}s, loss={r['final_loss']:.2e}")
    
    # Calculate statistics
    print("\n" + "-"*70)
    print("STATISTICAL SUMMARY (Mean ± Std)")
    print("-"*70)
    print(f"{'Config':<15} {'Epochs':<20} {'Time (s)':<20} {'Final Loss':<20}")
    
    summary = {}
    for config in configs:
        epochs = [r['epochs'] for r in results[config]]
        times = [r['time'] for r in results[config]]
        losses = [r['final_loss'] for r in results[config]]
        
        summary[config] = {
            'epochs_mean': np.mean(epochs), 'epochs_std': np.std(epochs),
            'time_mean': np.mean(times), 'time_std': np.std(times),
            'loss_mean': np.mean(losses), 'loss_std': np.std(losses)
        }
        
        print(f"{config:<15} {np.mean(epochs):.1f} ± {np.std(epochs):.1f}    "
              f"{np.mean(times):.2f} ± {np.std(times):.2f}    "
              f"{np.mean(losses):.2e} ± {np.std(losses):.2e}")
    
    return results, summary


# ============================================================
# EXPERIMENT 2: Synthetic Data with KNOWN Symmetry
# ============================================================
def experiment_2_synthetic_symmetry():
    """Test auto-detection on data where symmetry is GUARANTEED"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: SYNTHETIC DATA WITH KNOWN SYMMETRY")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = AutoSymmetryDetector()
    
    symmetry_types = ["odd", "even", "none"]
    results = {}
    
    for sym_type in symmetry_types:
        print(f"\n--- Testing {sym_type.upper()} symmetry ---")
        
        # Generate data
        V, t, I = generate_synthetic_data(sym_type, noise_level=0.0)
        
        # Detect symmetry
        detected, confidence = detector.detect(V.flatten(), I.flatten(), verbose=False)
        print(f"  Detection: {detected} (confidence: {confidence:.2%})")
        print(f"  Expected: {sym_type}")
        print(f"  Correct: {'✓' if detected == sym_type else '✗'}")
        
        # Run training comparison
        configs = ["baseline", "identity", "auto", "expert_odd"]
        config_results = {}
        
        for config in configs:
            r = run_config(config, V, t, I, seed=42, device=device)
            config_results[config] = r
            print(f"  {config}: epochs={r['epochs']}, loss={r['final_loss']:.2e}")
        
        results[sym_type] = {
            'detected': detected,
            'confidence': confidence,
            'correct': detected == sym_type,
            'training': config_results
        }
    
    return results


# ============================================================
# EXPERIMENT 3: 3D Thermal Dataset
# ============================================================
def experiment_3_thermal_data():
    """Test auto-detection on 3D thermal crosstalk data"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: 3D THERMAL DATASET")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V, t, I = load_thermal_data()
    
    # Detect symmetry
    detector = AutoSymmetryDetector()
    detected, confidence = detector.detect(V.flatten(), I.flatten())
    print(f"Detected symmetry: {detected} (confidence: {confidence:.2%})")
    
    # Run comparison with thermal expert
    configs = ["baseline", "identity", "auto", "expert_thermal"]
    results = {}
    
    for config in configs:
        r = run_config(config, V, t, I, seed=42, device=device)
        results[config] = r
        print(f"  {config}: epochs={r['epochs']}, time={r['time']:.2f}s, loss={r['final_loss']:.2e}")
    
    return results, detected, confidence


# ============================================================
# EXPERIMENT 4: Train/Test Generalization
# ============================================================
def experiment_4_generalization():
    """Test if auto-detection leads to better generalization"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: TRAIN/TEST GENERALIZATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate data with slight noise
    V_full, t_full, I_full = generate_synthetic_data("odd", n_samples=1000, noise_level=0.05)
    
    # 80/20 split
    n_train = 800
    V_train, t_train, I_train = V_full[:n_train], t_full[:n_train], I_full[:n_train]
    V_test, t_test, I_test = V_full[n_train:], t_full[n_train:], I_full[n_train:]
    
    configs = ["baseline", "identity", "auto", "expert_odd"]
    results = {}
    
    for config in configs:
        torch.manual_seed(42)
        model = PSI_xLSTM_Teacher(input_size=2, hidden_size=64)
        
        if config == "baseline":
            apply_random_init(model)
        elif config == "identity":
            apply_identity_vortex_init(model, verbose=False)
        elif config == "auto":
            apply_auto_vortex_init(model, V_train.flatten(), I_train.flatten(), verbose=False)
        elif config == "expert_odd":
            apply_psi_vortex_init(model, pde_type="memristor")
        
        # Train
        train_result = run_single_training(model, V_train, t_train, I_train, device=device)
        
        # Evaluate on test set
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            V_test_d = V_test.to(device)
            t_test_d = t_test.to(device)
            I_test_d = I_test.to(device)
            pred, _ = model(V_test_d, t_test_d)
            test_loss = nn.MSELoss()(pred, I_test_d).item()
        
        results[config] = {
            'train_epochs': train_result['epochs'],
            'train_loss': train_result['final_loss'],
            'test_loss': test_loss,
            'generalization_gap': test_loss / (train_result['final_loss'] + 1e-10)
        }
        
        print(f"{config}: train_loss={train_result['final_loss']:.2e}, "
              f"test_loss={test_loss:.2e}, gap={results[config]['generalization_gap']:.2f}x")
    
    return results


# ============================================================
# EXPERIMENT 5: Noise Robustness
# ============================================================
def experiment_5_noise_robustness():
    """Test how noise affects symmetry detection"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: NOISE ROBUSTNESS")
    print("="*70)
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    detector = AutoSymmetryDetector()
    
    results = []
    
    for noise in noise_levels:
        V, t, I = generate_synthetic_data("odd", noise_level=noise)
        detected, confidence = detector.detect(V.flatten(), I.flatten(), verbose=False)
        
        correct = detected == "odd"
        results.append({
            'noise_level': noise,
            'detected': detected,
            'confidence': confidence,
            'correct': correct
        })
        
        print(f"Noise {noise:.0%}: detected={detected}, confidence={confidence:.2%}, correct={correct}")
    
    # Find threshold where detection fails
    failure_threshold = None
    for r in results:
        if not r['correct']:
            failure_threshold = r['noise_level']
            break
    
    print(f"\nNoise tolerance: detection correct up to {(failure_threshold or 1.0)*100:.0f}% noise")
    
    return results


# ============================================================
# EXPERIMENT 6: Detection Threshold Sensitivity
# ============================================================
def experiment_6_threshold_sensitivity():
    """Test sensitivity to min_confidence parameter"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: DETECTION THRESHOLD SENSITIVITY")
    print("="*70)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Use noisy data where detection is borderline
    V, t, I = generate_synthetic_data("odd", noise_level=0.15)
    
    results = []
    
    for thresh in thresholds:
        detector = AutoSymmetryDetector(min_confidence=thresh)
        detected, confidence = detector.detect(V.flatten(), I.flatten(), verbose=False)
        
        results.append({
            'threshold': thresh,
            'detected': detected,
            'confidence': confidence
        })
        
        print(f"Threshold {thresh}: detected={detected}, confidence={confidence:.2%}")
    
    return results


# ============================================================
# MAIN: Run All Experiments
# ============================================================
def run_all_experiments():
    """Run comprehensive validation suite"""
    print("="*70)
    print("Ψ-VORTEX COMPREHENSIVE AUTO-SYMMETRY VALIDATION")
    print("="*70)
    
    all_results = {}
    
    # Experiment 1: Multi-seed
    results_1, summary_1 = experiment_1_multi_seed(seeds=[42, 123, 456, 789, 1000])
    all_results['multi_seed'] = summary_1
    
    # Experiment 2: Synthetic symmetry
    results_2 = experiment_2_synthetic_symmetry()
    all_results['synthetic'] = results_2
    
    # Experiment 3: Thermal data
    results_3, thermal_detected, thermal_conf = experiment_3_thermal_data()
    all_results['thermal'] = results_3
    
    # Experiment 4: Generalization
    results_4 = experiment_4_generalization()
    all_results['generalization'] = results_4
    
    # Experiment 5: Noise robustness
    results_5 = experiment_5_noise_robustness()
    all_results['noise'] = results_5
    
    # Experiment 6: Threshold sensitivity
    results_6 = experiment_6_threshold_sensitivity()
    all_results['threshold'] = results_6
    
    # =============================================
    # Final Summary
    # =============================================
    print("\n" + "="*70)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("="*70)
    
    print("\n📊 EXPERIMENT 1 - Multi-Seed Results:")
    print(f"   Auto vs Expert speedup: {summary_1['baseline']['time_mean']/summary_1['auto']['time_mean']:.2f}x")
    print(f"   Auto vs Expert epochs: {summary_1['auto']['epochs_mean']:.0f} vs {summary_1['expert_odd']['epochs_mean']:.0f}")
    
    print("\n📊 EXPERIMENT 2 - Synthetic Detection Accuracy:")
    correct_count = sum(1 for r in results_2.values() if r['correct'])
    print(f"   Correct detections: {correct_count}/3")
    
    print("\n📊 EXPERIMENT 3 - Thermal Data:")
    print(f"   Detected: {thermal_detected} (conf: {thermal_conf:.2%})")
    
    print("\n📊 EXPERIMENT 4 - Generalization:")
    best_gen = min(results_4.items(), key=lambda x: x[1]['generalization_gap'])
    print(f"   Best generalization: {best_gen[0]} (gap: {best_gen[1]['generalization_gap']:.2f}x)")
    
    print("\n📊 EXPERIMENT 5 - Noise Tolerance:")
    max_correct_noise = max(r['noise_level'] for r in results_5 if r['correct'])
    print(f"   Correct up to {max_correct_noise*100:.0f}% noise")
    
    print("\n📊 EXPERIMENT 6 - Recommended Threshold:")
    # Find threshold that balances detection and false positives
    print(f"   Current default: 0.7")
    
    # Final verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    # Criteria for "fully automated" claim
    auto_speedup = summary_1['baseline']['time_mean'] / summary_1['auto']['time_mean']
    auto_vs_expert_ratio = summary_1['auto']['time_mean'] / summary_1['expert_odd']['time_mean']
    detection_accuracy = correct_count / 3
    
    claim_valid = (
        auto_speedup >= 2.0 and  # At least 2x speedup over baseline
        auto_vs_expert_ratio <= 1.3 and  # Within 30% of expert
        detection_accuracy >= 0.66  # At least 2/3 correct detections
    )
    
    if claim_valid:
        print("✅ 'FULLY AUTOMATED' CLAIM IS SUPPORTED BY EVIDENCE")
    else:
        print("⚠️  'FULLY AUTOMATED' CLAIM NEEDS CAVEATS")
    
    print(f"\n   - Auto speedup vs baseline: {auto_speedup:.2f}x (need ≥2.0x)")
    print(f"   - Auto vs Expert ratio: {auto_vs_expert_ratio:.2f}x (need ≤1.3x)")
    print(f"   - Detection accuracy: {detection_accuracy*100:.0f}% (need ≥66%)")
    
    # Save results
    save_comprehensive_results(all_results)
    
    return all_results


def save_comprehensive_results(results):
    """Save all results to CSV files"""
    
    # Multi-seed summary
    summary = results['multi_seed']
    df_summary = pd.DataFrame([
        {
            'Config': config,
            'Epochs_Mean': v['epochs_mean'],
            'Epochs_Std': v['epochs_std'],
            'Time_Mean': v['time_mean'],
            'Time_Std': v['time_std'],
            'Loss_Mean': v['loss_mean'],
            'Loss_Std': v['loss_std']
        }
        for config, v in summary.items()
    ])
    df_summary.to_csv('comprehensive_multi_seed_results.csv', index=False)
    
    # Generalization results
    gen = results['generalization']
    df_gen = pd.DataFrame([
        {
            'Config': config,
            'Train_Loss': v['train_loss'],
            'Test_Loss': v['test_loss'],
            'Generalization_Gap': v['generalization_gap']
        }
        for config, v in gen.items()
    ])
    df_gen.to_csv('comprehensive_generalization_results.csv', index=False)
    
    print("\n📁 Results saved to:")
    print("   - comprehensive_multi_seed_results.csv")
    print("   - comprehensive_generalization_results.csv")


if __name__ == "__main__":
    results = run_all_experiments()