"""
Ψ-Vortex Experiment 6: Automatic Symmetry Detection Validation
================================================================
Compares four initialization configurations to validate that automatic
symmetry detection achieves comparable performance to manual domain expertise.

Configurations tested:
1. Baseline: Random Xavier initialization (no symmetry)
2. Identity: Orthogonal init with M_sym = 1 (no symmetry prior)
3. Auto-Detected: Orthogonal init with automatically detected M_sym
4. Expert/Manual: Orthogonal init with manually specified M_sym (original)

Expected results:
- Auto should perform close to Expert (within 10%)
- Both Auto and Expert should beat Baseline by ~3x
- Identity should be between Baseline and Expert

If Auto ≈ Expert: "Fully automated" claim is valid
If Auto << Expert: Keep limitations section, save for future work

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import existing modules
from core_psi_xlstm import mLSTMBlock, sLSTMBlock
from core_physics_init import apply_psi_vortex_init

# Import NEW automatic symmetry module
from core_auto_symmetry import (
    apply_auto_vortex_init,
    apply_identity_vortex_init,
    AutoSymmetryDetector,
    test_symmetry_detection
)

# Data path
DATA_PATH = 'printed_memristor_training_data.csv'


class PSI_xLSTM_Teacher(nn.Module):
    """
    PSI-xLSTM Teacher model (same as in benchmark)
    """
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
        hidden_states = {
            'fused': h2.squeeze(1),
            'block_hiddens': [h_final1, h_final2],
            'block_memories': [C_final, c_final]
        }
        return output, hidden_states
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_memristor_data():
    """Load and preprocess printed memristor data"""
    print(f"Loading data from {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print("Data file not found, generating synthetic memristor data...")
        t = torch.linspace(0, 0.01, 1000)  # 10ms
        freq = 150e3  # 150 kHz
        V = 2.0 * torch.sin(2 * np.pi * freq * t)
        # Memristor with TRUE ODD symmetry: I(-V) = -I(V)
        # Using sinh which is odd: sinh(-V) = -sinh(V)
        # Adding hysteresis effect that preserves odd symmetry
        I = 1e-4 * torch.sinh(V) * (1 + 0.3 * torch.cos(4 * np.pi * freq * t))
        
        V = V.view(-1, 1)
        t = t.view(-1, 1)
        I = I.view(-1, 1)
        return V, t, I
    
    df = pd.read_csv(DATA_PATH)
    df = df[(df['device_id'] == 0) & (df['cycle_id'] == 0)]
    
    V = torch.tensor(df['voltage'].values, dtype=torch.float32).view(-1, 1)
    I = torch.tensor(df['current'].values, dtype=torch.float32).view(-1, 1)
    t = torch.linspace(0, 1, len(V)).view(-1, 1)
    
    return V, t, I


def apply_random_init(model):
    """Standard random initialization (baseline)"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def run_training(model_name: str, model: nn.Module, V: torch.Tensor, 
                 t: torch.Tensor, I_target: torch.Tensor,
                 target_mse: float = 1e-6, max_epochs: int = 500):
    """
    Run training and measure convergence time
    """
    print(f"\nTraining {model_name}...")
    print(f"  Model parameters: {model.count_parameters():,}")
    
    device = next(model.parameters()).device
    V, t, I_target = V.to(device), t.to(device), I_target.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    loss_history = []
    start_time = time.time()
    converged_epoch = max_epochs
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred, _ = model(V, t)
        loss = loss_fn(pred, I_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.2e}")
        
        if loss.item() < target_mse:
            converged_epoch = epoch + 1
            print(f"  -> Converged at Epoch {converged_epoch}")
            break
    
    duration = time.time() - start_time
    print(f"  Final Loss: {loss_history[-1]:.2e}")
    print(f"  Training Time: {duration:.2f}s")
    
    return loss_history, duration, converged_epoch


def run_auto_symmetry_experiment():
    """
    Main experiment: Compare all four initialization strategies
    
    This validates whether automatic symmetry detection can replace
    manual domain expertise.
    """
    print("=" * 70)
    print("Ψ-VORTEX AUTOMATIC SYMMETRY DETECTION EXPERIMENT")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    V, t, I = load_memristor_data()
    
    # Store all results
    results = {}
    
    # =============================================
    # Test 0: Validate Symmetry Detection on Data
    # =============================================
    print("\n" + "=" * 50)
    print("STEP 0: SYMMETRY DETECTION TEST")
    print("=" * 50)
    
    detector = AutoSymmetryDetector()
    sym_type, confidence = detector.detect(V.flatten(), I.flatten())
    print(f"\nData Analysis:")
    print(f"  Detected symmetry: {sym_type}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Expected for memristor: odd (I(-V) = -I(V))")
    
    if sym_type != "odd":
        print("\n⚠️  WARNING: Expected odd symmetry but detected:", sym_type)
        print("    This may affect experiment results.")
    
    # =============================================
    # 1. Baseline (Random Initialization)
    # =============================================
    print("\n" + "-" * 50)
    print("CONFIG 1: BASELINE (Random Xavier Init)")
    print("-" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_baseline = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
    apply_random_init(model_baseline)
    
    loss_base, time_base, epochs_base = run_training(
        "Baseline", model_baseline, V, t, I
    )
    results['baseline'] = {
        'time': time_base, 
        'epochs': epochs_base, 
        'loss': loss_base,
        'final_loss': loss_base[-1]
    }
    
    # =============================================
    # 2. Identity Init (Orthogonal, M_sym = 1)
    # =============================================
    print("\n" + "-" * 50)
    print("CONFIG 2: IDENTITY (Orthogonal, no symmetry prior)")
    print("-" * 50)
    
    torch.manual_seed(42)
    
    model_identity = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
    apply_identity_vortex_init(model_identity, verbose=True)
    
    loss_ident, time_ident, epochs_ident = run_training(
        "Identity Init", model_identity, V, t, I
    )
    results['identity'] = {
        'time': time_ident, 
        'epochs': epochs_ident, 
        'loss': loss_ident,
        'final_loss': loss_ident[-1]
    }
    
    # =============================================
    # 3. Auto-Detected Symmetry
    # =============================================
    print("\n" + "-" * 50)
    print("CONFIG 3: AUTO-DETECTED SYMMETRY")
    print("-" * 50)
    
    torch.manual_seed(42)
    
    model_auto = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
    init_info = apply_auto_vortex_init(model_auto, V.flatten(), I.flatten(), verbose=True)
    
    loss_auto, time_auto, epochs_auto = run_training(
        "Auto-Detected Init", model_auto, V, t, I
    )
    results['auto'] = {
        'time': time_auto, 
        'epochs': epochs_auto, 
        'loss': loss_auto,
        'final_loss': loss_auto[-1],
        'detected_symmetry': init_info['symmetry_type'],
        'confidence': init_info['confidence']
    }
    
    # =============================================
    # 4. Expert/Manual Symmetry (Original)
    # =============================================
    print("\n" + "-" * 50)
    print("CONFIG 4: EXPERT/MANUAL SYMMETRY (memristor odd)")
    print("-" * 50)
    
    torch.manual_seed(42)
    
    model_expert = PSI_xLSTM_Teacher(input_size=2, hidden_size=64).to(device)
    apply_psi_vortex_init(model_expert, pde_type="memristor")
    
    loss_expert, time_expert, epochs_expert = run_training(
        "Expert Init", model_expert, V, t, I
    )
    results['expert'] = {
        'time': time_expert, 
        'epochs': epochs_expert, 
        'loss': loss_expert,
        'final_loss': loss_expert[-1]
    }
    
    # =============================================
    # 5. Calculate and Display Results
    # =============================================
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS: AUTO vs EXPERT COMPARISON")
    print("=" * 70)
    
    # Calculate speedups relative to baseline
    speedup_identity = time_base / time_ident if time_ident > 0 else 0
    speedup_auto = time_base / time_auto if time_auto > 0 else 0
    speedup_expert = time_base / time_expert if time_expert > 0 else 0
    
    # Calculate auto vs expert difference
    auto_vs_expert_time = time_auto / time_expert if time_expert > 0 else float('inf')
    auto_vs_expert_epochs = epochs_auto / epochs_expert if epochs_expert > 0 else float('inf')
    
    print(f"\n{'Config':<20} {'Epochs':<12} {'Time (s)':<12} {'Final Loss':<15} {'vs Baseline':<12}")
    print("-" * 70)
    print(f"{'Baseline':<20} {epochs_base:<12} {time_base:<12.2f} {loss_base[-1]:<15.2e} {'1.00x':<12}")
    print(f"{'Identity':<20} {epochs_ident:<12} {time_ident:<12.2f} {loss_ident[-1]:<15.2e} {speedup_identity:<12.2f}x")
    print(f"{'Auto-Detected':<20} {epochs_auto:<12} {time_auto:<12.2f} {loss_auto[-1]:<15.2e} {speedup_auto:<12.2f}x")
    print(f"{'Expert/Manual':<20} {epochs_expert:<12} {time_expert:<12.2f} {loss_expert[-1]:<15.2e} {speedup_expert:<12.2f}x")
    print("-" * 70)
    
    print(f"\n📊 AUTO vs EXPERT COMPARISON:")
    print(f"   Time ratio (Auto/Expert): {auto_vs_expert_time:.2f}x")
    print(f"   Epoch ratio (Auto/Expert): {auto_vs_expert_epochs:.2f}x")
    print(f"   Loss ratio (Auto/Expert): {loss_auto[-1]/loss_expert[-1]:.2f}x")
    
    # Determine if auto is "good enough"
    auto_acceptable = auto_vs_expert_time <= 1.2 and auto_vs_expert_epochs <= 1.2
    
    print(f"\n🎯 VERDICT:")
    if auto_acceptable:
        print("   ✅ AUTO-DETECTION IS ACCEPTABLE")
        print("   Auto-detected symmetry performs within 20% of expert knowledge.")
        print("   → 'Fully automated' claim is VALIDATED")
    else:
        print("   ⚠️  AUTO-DETECTION NEEDS IMPROVEMENT")
        print(f"   Auto-detected is {auto_vs_expert_time:.0%} slower than expert.")
        print("   → Consider keeping limitations section")
    
    # =============================================
    # 6. Generate Plot
    # =============================================
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_base, label='Baseline (Random)', color='gray', alpha=0.7)
    plt.plot(loss_ident, label='Identity (Orth only)', color='blue', alpha=0.7)
    plt.plot(loss_auto, label='Auto-Detected', color='orange', linewidth=2)
    plt.plot(loss_expert, label='Expert/Manual', color='green', linewidth=2, linestyle='--')
    plt.yscale('log')
    plt.axhline(y=1e-6, color='black', linestyle=':', label='Target MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Convergence Comparison:\nAuto-Detection vs Expert Knowledge')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    configs = ['Baseline', 'Identity', 'Auto', 'Expert']
    times = [time_base, time_ident, time_auto, time_expert]
    colors = ['gray', 'blue', 'orange', 'green']
    
    bars = plt.bar(configs, times, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    
    # Add speedup labels
    for i, (bar, t) in enumerate(zip(bars, times)):
        speedup = time_base / t if t > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{speedup:.2f}x', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('auto_symmetry_experiment.png', dpi=300)
    print("\n📈 Plot saved to 'auto_symmetry_experiment.png'")
    
    # =============================================
    # 7. Save Results to CSV
    # =============================================
    results_df = pd.DataFrame({
        'Configuration': ['Baseline', 'Identity', 'Auto-Detected', 'Expert/Manual'],
        'Init_Type': ['Random Xavier', 'Orthogonal (M=1)', 'Orthogonal (M=auto)', 'Orthogonal (M=expert)'],
        'Symmetry_Used': ['none', 'none', results['auto'].get('detected_symmetry', 'N/A'), 'odd (memristor)'],
        'Detection_Confidence': ['N/A', 'N/A', f"{results['auto'].get('confidence', 0):.2%}", 'N/A'],
        'Epochs': [epochs_base, epochs_ident, epochs_auto, epochs_expert],
        'Time_s': [f'{time_base:.2f}', f'{time_ident:.2f}', f'{time_auto:.2f}', f'{time_expert:.2f}'],
        'Final_Loss': [f'{loss_base[-1]:.2e}', f'{loss_ident[-1]:.2e}', f'{loss_auto[-1]:.2e}', f'{loss_expert[-1]:.2e}'],
        'Speedup_vs_Baseline': ['1.00x', f'{speedup_identity:.2f}x', f'{speedup_auto:.2f}x', f'{speedup_expert:.2f}x']
    })
    results_df.to_csv('auto_symmetry_results.csv', index=False)
    print("📊 Results saved to 'auto_symmetry_results.csv'")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PRELIMINARY: Testing symmetry detection algorithms")
    print("="*70)
    test_symmetry_detection()
    
    print("\n\n")
    results = run_auto_symmetry_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nNext steps based on results:")
    print("  - If Auto ≈ Expert: Update paper to claim 'fully automated'")
    print("  - If Auto << Expert: Keep limitations section for honest reporting")