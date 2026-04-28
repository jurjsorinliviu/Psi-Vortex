"""
Automatic Architecture Selection for Ψ-Vortex

This module implements VALIDATION-BASED architecture selection, which is more
appropriate for neural networks than information-theoretic criteria like BIC.

Key Insight: BIC was designed for statistical model selection with explicit
parameter counting. Neural networks have complex parameter interactions that
make raw parameter counting inappropriate. Instead, we use:

1. Validation-based selection: Train each candidate, pick best validation MSE
2. Efficiency tiebreaker: If MSEs are similar (within 10%), prefer smaller
3. Early stopping: Reduce search cost through adaptive evaluation

Mathematical Foundation:
------------------------
a* = argmin_{a ∈ A} MSE_val(a)

With efficiency preference:
If |MSE(a1) - MSE(a2)| / MSE(a1) < 0.1:
    prefer a with smaller k(a)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class ArchConfig:
    """Architecture configuration"""
    hidden_dim: int
    num_layers: int
    memory_size: int = 16
    
    @property
    def params(self) -> int:
        """Estimated parameter count"""
        h = self.hidden_dim
        L = self.num_layers
        return 2 * h + 4 * h * (h + 2) * L + h + 16 * L
    
    def __repr__(self):
        return f"(h={self.hidden_dim}, L={self.num_layers}, m={self.memory_size})"


class SimpleModel(nn.Module):
    """Simple LSTM model for architecture evaluation"""
    
    def __init__(self, config: ArchConfig, input_dim: int = 2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, config.hidden_dim)
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim, 
                           config.num_layers, batch_first=True)
        self.proj_out = nn.Linear(config.hidden_dim, 1)
        self.act = nn.Tanh()
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.act(self.proj_in(x))
        h, _ = self.lstm(h)
        return self.proj_out(h).squeeze(-1).squeeze(-1)


class AutoArchitectureSelector:
    """
    Automatic architecture selection through validation-based search.
    
    Algorithm:
    1. Define candidate architectures (grid)
    2. Train each candidate with early stopping
    3. Select architecture with best validation MSE
    4. Apply efficiency tiebreaker for similar MSEs
    """
    
    # Candidate architecture grid
    CANDIDATES = [
        # Small architectures
        ArchConfig(16, 1, 8),
        ArchConfig(16, 2, 8),
        ArchConfig(32, 1, 8),
        ArchConfig(32, 2, 16),
        # Medium architectures  
        ArchConfig(32, 2, 16),
        ArchConfig(64, 1, 16),
        ArchConfig(64, 2, 16),
        ArchConfig(64, 2, 32),
        # Large architectures
        ArchConfig(64, 3, 32),
        ArchConfig(128, 2, 32),
        ArchConfig(128, 3, 32),
    ]
    
    def __init__(
        self,
        candidates: Optional[List[ArchConfig]] = None,
        epochs: int = 100,
        patience: int = 15,
        lr: float = 0.001,
        efficiency_threshold: float = 0.1,  # 10% MSE similarity
        device: str = 'cpu'
    ):
        self.candidates = candidates or self.CANDIDATES
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.efficiency_threshold = efficiency_threshold
        self.device = device
        self.results = []
    
    def _train_and_evaluate(
        self, 
        config: ArchConfig,
        X_train: torch.Tensor, 
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        input_dim: int
    ) -> Dict:
        """Train model and return validation MSE"""
        
        model = SimpleModel(config, input_dim).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        X_val = X_val.to(self.device)
        Y_val = Y_val.to(self.device)
        
        best_val = float('inf')
        wait = 0
        
        t0 = time.time()
        for epoch in range(self.epochs):
            # Train step
            model.train()
            opt.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            opt.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, Y_val).item()
            
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        
        return {
            'config': config,
            'val_mse': best_val,
            'train_time': time.time() - t0,
            'epochs': epoch + 1,
            'params': config.params
        }
    
    def select(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        val_split: float = 0.2,
        input_dim: int = 2,
        verbose: bool = True
    ) -> Tuple[ArchConfig, Dict]:
        """
        Select optimal architecture for given data.
        
        Returns best architecture and selection info.
        """
        # Split data
        n = len(X)
        n_val = int(n * val_split)
        idx = torch.randperm(n)
        
        X_train, Y_train = X[idx[n_val:]], Y[idx[n_val:]]
        X_val, Y_val = X[idx[:n_val]], Y[idx[:n_val]]
        
        if verbose:
            print("=" * 60)
            print("AUTO-ARCHITECTURE SELECTION")
            print("=" * 60)
            print(f"\nEvaluating {len(self.candidates)} candidates...")
        
        # Evaluate all candidates
        self.results = []
        for config in self.candidates:
            if verbose:
                print(f"  Testing {config}...", end=" ")
            
            result = self._train_and_evaluate(
                config, X_train, Y_train, X_val, Y_val, input_dim
            )
            self.results.append(result)
            
            if verbose:
                print(f"MSE: {result['val_mse']:.2e}")
        
        # Find best by validation MSE
        sorted_results = sorted(self.results, key=lambda x: x['val_mse'])
        best = sorted_results[0]
        
        # Apply efficiency tiebreaker
        best_mse = best['val_mse']
        for result in sorted_results[1:]:
            relative_diff = (result['val_mse'] - best_mse) / (best_mse + 1e-10)
            if relative_diff < self.efficiency_threshold:
                # Similar MSE, prefer smaller model
                if result['params'] < best['params']:
                    if verbose:
                        print(f"\n  Efficiency tiebreaker: {result['config']} vs {best['config']}")
                    best = result
        
        # Summary
        info = {
            'selected': best['config'],
            'val_mse': best['val_mse'],
            'params': best['params'],
            'candidates_tested': len(self.candidates),
            'all_results': sorted_results
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("SELECTION RESULT")
            print("=" * 60)
            print(f"Selected: {best['config']}")
            print(f"Validation MSE: {best['val_mse']:.2e}")
            print(f"Parameters: {best['params']:,}")
            print(f"\nTop 3 candidates:")
            for i, r in enumerate(sorted_results[:3]):
                print(f"  {i+1}. {r['config']} - MSE: {r['val_mse']:.2e}")
        
        return best['config'], info


def auto_select_architecture(
    X: torch.Tensor, 
    Y: torch.Tensor,
    verbose: bool = True
) -> Tuple[ArchConfig, Dict]:
    """
    Convenience function for automatic architecture selection.
    
    Usage:
        >>> config, info = auto_select_architecture(X, Y)
        >>> print(f"Selected: h={config.hidden_dim}, L={config.num_layers}")
    """
    selector = AutoArchitectureSelector()
    return selector.select(X, Y, verbose=verbose)


# Validation test
def run_validation():
    """Validate auto-architecture selection"""
    
    print("=" * 70)
    print("AUTO-ARCHITECTURE VALIDATION")
    print("=" * 70)
    
    # Test datasets
    def make_data(complexity, n=500):
        t = torch.linspace(0, 1, n)
        if complexity == 'simple':
            V = 2 * torch.sin(2 * np.pi * t)
            I = 0.1 * V + 0.01 * V**3
        elif complexity == 'medium':
            V = 2 * torch.sin(2 * np.pi * 5 * t)
            s = torch.zeros(n)
            for i in range(1, n):
                s[i] = s[i-1] + 0.001 * V[i] * (1 - s[i-1]**2)
            I = V * (0.1 + 0.9 * torch.sigmoid(s * 5))
        else:
            V = 1.5 * torch.sin(2 * np.pi * 10 * t) + 0.5 * torch.sin(2 * np.pi * 50 * t)
            s, T = torch.zeros(n), torch.zeros(n)
            for i in range(1, n):
                T[i] = T[i-1] * 0.99 + V[i]**2 * 0.001
                s[i] = s[i-1] + 0.001 * V[i] * (1 - s[i-1]**2) * (1 + 0.1 * T[i])
            I = V * (0.1 + 0.9 * torch.sigmoid(s * 5)) * (1 + 0.05 * T)
        X = torch.stack([V, t], dim=1)
        return X, I + 0.001 * torch.randn(n)
    
    # Manual references
    MANUAL = {
        'simple': ArchConfig(32, 1, 8),
        'medium': ArchConfig(64, 2, 16),
        'complex': ArchConfig(128, 3, 32)
    }
    
    results = []
    
    for complexity in ['simple', 'medium', 'complex']:
        print(f"\n{'='*60}")
        print(f"COMPLEXITY: {complexity.upper()}")
        print('='*60)
        
        X, Y = make_data(complexity)
        
        # Manual baseline
        manual = MANUAL[complexity]
        n_val = 100
        X_t, Y_t = X[n_val:], Y[n_val:]
        X_v, Y_v = X[:n_val], Y[:n_val]
        
        model = SimpleModel(manual, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(100):
            opt.zero_grad()
            nn.MSELoss()(model(X_t), Y_t).backward()
            opt.step()
        with torch.no_grad():
            manual_mse = nn.MSELoss()(model(X_v), Y_v).item()
        
        print(f"Manual: {manual} -> MSE: {manual_mse:.2e}")
        
        # Auto selection
        selector = AutoArchitectureSelector(epochs=100, patience=15)
        auto_config, info = selector.select(X, Y, verbose=False)
        auto_mse = info['val_mse']
        
        print(f"Auto:   {auto_config} -> MSE: {auto_mse:.2e}")
        
        ratio = auto_mse / manual_mse if manual_mse > 0 else float('inf')
        passed = ratio <= 1.5  # Within 50% is acceptable for auto
        
        print(f"Ratio: {ratio:.2f}x {'✓' if passed else '✗'}")
        
        results.append({
            'complexity': complexity,
            'manual': str(manual),
            'manual_mse': manual_mse,
            'auto': str(auto_config),
            'auto_mse': auto_mse,
            'ratio': ratio,
            'passed': passed
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    avg_ratio = df['ratio'].mean()
    all_pass = df['passed'].all()
    
    print(f"\nAverage ratio: {avg_ratio:.2f}x")
    print(f"All passed (≤1.5×): {all_pass}")
    
    if all_pass:
        print("\n✓ AUTO-ARCHITECTURE SELECTION VALIDATED")
    
    return df


if __name__ == "__main__":
    run_validation()