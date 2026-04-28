"""
Ψ-Vortex Unified Two-Phase Training
====================================
Implements the synergistic training strategy from Chapter 3:

Phase 1: Rapid convergence via Physics-Aware Initialization
         - Uses Equation 5 initialization
         - Fast MSE convergence (27 epochs)
         - Achieves 10.14x speedup over baseline

Phase 2: Structural optimization via Adaptive BIC
         - Uses RRAD loss (Equation 4)
         - BIC regularization (Equations 6-7)
         - Additional 100 epochs for structure discovery
         - Achieves negative loss values (-14.9 to -16.9)

Combined: 2.77x speedup over BIC-alone while maintaining structural quality
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Tuple, Optional

# Import Ψ-Vortex components
from core_rrad_loss import RRADLoss, RecurrentRelationAwareDistillation, create_rrad_trainer
from core_adaptive_bic import AdaptiveStructureLoss, DifferentiableBIC, ClusteringStudent
from core_physics_init import apply_psi_vortex_init, PhysicsAwareInitializer


class VortexTrainer:
    """
    Implements Ψ-Vortex two-phase training strategy:
    
    Phase 1: Rapid convergence via Physics-Aware Initialization
             Target: MSE convergence (< 1e-6)
             Duration: ~27 epochs (vs 256 baseline)
             
    Phase 2: Structural optimization via Adaptive BIC
             Objective: Minimize L_Vortex = L_RRAD + λ_struct · R_BIC
             Duration: Additional 100 epochs
             
    This synergistic approach achieves:
    - 10.14x speedup in Phase 1
    - 2.77x speedup over BIC-alone
    - Comparable structural quality (loss: -14.9)
    """
    
    def __init__(self, 
                 lambda_struct: float = 0.01, 
                 gamma: float = 0.1,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 target_mse: float = 1e-6):
        """
        Initialize Vortex trainer
        
        Args:
            lambda_struct: Weight for BIC regularization (Eq. 6)
            gamma: Kernel bandwidth for BIC density estimation (Eq. 7)
            alpha: Weight for logit matching in RRAD (Eq. 4)
            beta: Weight for temporal gradient matching in RRAD (Eq. 4)
            target_mse: MSE threshold for Phase 1 convergence
        """
        self.lambda_struct = lambda_struct
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.target_mse = target_mse
        
        # Initialize adaptive loss
        self.adaptive_loss = AdaptiveStructureLoss(lambda_struct, gamma)
        
        # Training history
        self.training_history = {
            'phase1_loss': [],
            'phase1_mse': [],
            'phase2_loss': [],
            'phase2_mse': [],
            'bic_loss': [],
            'phase1_time': None,
            'phase2_time': None,
            'total_time': None,
            'phase1_epochs': 0,
            'phase2_epochs': 0,
            'optimal_structure': None
        }
    
    def train_vortex(self, 
                     teacher: nn.Module, 
                     student: nn.Module,
                     dataset: Dict, 
                     num_epochs_phase1: int = 50,
                     num_epochs_phase2: int = 100,
                     lr_phase1: float = 1e-3,
                     lr_phase2: float = 1e-4,
                     batch_size: int = 256,
                     device: str = 'cuda') -> Tuple[nn.Module, Dict]:
        """
        Full Ψ-Vortex training pipeline
        
        Args:
            teacher: Pre-trained PSI-xLSTM teacher model
            student: Student model to train
            dataset: Dictionary with 'train' and 'val' splits
            num_epochs_phase1: Max epochs for Phase 1
            num_epochs_phase2: Epochs for Phase 2
            lr_phase1: Learning rate for Phase 1
            lr_phase2: Learning rate for Phase 2
            batch_size: Training batch size
            device: Device to train on
            
        Returns:
            student: Trained student model
            training_history: Dictionary of training metrics
        """
        print("=" * 70)
        print("Starting Ψ-Vortex Unified Training")
        print("=" * 70)
        
        # Ensure models are on correct device
        teacher = teacher.to(device)
        student = student.to(device)
        
        # Freeze teacher
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        
        # =============================================
        # PHASE 1: Rapid MSE convergence
        # =============================================
        print("\n" + "=" * 50)
        print("PHASE 1: Physics-Aware Convergence")
        print("=" * 50)
        
        start_time = time.time()
        student, phase1_history = self._train_phase1(
            student, dataset, num_epochs_phase1, lr_phase1, batch_size, device
        )
        phase1_time = time.time() - start_time
        
        self.training_history['phase1_loss'] = phase1_history['loss']
        self.training_history['phase1_mse'] = phase1_history['mse']
        self.training_history['phase1_time'] = phase1_time
        self.training_history['phase1_epochs'] = phase1_history['epochs']
        
        print(f"\nPhase 1 Complete:")
        print(f"  Time: {phase1_time:.2f}s")
        print(f"  Epochs: {phase1_history['epochs']}")
        print(f"  Final MSE: {phase1_history['mse'][-1]:.2e}")
        
        # =============================================
        # PHASE 2: Structural optimization with BIC
        # =============================================
        print("\n" + "=" * 50)
        print("PHASE 2: Adaptive Structure Discovery")
        print("=" * 50)
        
        start_time = time.time()
        student, phase2_history = self._train_phase2(
            teacher, student, dataset, num_epochs_phase2, lr_phase2, batch_size, device
        )
        phase2_time = time.time() - start_time
        
        self.training_history['phase2_loss'] = phase2_history['loss']
        self.training_history['phase2_mse'] = phase2_history['mse']
        self.training_history['bic_loss'] = phase2_history['bic']
        self.training_history['phase2_time'] = phase2_time
        self.training_history['phase2_epochs'] = num_epochs_phase2
        self.training_history['optimal_structure'] = phase2_history.get('optimal_structure')
        self.training_history['total_time'] = phase1_time + phase2_time
        
        print(f"\nPhase 2 Complete:")
        print(f"  Time: {phase2_time:.2f}s")
        print(f"  Epochs: {num_epochs_phase2}")
        print(f"  Final Loss: {phase2_history['loss'][-1]:.2e}")
        print(f"  Final BIC: {phase2_history['bic'][-1]:.2e}")
        
        # =============================================
        # Summary
        # =============================================
        print("\n" + "=" * 70)
        print("Ψ-VORTEX TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total Time: {self.training_history['total_time']:.2f}s")
        print(f"Total Epochs: {self.training_history['phase1_epochs'] + num_epochs_phase2}")
        print(f"Final Structure: {self.training_history['optimal_structure']}")
        
        return student, self.training_history
    
    def _train_phase1(self, 
                      student: nn.Module, 
                      dataset: Dict, 
                      num_epochs: int,
                      lr: float,
                      batch_size: int,
                      device: str) -> Tuple[nn.Module, Dict]:
        """
        Phase 1: Fast MSE convergence without structural regularization
        
        Uses only data fitting loss for rapid convergence.
        Leverages physics-aware initialization for speedup.
        """
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        loss_history = []
        mse_history = []
        converged_epoch = num_epochs
        
        V_train = dataset['train']['V']
        t_train = dataset['train']['t']
        I_train = dataset['train']['I']
        n_samples = len(V_train)
        
        for epoch in range(num_epochs):
            student.train()
            total_loss = 0
            n_batches = 0
            
            # Mini-batch training
            indices = torch.randperm(n_samples, device=device)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:min(i + batch_size, n_samples)]
                V_batch = V_train[batch_idx]
                t_batch = t_train[batch_idx]
                I_batch = I_train[batch_idx]
                
                optimizer.zero_grad()
                I_pred, _ = student(V_batch, t_batch)
                loss = criterion(I_pred, I_batch)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            loss_history.append(avg_loss)
            mse_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: MSE = {avg_loss:.3e}")
            
            # Check convergence
            if avg_loss < self.target_mse:
                converged_epoch = epoch + 1
                print(f"  -> Converged at epoch {converged_epoch}")
                break
        
        return student, {
            'loss': loss_history,
            'mse': mse_history,
            'epochs': converged_epoch
        }
    
    def _train_phase2(self, 
                      teacher: nn.Module,
                      student: nn.Module, 
                      dataset: Dict, 
                      num_epochs: int,
                      lr: float,
                      batch_size: int,
                      device: str) -> Tuple[nn.Module, Dict]:
        """
        Phase 2: Structural optimization with RRAD + Adaptive BIC
        
        Implements Equation 6: L_Vortex = L_RRAD + λ_struct · R_BIC
        """
        # Initialize RRAD trainer
        rrad = RecurrentRelationAwareDistillation(
            teacher, student, 
            alpha=self.alpha, 
            beta=self.beta, 
            gamma=0.1
        )
        
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        
        loss_history = []
        mse_history = []
        bic_history = []
        
        V_train = dataset['train']['V']
        t_train = dataset['train']['t']
        I_train = dataset['train']['I']
        n_samples = len(V_train)
        
        # First, find optimal structure using BIC
        print("  Finding optimal structure via BIC...")
        optimal_structure = self.adaptive_loss.compute_optimal_structure(student, dataset)
        print(f"  Estimated clusters: {optimal_structure.get('estimated_clusters', 'N/A')}")
        
        # Apply optimal structure if found
        if optimal_structure.get('optimal_structure'):
            opt = optimal_structure['optimal_structure']
            if opt['type'] == 'clusters' and hasattr(student, 'num_clusters'):
                student.num_clusters = opt['value']
                print(f"  Applied optimal cluster count: {opt['value']}")
            elif opt['type'] == 'rank' and hasattr(student, 'rank'):
                student.rank = opt['value']
                print(f"  Applied optimal rank: {opt['value']}")
        
        print("  Starting structural optimization...")
        
        for epoch in range(num_epochs):
            student.train()
            total_loss = 0
            total_mse = 0
            total_bic = 0
            n_batches = 0
            
            indices = torch.randperm(n_samples, device=device)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:min(i + batch_size, n_samples)]
                V_batch = V_train[batch_idx]
                t_batch = t_train[batch_idx].requires_grad_(True)
                I_batch = I_train[batch_idx]
                
                optimizer.zero_grad()
                
                # Compute RRAD loss (Equation 4)
                rrad_loss, rrad_components = rrad.compute_distillation_loss(
                    V_batch, t_batch, I_batch
                )
                
                # Compute MSE for BIC
                with torch.no_grad():
                    I_pred, _ = student(V_batch, t_batch)
                    mse = torch.mean((I_pred - I_batch) ** 2)
                
                # Compute total Vortex loss (Equation 6)
                total_vortex_loss, bic_loss = self.adaptive_loss(
                    student, rrad_loss, mse, len(V_batch)
                )
                
                total_vortex_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                
                total_loss += total_vortex_loss.item()
                total_mse += mse.item()
                total_bic += bic_loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            avg_mse = total_mse / n_batches
            avg_bic = total_bic / n_batches
            
            loss_history.append(avg_loss)
            mse_history.append(avg_mse)
            bic_history.append(avg_bic)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"Loss = {avg_loss:.3e}, MSE = {avg_mse:.3e}, BIC = {avg_bic:.3e}")
        
        return student, {
            'loss': loss_history,
            'mse': mse_history,
            'bic': bic_history,
            'optimal_structure': optimal_structure
        }


def create_vortex_trainer(config: Optional[Dict] = None) -> VortexTrainer:
    """
    Factory function to create VortexTrainer with default configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured VortexTrainer instance
    """
    default_config = {
        'lambda_struct': 0.01,   # BIC weight (Eq. 6)
        'gamma': 0.1,            # BIC kernel bandwidth (Eq. 7)
        'alpha': 1.0,            # RRAD logit weight (Eq. 4)
        'beta': 0.5,             # RRAD temporal weight (Eq. 4)
        'target_mse': 1e-6       # Phase 1 convergence target
    }
    
    if config:
        default_config.update(config)
    
    return VortexTrainer(**default_config)


def run_ablation_experiment(teacher: nn.Module,
                           dataset: Dict,
                           device: str = 'cuda') -> Dict:
    """
    Run ablation study comparing all four configurations:
    1. Baseline (Random Init + L2)
    2. Init-Only (Psi-Init + L2)
    3. BIC-Only (Random Init + Adaptive BIC)
    4. Full Ψ-Vortex (Psi-Init + Adaptive BIC)
    
    Returns dictionary of results for each configuration.
    """
    from core_psi_xlstm import PSI_xLSTM
    
    results = {}
    
    # Import here to avoid circular imports
    configs = [
        ('Baseline', False, False),
        ('Init-Only', True, False),
        ('BIC-Only', False, True),
        ('Psi-Vortex', True, True)
    ]
    
    for name, use_init, use_bic in configs:
        print(f"\n{'='*50}")
        print(f"Running {name} [Init={use_init}, BIC={use_bic}]")
        print(f"{'='*50}")
        
        # Create fresh student
        student = ClusteringStudent(input_size=2, hidden_size=16, output_size=1).to(device)
        
        # Apply initialization if enabled
        if use_init:
            apply_psi_vortex_init(student, pde_type="memristor")
        
        if use_bic:
            # Full two-phase training
            trainer = create_vortex_trainer()
            student, history = trainer.train_vortex(
                teacher, student, dataset,
                num_epochs_phase1=50,
                num_epochs_phase2=100,
                device=device
            )
            results[name] = {
                'time': history['total_time'],
                'epochs': history['phase1_epochs'] + history['phase2_epochs'],
                'final_loss': history['phase2_loss'][-1] if history['phase2_loss'] else None,
                'final_bic': history['bic_loss'][-1] if history['bic_loss'] else None,
                'history': history
            }
        else:
            # Phase 1 only (MSE convergence)
            trainer = VortexTrainer(target_mse=1e-6)
            start_time = time.time()
            student, phase1_history = trainer._train_phase1(
                student, dataset, num_epochs=500, lr=1e-3, batch_size=256, device=device
            )
            elapsed = time.time() - start_time
            
            results[name] = {
                'time': elapsed,
                'epochs': phase1_history['epochs'],
                'final_loss': phase1_history['mse'][-1],
                'final_bic': None,
                'history': phase1_history
            }
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Config':<15} {'Time (s)':<12} {'Epochs':<10} {'Final Loss':<15} {'Final BIC':<15}")
    print("-" * 70)
    
    for name, data in results.items():
        bic_str = f"{data['final_bic']:.2e}" if data['final_bic'] else "N/A"
        print(f"{name:<15} {data['time']:<12.2f} {data['epochs']:<10} "
              f"{data['final_loss']:<15.2e} {bic_str:<15}")
    
    return results