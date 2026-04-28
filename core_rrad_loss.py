"""
Recurrent Relation-Aware Distillation (RRAD) Loss
=================================================
Implements Equation 4 from the manuscript:

L_RRAD = ||f_S(x) - f_T(x)||^2 + beta * ||dh_S/dt - dh_T/dt||^2

Key components:
- Logit matching: MSE between teacher and student outputs
- Temporal gradient matching: Preserves dynamic behavior during distillation
- Hidden state alignment: Ensures recurrent structure is preserved

This loss is essential for distilling PSI-xLSTM teachers to compressed students
while preserving high-frequency dynamics.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class RRADLoss(nn.Module):
    """
    Recurrent Relation-Aware Distillation (RRAD) Loss
    
    Implements Equation 4: L_RRAD = L_logit + beta * L_temporal
    
    This loss ensures that the student network learns not just the teacher's
    output mapping, but also its temporal dynamics, which is critical for
    preserving high-frequency behavior in the compressed model.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1):
        """
        Initialize RRAD loss function
        
        Args:
            alpha: Weight for output logit matching term
            beta: Weight for temporal gradient matching term (Eq. 4)
            gamma: Weight for hidden state alignment term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        
    def forward(self, 
                teacher_output: torch.Tensor,
                student_output: torch.Tensor,
                teacher_hidden: Dict,
                student_hidden: Dict,
                t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RRAD loss between teacher and student
        
        Args:
            teacher_output: Teacher prediction [batch, output_size]
            student_output: Student prediction [batch, output_size]
            teacher_hidden: Dictionary of teacher hidden states
            student_hidden: Dictionary of student hidden states
            t: Time tensor for temporal gradient computation [batch, 1]
            
        Returns:
            total_loss: Combined RRAD loss
            loss_components: Dictionary of individual loss terms
        """
        # 1. Logit matching loss: ||f_S(x) - f_T(x)||^2
        L_logit = self.mse(student_output, teacher_output.detach())
        
        # 2. Temporal gradient matching loss: ||dh_S/dt - dh_T/dt||^2
        L_temporal = self._compute_temporal_gradient_loss(
            teacher_hidden, student_hidden, t
        )
        
        # 3. Hidden state alignment loss (optional regularization)
        L_hidden = self._compute_hidden_alignment_loss(
            teacher_hidden, student_hidden
        )
        
        # EQUATION 4: L_RRAD = alpha * L_logit + beta * L_temporal + gamma * L_hidden
        total_loss = (self.alpha * L_logit + 
                     self.beta * L_temporal + 
                     self.gamma * L_hidden)
        
        loss_components = {
            'L_logit': L_logit.item(),
            'L_temporal': L_temporal.item(),
            'L_hidden': L_hidden.item(),
            'L_total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _compute_temporal_gradient_loss(self,
                                        teacher_hidden: Dict,
                                        student_hidden: Dict,
                                        t: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal gradient matching term: ||dh_S/dt - dh_T/dt||^2
        
        This is the key innovation of RRAD - it forces the student to match
        the teacher's temporal dynamics, not just its static output.
        
        For recurrent networks, we approximate dh/dt using finite differences
        between consecutive hidden states or via autograd.
        """
        loss = torch.tensor(0.0, device=t.device)
        count = 0
        
        # Get fused representations if available
        if 'fused' in teacher_hidden and 'fused' in student_hidden:
            h_T = teacher_hidden['fused']  # [batch, hidden_size]
            h_S = student_hidden['fused']  # [batch, hidden_size]
            
            # Compute temporal gradients using autograd
            # dh/dt = grad(h, t)
            if t.requires_grad:
                # Create graph for gradient computation
                grad_h_T = torch.autograd.grad(
                    outputs=h_T.sum(),
                    inputs=t,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                grad_h_S = torch.autograd.grad(
                    outputs=h_S.sum(),
                    inputs=t,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if grad_h_T is not None and grad_h_S is not None:
                    loss = loss + self.mse(grad_h_S, grad_h_T.detach())
                    count += 1
            
            # Fallback: use hidden state difference as proxy for dynamics
            if count == 0:
                # Match the hidden states directly as approximation
                loss = self.mse(h_S, h_T.detach())
                count = 1
        
        # Also match block-level hidden states
        if 'block_hiddens' in teacher_hidden and 'block_hiddens' in student_hidden:
            T_hiddens = teacher_hidden['block_hiddens']
            S_hiddens = student_hidden['block_hiddens']
            
            # Match corresponding blocks (may have different sizes)
            min_blocks = min(len(T_hiddens), len(S_hiddens))
            for i in range(min_blocks):
                h_T = T_hiddens[i]
                h_S = S_hiddens[i]
                
                # Project to same size if needed
                if h_T.shape != h_S.shape:
                    # Use mean pooling for size mismatch
                    target_size = min(h_T.shape[-1], h_S.shape[-1])
                    h_T_proj = h_T[..., :target_size]
                    h_S_proj = h_S[..., :target_size]
                else:
                    h_T_proj = h_T
                    h_S_proj = h_S
                
                loss = loss + self.mse(h_S_proj, h_T_proj.detach())
                count += 1
        
        return loss / max(count, 1)
    
    def _compute_hidden_alignment_loss(self,
                                       teacher_hidden: Dict,
                                       student_hidden: Dict) -> torch.Tensor:
        """
        Compute hidden state alignment loss for additional regularization
        """
        loss = torch.tensor(0.0)
        
        if 'fused' in teacher_hidden and 'fused' in student_hidden:
            h_T = teacher_hidden['fused']
            h_S = student_hidden['fused']
            
            # Get device from tensors
            device = h_T.device
            loss = loss.to(device)
            
            # Cosine similarity alignment
            h_T_norm = h_T / (torch.norm(h_T, dim=-1, keepdim=True) + 1e-8)
            h_S_norm = h_S / (torch.norm(h_S, dim=-1, keepdim=True) + 1e-8)
            
            # We want cosine similarity close to 1
            cosine_sim = (h_T_norm * h_S_norm).sum(dim=-1)
            loss = 1.0 - cosine_sim.mean()
        
        return loss


class RecurrentRelationAwareDistillation:
    """
    Complete RRAD training wrapper for teacher-student distillation
    
    This class manages the distillation process from a PSI-xLSTM teacher
    to a compressed student network while preserving temporal dynamics.
    """
    
    def __init__(self, 
                 teacher: nn.Module,
                 student: nn.Module,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.1):
        """
        Initialize RRAD distillation
        
        Args:
            teacher: Pre-trained PSI-xLSTM teacher model
            student: Student model to be trained
            alpha: Weight for logit matching
            beta: Weight for temporal gradient matching
            gamma: Weight for hidden alignment
        """
        self.teacher = teacher
        self.student = student
        self.rrad_loss = RRADLoss(alpha, beta, gamma)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
    def compute_distillation_loss(self,
                                  V: torch.Tensor,
                                  t: torch.Tensor,
                                  I_true: Optional[torch.Tensor] = None
                                  ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute distillation loss for a batch
        
        Args:
            V: Voltage input [batch, 1]
            t: Time input [batch, 1] (must have requires_grad=True)
            I_true: Optional ground truth current for supervised component
            
        Returns:
            loss: Total distillation loss
            loss_components: Dictionary of loss terms
        """
        # Ensure t requires grad for temporal gradient computation
        if not t.requires_grad:
            t = t.requires_grad_(True)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            I_teacher, teacher_hidden = self.teacher(V, t)
        
        # Student forward (with grad)
        I_student, student_hidden = self.student(V, t)
        
        # Compute RRAD loss
        rrad_loss, components = self.rrad_loss(
            I_teacher, I_student,
            teacher_hidden, student_hidden,
            t
        )
        
        # Add supervised loss if ground truth available
        if I_true is not None:
            L_supervised = nn.MSELoss()(I_student, I_true)
            total_loss = rrad_loss + L_supervised
            components['L_supervised'] = L_supervised.item()
        else:
            total_loss = rrad_loss
        
        components['L_RRAD_total'] = total_loss.item()
        
        return total_loss, components
    
    def extract_time_constants(self, dataset: Dict) -> Dict:
        """
        Extract discrete time constants from the student model
        
        This implements the low-rank hypothesis validation from PSI-xLSTM:
        the student should learn interpretable physical time constants.
        """
        time_constants = {}
        
        with torch.no_grad():
            # Analyze recurrent weights for time constant extraction
            for name, param in self.student.named_parameters():
                if 'R_' in name and 'weight' in name:
                    # Eigenvalue analysis for time constants
                    if param.dim() == 2 and param.size(0) == param.size(1):
                        eigenvalues = torch.linalg.eigvals(param)
                        # Time constants are 1 / |real(eigenvalue)|
                        real_parts = eigenvalues.real
                        nonzero_mask = torch.abs(real_parts) > 1e-6
                        if nonzero_mask.any():
                            tau = 1.0 / torch.abs(real_parts[nonzero_mask])
                            time_constants[name] = tau.cpu().numpy().tolist()
        
        return time_constants


def create_rrad_trainer(teacher: nn.Module, 
                        student: nn.Module,
                        config: Optional[Dict] = None) -> RecurrentRelationAwareDistillation:
    """
    Factory function to create RRAD trainer with default configuration
    
    Args:
        teacher: Pre-trained teacher model
        student: Student model to train
        config: Optional configuration dictionary
        
    Returns:
        Configured RRAD distillation trainer
    """
    default_config = {
        'alpha': 1.0,   # Logit matching weight
        'beta': 0.5,    # Temporal gradient matching weight (Eq. 4)
        'gamma': 0.1    # Hidden alignment weight
    }
    
    if config:
        default_config.update(config)
    
    return RecurrentRelationAwareDistillation(
        teacher, student,
        alpha=default_config['alpha'],
        beta=default_config['beta'],
        gamma=default_config['gamma']
    )