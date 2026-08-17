"""Physics-aware recurrent initialization utilities."""
import math
import torch
from torch import nn
from .models import MatrixLSTMBlock,ScalarLSTMBlock


def random_xavier_initialize(model):
    """Random baseline using Xavier matrices and zero biases."""
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            else:
                parameter.zero_()
    return model


def physics_aware_initialize(model,time_constant:float,delta_t:float):
    """Initialize forget dynamics near exp(-Δt/τ), preserving trainability."""
    if time_constant<=0 or delta_t<=0: raise ValueError("time_constant and delta_t must be positive")
    retention=math.exp(-delta_t/time_constant)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,MatrixLSTMBlock): module.W_f.weight.zero_(); module.R_f.weight.zero_(); module.W_f.bias.fill_(math.log(retention/(1-retention)))
            elif isinstance(module,ScalarLSTMBlock):
                size=module.hidden_size; module.cell.bias_ih[size:2*size].fill_(.5*math.log(retention/(1-retention))); module.cell.bias_hh[size:2*size].fill_(.5*math.log(retention/(1-retention)))
    return model


def symmetry_orthogonal_initialize(
    model,
    symmetry_type: str,
    *,
    input_feature: int = 0,
    epsilon: float = 0.01,
    sigma: float = 0.01,
    scale: float = 0.1,
    preserve_recurrence: bool = False,
):
    """Equation-5 orthogonal initialization with an explicit odd/even/identity mask."""
    if symmetry_type not in ("odd", "even", "none"):
        raise ValueError("symmetry_type must be odd, even, or none")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "weight" in name and parameter.ndim >= 2:
                is_recurrent = any(
                    token in name for token in ("R_", "weight_hh", "recurrent")
                )
                if preserve_recurrence and is_recurrent:
                    nn.init.orthogonal_(parameter)
                    continue
                nn.init.orthogonal_(parameter)
                mask = torch.ones_like(parameter)
                is_input_or_output = any(
                    token in name
                    for token in ("W_i", "W_f", "W_o", "W_k", "W_q", "W_v", "weight_ih", "readout")
                )
                if symmetry_type == "odd" and is_input_or_output and input_feature < parameter.shape[1]:
                    mask[parameter.shape[0] // 2 :, input_feature] = -1.0
                if symmetry_type == "even" and is_input_or_output:
                    parameter.abs_()
                parameter.mul_(mask)
                parameter.add_(epsilon * sigma * torch.randn_like(parameter))
                parameter.mul_(scale)
            elif "bias" in name:
                parameter.fill_(0.01)
    return model
