"""Public chronological Ψ-Vortex API."""
__version__ = "1.0.0"
from .architecture import ArchitectureCandidate, ArchitectureScore, select_architecture
from .baselines import PIKAN, SINDyRegressor, StaticMLP, VanillaLSTM, train_static_model
from .bic import GloballyClusteredModel, materialize_weight_clusters, select_cluster_count
from .contracts import SequenceBatch, require_chronological, require_sequence
from .data import Trajectory, contiguous_chunks, iter_batches, make_windows
from .datasets import (
    ThermalSample,
    load_3d_thermal_csv,
    load_measured_cycles,
    load_printed_memristor,
    split_printed_memristor_sources,
    split_measured_cycles,
    thermal_split,
    thermal_trajectory,
)
from .export import compression_report, export_torchscript
from .initialization import physics_aware_initialize, random_xavier_initialize, symmetry_orthogonal_initialize
from .latent import CouplingEstimate, aggregate_estimates, fit_coupling, relative_error
from .metrics import correlation, mse, nrmse, r2_score
from .models import (
    LowRankRecurrentStudent,
    MatrixLSTMBlock,
    PsiXLSTMTeacher,
    RecurrentStudent,
    RecurrentTeacher,
    ScalarLSTMBlock,
)
from .pipeline import EndToEndPipeline, assert_source_disjoint
from .rrad import RRADLoss
from .scalable_bic import compare_estimators, minibatch_effective_dof, rff_effective_dof
from .symmetry import AutoSymmetryDetector, SymmetryResult, apply_auto_physics_initialization
from .trainer import SequenceTrainer, VortexTrainer, evaluate
from .verilog_a import compile_openvaf, generate_verilog_a, ngspice_version, simulate_osdi

__all__ = [
    "ArchitectureCandidate",
    "ArchitectureScore",
    "AutoSymmetryDetector",
    "CouplingEstimate",
    "EndToEndPipeline",
    "GloballyClusteredModel",
    "LowRankRecurrentStudent",
    "MatrixLSTMBlock",
    "PIKAN",
    "PsiXLSTMTeacher",
    "RRADLoss",
    "RecurrentStudent",
    "RecurrentTeacher",
    "SINDyRegressor",
    "ScalarLSTMBlock",
    "SequenceBatch",
    "SequenceTrainer",
    "StaticMLP",
    "SymmetryResult",
    "ThermalSample",
    "Trajectory",
    "VanillaLSTM",
    "VortexTrainer",
    "aggregate_estimates",
    "apply_auto_physics_initialization",
    "assert_source_disjoint",
    "compare_estimators",
    "compile_openvaf",
    "compression_report",
    "contiguous_chunks",
    "correlation",
    "evaluate",
    "export_torchscript",
    "fit_coupling",
    "generate_verilog_a",
    "iter_batches",
    "load_3d_thermal_csv",
    "load_measured_cycles",
    "load_printed_memristor",
    "make_windows",
    "materialize_weight_clusters",
    "minibatch_effective_dof",
    "mse",
    "ngspice_version",
    "nrmse",
    "physics_aware_initialize",
    "random_xavier_initialize",
    "r2_score",
    "relative_error",
    "require_chronological",
    "require_sequence",
    "rff_effective_dof",
    "select_architecture",
    "select_cluster_count",
    "simulate_osdi",
    "split_measured_cycles",
    "split_printed_memristor_sources",
    "thermal_split",
    "thermal_trajectory",
    "train_static_model",
    "symmetry_orthogonal_initialize",
]
