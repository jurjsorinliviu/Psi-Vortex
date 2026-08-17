"""Single source of truth for every retained Ψ-Vortex experiment group."""
from __future__ import annotations

from collections import OrderedDict

from .deployment import circuit_validation, export_validation
from .measured import (
    compression_fidelity,
    cross_device,
    cross_rate,
    measured_baselines,
    measured_fidelity,
)
from .reporting import figures_and_tables, statistical_summary
from .structure import (
    architecture_search,
    automatic_symmetry,
    bic_structural_ablation,
    dataset_integrity,
    frequency_response,
    learning_rate_sensitivity,
    long_sequence,
    runtime_benchmark,
    scalable_bic,
)
from .synthetic import (
    artifact_stress,
    baseline_comparison,
    detection_regime,
    geometry_transfer,
    initialization_ablation,
    latent_recovery,
    multilayer,
    negative_controls,
    recurrence_controls,
    rrad_ablation,
)


EXPERIMENTS = OrderedDict(
    [
        ("dataset_integrity", dataset_integrity),
        ("latent_recovery", latent_recovery),
        ("recurrence_controls", recurrence_controls),
        ("negative_controls", negative_controls),
        ("synthetic_baselines", baseline_comparison),
        ("rrad_ablation", rrad_ablation),
        ("initialization_ablation", initialization_ablation),
        ("bic_structural_ablation", bic_structural_ablation),
        ("architecture_search", architecture_search),
        ("automatic_symmetry", automatic_symmetry),
        ("detection_regime", detection_regime),
        ("artifact_stress", artifact_stress),
        ("geometry_transfer", geometry_transfer),
        ("multilayer", multilayer),
        ("measured_fidelity", measured_fidelity),
        ("measured_baselines", measured_baselines),
        ("cross_device", cross_device),
        ("cross_rate", cross_rate),
        ("compression_fidelity", compression_fidelity),
        ("frequency_response", frequency_response),
        ("learning_rate_sensitivity", learning_rate_sensitivity),
        ("long_sequence", long_sequence),
        ("scalable_bic", scalable_bic),
        ("runtime_benchmark", runtime_benchmark),
        ("export_validation", export_validation),
        ("circuit_validation", circuit_validation),
        ("statistical_summary", statistical_summary),
        ("figures_and_tables", figures_and_tables),
    ]
)
