"""TorchScript, Verilog-A, OpenVAF, and ngspice deployment experiments."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from psi_vortex import (
    compile_openvaf,
    compression_report,
    export_torchscript,
    generate_verilog_a,
    ngspice_version,
    simulate_osdi,
)

from .common import RunContext, predict, source_record
from .train import fit_pipeline, make_pipeline, thermal_data


def _stream_loaded(loaded, features: torch.Tensor) -> torch.Tensor:
    state = loaded.initial_state(features.shape[0])
    pieces = []
    for index in range(features.shape[1]):
        output, state = loaded.step(features[:, index], state)
        pieces.append(output.unsqueeze(1))
    return torch.cat(pieces, dim=1)


def export_validation(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    seed = config.get("deployment_seed", config["seeds"][0])
    train, validation, test, _ = thermal_data(config, 0.08)
    rows: list[dict[str, Any]] = []
    for student_type in ("gru", "low_rank"):
        pipeline = make_pipeline(
            config,
            seed,
            student_type=student_type,
            student_rank=config["student_rank"],
        )
        fit_pipeline(pipeline, train, validation, test, config)
        example = test[0].features.unsqueeze(0)
        destination = context.output / "artifacts" / f"{student_type}_student.pt"
        export_torchscript(pipeline.student, example, destination)
        loaded = torch.jit.load(str(destination))
        expected = torch.tensor(predict(pipeline.student, test[0])).unsqueeze(0)
        batch_output = loaded(example)
        streaming_output = _stream_loaded(loaded, example)
        guard_passed = False
        try:
            loaded(example[:, :1])
        except (RuntimeError, torch.jit.Error):
            guard_passed = True
        rows.append(
            {
                "student_type": student_type,
                "seed": seed,
                "device": config["device"],
                "cluster_count": pipeline.selected_cluster_count,
                "batch_max_abs_error": float((batch_output - expected).abs().max()),
                "streaming_max_abs_error": float((streaming_output - expected).abs().max()),
                "length_one_guard": guard_passed,
                **compression_report(pipeline.student, destination),
                **context.checkpoint(
                    pipeline.student, f"export_{student_type}_seed{seed}"
                ),
                **source_record(train, validation, test),
                "profile": config["profile"],
            }
        )
    return rows


def circuit_validation(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    seed = config.get("deployment_seed", config["seeds"][0])
    train, validation, test, _ = thermal_data(config, 0.08)
    pipeline = make_pipeline(config, seed)
    fit_pipeline(pipeline, train, validation, test, config)
    checkpoint = context.checkpoint(
        pipeline.student,
        f"circuit_seed{seed}",
    )
    module_name = "psi_vortex_student"
    source = context.output / "artifacts" / f"{module_name}.va"
    generate_verilog_a(
        pipeline.student,
        source,
        module_name=module_name,
        sample_period=config["deployment_sample_period"],
    )
    compiled = compile_openvaf(source)
    simulator = ngspice_version()
    compiler_record = {
        "openvaf_status": compiled["status"],
        "openvaf_executable": compiled.get("executable"),
        "openvaf_version": compiled.get("version"),
        "openvaf_returncode": compiled.get("returncode"),
        "openvaf_command": compiled.get("command"),
        "openvaf_stdout": compiled.get("stdout"),
        "openvaf_stderr": compiled.get("stderr"),
        "openvaf_artifact": compiled.get("artifact"),
        "openvaf_skip_reason": compiled.get("reason"),
    }
    simulator_record = {
        "ngspice_status": simulator["status"],
        "ngspice_executable": simulator.get("executable"),
        "ngspice_version": simulator.get("version"),
        "ngspice_version_label": simulator.get("version_label"),
        "ngspice_skip_reason": simulator.get("reason"),
    }
    base = test[0].features.detach().cpu().numpy()
    length = len(base)
    sinusoid = np.sin(np.linspace(0, 4 * np.pi, length))[:, None] + 1.0
    dc = np.full_like(base, 0.2)
    ood = np.zeros_like(base)
    ood[length // 3 : 2 * length // 3] = 3.0
    stimuli = {
        "dc_read": dc,
        "pulse_train": base,
        "sinusoid": sinusoid.astype(base.dtype),
        "crosstalk_held_out": (
            test[1].features.detach().cpu().numpy() if len(test) > 1 else base.copy()
        ),
        "ood_pulse": ood,
    }
    rows: list[dict[str, Any]] = []
    for stimulus, values_in in stimuli.items():
        waveform_error = None
        waveform_mae = None
        if compiled["status"] == "passed":
            simulation = simulate_osdi(
                compiled["artifact"],
                module_name,
                values_in,
                sample_period=config["deployment_sample_period"],
                work_directory=context.output / "artifacts" / "ngspice" / stimulus,
            )
            if simulation["status"] == "passed":
                with torch.no_grad():
                    expected = pipeline.student(
                        torch.tensor(values_in, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(config["device"]),
                        None,
                    )[0][0, :, 0].cpu().numpy()
                times = np.asarray(simulation["times"])
                outputs = np.asarray(simulation["outputs"])
                sample_times = np.asarray(simulation["sample_times"])
                sampled = np.interp(sample_times, times, outputs)
                waveform_error = float(np.max(np.abs(sampled - expected)))
                waveform_mae = float(np.mean(np.abs(sampled - expected)))
        else:
            simulation = {
                "status": "skipped",
                "reason": "compiled OSDI artifact unavailable",
            }
        simulation_record = {
            "osdi_simulation_status": simulation["status"],
            "osdi_returncode": simulation.get("returncode"),
            "osdi_stdout": simulation.get("stdout"),
            "osdi_stderr": simulation.get("stderr"),
            "osdi_netlist": simulation.get("netlist"),
            "osdi_skip_reason": simulation.get("reason"),
        }
        rows.append(
            {
            "stimulus": stimulus,
            "seed": seed,
            "device": config["device"],
            "cluster_count": pipeline.selected_cluster_count,
            "verilog_a_status": "generated",
            "verilog_a_artifact": str(source.relative_to(context.output)),
            **compiler_record,
            **simulator_record,
            **simulation_record,
            "osdi_max_abs_error": waveform_error,
            "osdi_mean_abs_error": waveform_mae,
            "circuit_claim_valid": bool(
                compiled["status"] == "passed"
                and simulation["status"] == "passed"
                and waveform_error is not None
                and waveform_error <= config.get("deployment_max_abs_tolerance", 1e-3)
            ),
            "max_abs_tolerance": config.get("deployment_max_abs_tolerance", 1e-3),
            **checkpoint,
            **source_record(train, validation, test),
            "profile": config["profile"],
            }
        )
    return rows
