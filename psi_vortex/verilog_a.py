"""Behavioral Verilog-A export for audited recurrent students.

The generated module advances one recurrent step per ``sample_period``. It is a separate
deployment target from TorchScript and must be compiled/simulated before a circuit claim
is recorded.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import torch
from torch import nn

from .models import LowRankRecurrentStudent, RecurrentStudent


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean_tool_output(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).strip()


def _number(value: float) -> str:
    return f"{float(value):.17g}"


def _affine_expression(
    weights: np.ndarray, bias: float, input_names: list[str]
) -> str:
    pieces = [_number(bias)]
    pieces.extend(f"({_number(weight)})*({name})" for weight, name in zip(weights, input_names))
    return " + ".join(pieces)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.bake() if hasattr(model, "bake") else model


def generate_verilog_a(
    model: nn.Module,
    destination: str | Path,
    *,
    module_name: str = "psi_vortex_student",
    sample_period: float = 1e-6,
) -> Path:
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    model = _unwrap(model).cpu().eval()
    if isinstance(model, RecurrentStudent):
        if model.num_layers != 1:
            raise ValueError("Verilog-A export currently requires a one-layer GRU")
        text = _generate_gru(model, module_name, sample_period)
    elif isinstance(model, LowRankRecurrentStudent):
        text = _generate_low_rank(model, module_name, sample_period)
    else:
        raise TypeError("Verilog-A export supports GRU and low-rank recurrent students")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _header(
    module_name: str, inputs: int, outputs: int
) -> tuple[list[str], list[str], list[str], str]:
    input_ports = [f"in_{index}" for index in range(inputs)]
    output_ports = [f"out_{index}" for index in range(outputs)]
    clock_port = "sample_clock"
    all_inputs = input_ports + [clock_port]
    lines = [
        "`include \"constants.vams\"",
        "`include \"disciplines.vams\"",
        f"module {module_name}({', '.join(all_inputs + output_ports)});",
        f"  input {', '.join(all_inputs)};",
        f"  output {', '.join(output_ports)};",
        f"  electrical {', '.join(all_inputs + output_ports)};",
    ]
    return lines, input_ports, output_ports, clock_port


def _generate_gru(model: RecurrentStudent, module_name: str, sample_period: float) -> str:
    lines, input_ports, output_ports, clock_port = _header(
        module_name, model.input_size, model.output_size
    )
    hidden = model.hidden_size
    input_names = [f"V({name})" for name in input_ports]
    state_nodes = [f"state_{index}" for index in range(hidden)]
    candidate_nodes = [f"candidate_{index}" for index in range(hidden)]
    hidden_names = [f"V({name})" for name in state_nodes]
    lines.append(f"  electrical {', '.join(state_nodes + candidate_nodes)};")
    for prefix in ("r", "z", "n", "next_h"):
        for index in range(hidden):
            lines.append(f"  real {prefix}_{index};")
    for index in range(model.output_size):
        lines.append(f"  real y_{index};")
    weight_ih = model.recurrent.weight_ih_l0.detach().numpy()
    weight_hh = model.recurrent.weight_hh_l0.detach().numpy()
    bias_ih = model.recurrent.bias_ih_l0.detach().numpy()
    bias_hh = model.recurrent.bias_hh_l0.detach().numpy()
    readout_weight = model.readout.weight.detach().numpy()
    readout_bias = model.readout.bias.detach().numpy()
    capacitance = 1e-12
    conductance = 50.0 * capacitance / sample_period
    lines.extend(
        [
            "  analog begin",
            "    // Two-phase sample-and-hold realizes one chronological recurrent step.",
            "    // Capacitive state is reset to zero by each independent transient run.",
        ]
    )
    for state, candidate in zip(state_nodes, candidate_nodes):
        lines.append(f"    I({state}) <+ {_number(capacitance)}*ddt(V({state}));")
        lines.append(
            f"    I({candidate}) <+ {_number(capacitance)}*ddt(V({candidate}));"
        )
    all_names = input_names + hidden_names
    for gate_index, prefix in enumerate(("r", "z")):
        for unit in range(hidden):
            row = gate_index * hidden + unit
            weights = np.concatenate((weight_ih[row], weight_hh[row]))
            bias = bias_ih[row] + bias_hh[row]
            affine = _affine_expression(weights, bias, all_names)
            lines.append(f"    {prefix}_{unit} = 1.0/(1.0 + exp(-({affine})));" )
    for unit in range(hidden):
        row = 2 * hidden + unit
        input_affine = _affine_expression(weight_ih[row], bias_ih[row], input_names)
        recurrent_affine = _affine_expression(weight_hh[row], bias_hh[row], hidden_names)
        lines.append(
            f"    n_{unit} = tanh(({input_affine}) + r_{unit}*({recurrent_affine}));"
        )
        lines.append(
            f"    next_h_{unit} = (1.0-z_{unit})*n_{unit} + "
            f"z_{unit}*V({state_nodes[unit]});"
        )
    lines.append(f"    if (V({clock_port}) < 0.5) begin")
    for unit, candidate in enumerate(candidate_nodes):
        lines.append(
            f"      I({candidate}) <+ {_number(conductance)}*"
            f"(V({candidate})-next_h_{unit});"
        )
    lines.append("    end else begin")
    for state, candidate in zip(state_nodes, candidate_nodes):
        lines.append(
            f"      I({state}) <+ {_number(conductance)}*"
            f"(V({state})-V({candidate}));"
        )
    lines.append("    end")
    for output in range(model.output_size):
        expression = _affine_expression(
            readout_weight[output], readout_bias[output],
            hidden_names,
        )
        lines.append(f"    y_{output} = {expression};")
    lines.extend(f"    V({port}) <+ y_{index};" for index, port in enumerate(output_ports))
    lines.extend(["  end", "endmodule", ""])
    return "\n".join(lines)


def _generate_low_rank(
    model: LowRankRecurrentStudent, module_name: str, sample_period: float
) -> str:
    lines, input_ports, output_ports, clock_port = _header(
        module_name, model.input_size, model.output_size
    )
    state_nodes = [f"state_{index}" for index in range(model.hidden_size)]
    candidate_nodes = [f"candidate_{index}" for index in range(model.hidden_size)]
    hidden_names = [f"V({name})" for name in state_nodes]
    input_names = [f"V({name})" for name in input_ports]
    lines.append(f"  electrical {', '.join(state_nodes + candidate_nodes)};")
    for index in range(model.rank):
        lines.append(f"  real latent_{index};")
    for index in range(model.hidden_size):
        lines.append(f"  real next_h_{index};")
    for index in range(model.output_size):
        lines.append(f"  real y_{index};")
    encode_weight = model.encode.weight.detach().numpy()
    encode_bias = model.encode.bias.detach().numpy()
    decode_weight = model.decode.weight.detach().numpy()
    readout_weight = model.readout.weight.detach().numpy()
    readout_bias = model.readout.bias.detach().numpy()
    capacitance = 1e-12
    conductance = 50.0 * capacitance / sample_period
    lines.extend(
        [
            "  analog begin",
            "    // Two-phase sample-and-hold realizes one chronological recurrent step.",
            "    // Capacitive state is reset to zero by each independent transient run.",
        ]
    )
    for state, candidate in zip(state_nodes, candidate_nodes):
        lines.append(f"    I({state}) <+ {_number(capacitance)}*ddt(V({state}));")
        lines.append(
            f"    I({candidate}) <+ {_number(capacitance)}*ddt(V({candidate}));"
        )
    combined = input_names + hidden_names
    for index in range(model.rank):
        expression = _affine_expression(encode_weight[index], encode_bias[index], combined)
        lines.append(f"    latent_{index} = {expression};")
    for index in range(model.hidden_size):
        expression = _affine_expression(
            decode_weight[index], 0.0, [f"latent_{j}" for j in range(model.rank)]
        )
        lines.append(f"    next_h_{index} = tanh({expression});")
    lines.append(f"    if (V({clock_port}) < 0.5) begin")
    for unit, candidate in enumerate(candidate_nodes):
        lines.append(
            f"      I({candidate}) <+ {_number(conductance)}*"
            f"(V({candidate})-next_h_{unit});"
        )
    lines.append("    end else begin")
    for state, candidate in zip(state_nodes, candidate_nodes):
        lines.append(
            f"      I({state}) <+ {_number(conductance)}*"
            f"(V({state})-V({candidate}));"
        )
    lines.append("    end")
    for output in range(model.output_size):
        expression = _affine_expression(
            readout_weight[output], readout_bias[output],
            hidden_names,
        )
        lines.append(f"    y_{output} = {expression};")
    lines.extend(f"    V({port}) <+ y_{index};" for index, port in enumerate(output_ports))
    lines.extend(["  end", "endmodule", ""])
    return "\n".join(lines)


def compile_openvaf(source: str | Path, destination: str | Path | None = None) -> dict[str, object]:
    configured = os.environ.get("PSI_VORTEX_OPENVAF")
    if configured:
        executable = shutil.which(configured)
        if executable is None:
            return {
                "status": "skipped",
                "reason": f"configured OpenVAF executable not found: {configured}",
                "executable": None,
                "version": None,
            }
    else:
        executable = shutil.which("openvaf") or shutil.which("openvaf-r")
    if executable is None:
        return {
            "status": "skipped",
            "reason": "OpenVAF executable not found; set PSI_VORTEX_OPENVAF or PATH",
            "executable": None,
            "version": None,
        }
    source = Path(source)
    destination = Path(destination) if destination is not None else source.with_suffix(".osdi")
    version_result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    version = _clean_tool_output(version_result.stdout or version_result.stderr)
    command = [executable, str(source), "-o", str(destination)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return {
        "status": "passed" if completed.returncode == 0 and destination.exists() else "failed",
        "executable": executable,
        "version": version,
        "command": command,
        "returncode": completed.returncode,
        "stdout": _clean_tool_output(completed.stdout),
        "stderr": _clean_tool_output(completed.stderr),
        "artifact": str(destination) if destination.exists() else None,
    }


def ngspice_version() -> dict[str, object]:
    executable = shutil.which("ngspice_con") or shutil.which("ngspice")
    if executable is None:
        return {"status": "skipped", "reason": "ngspice executable not found"}
    completed = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    version = _clean_tool_output(completed.stdout or completed.stderr)
    version_match = re.search(r"ngspice-[0-9.]+", version, flags=re.IGNORECASE)
    version_label = version_match.group(0) if version_match else (version.splitlines()[0] if version else "")
    return {
        "status": "available" if completed.returncode == 0 else "failed",
        "executable": executable,
        "version": version,
        "version_label": version_label,
    }


def simulate_osdi(
    osdi_artifact: str | Path,
    module_name: str,
    inputs: np.ndarray,
    *,
    sample_period: float,
    work_directory: str | Path,
) -> dict[str, object]:
    """Run a compiled module in ngspice and return the emitted output waveform."""
    executable = shutil.which("ngspice_con") or shutil.which("ngspice")
    if executable is None:
        return {"status": "skipped", "reason": "ngspice executable not found"}
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("OSDI simulation inputs must be [timesteps, features] with length > 1")
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    output_path = work / "osdi_output.txt"
    netlist_path = work / "osdi_validation.cir"
    osdi_reference = os.path.relpath(Path(osdi_artifact).resolve(), work.resolve()).replace(
        "\\", "/"
    )
    edge_epsilon = sample_period * 1e-6
    stop = len(values) * sample_period
    sources = []
    for feature in range(values.shape[1]):
        held_points: list[tuple[float, float]] = []
        for index in range(len(values)):
            start = index * sample_period
            end = (index + 1) * sample_period
            held_points.append((start, values[index, feature]))
            held_points.append((end - edge_epsilon, values[index, feature]))
        held_points.append((stop, values[-1, feature]))
        pairs = " ".join(f"{time:.17g} {value:.17g}" for time, value in held_points)
        sources.append(f"V{feature} in_{feature} 0 PWL({pairs})")
    clock_points: list[tuple[float, float]] = [(0.0, 0.0)]
    for index in range(len(values)):
        half = (index + 0.5) * sample_period
        end = (index + 1) * sample_period
        clock_points.extend(
            [
                (half - edge_epsilon, 0.0),
                (half, 1.0),
                (end - edge_epsilon, 1.0),
                (end, 0.0),
            ]
        )
    clock_pairs = " ".join(
        f"{time:.17g} {value:.17g}" for time, value in clock_points
    )
    sources.append(f"Vclock sample_clock 0 PWL({clock_pairs})")
    # The exporter currently emits one or more output ports; the reproducibility
    # validation uses output zero and extends naturally when additional ports are needed.
    ports = " ".join(
        [f"in_{index}" for index in range(values.shape[1])]
        + ["sample_clock", "out_0"]
    )
    netlist = "\n".join(
        [
            "* generated Ψ-Vortex OSDI validation",
            *sources,
            f"N1 {ports} dut",
            f".model dut {module_name}",
            ".control",
            f"pre_osdi {osdi_reference}",
            f"tran {sample_period / 20:.17g} {stop:.17g} uic",
            f"wrdata {output_path.name} v(out_0)",
            "quit",
            ".endc",
            ".end",
            "",
        ]
    )
    netlist_path.write_text(netlist, encoding="utf-8")
    completed = subprocess.run(
        [executable, "-b", netlist_path.name],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=work,
    )
    if completed.returncode != 0 or not output_path.exists():
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": _clean_tool_output(completed.stdout),
            "stderr": _clean_tool_output(completed.stderr),
            "netlist": str(netlist_path),
        }
    waveform = np.loadtxt(output_path)
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    sample_times = (np.arange(len(values), dtype=float) + 1.0) * sample_period
    sample_times -= edge_epsilon * 2.0
    return {
        "status": "passed",
        "returncode": completed.returncode,
        "times": waveform[:, 0].tolist(),
        "outputs": waveform[:, -1].tolist(),
        "sample_times": sample_times.tolist(),
        "netlist": str(netlist_path),
    }
