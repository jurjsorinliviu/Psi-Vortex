from __future__ import annotations

import ast
import csv
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.registry import EXPERIMENTS
from experiments.common import RunContext
from experiments.measured import compression_fidelity
from experiments.reporting import _latex_ready, finalize
from experiments.run import run_experiments
from experiments.structure import learning_rate_sensitivity
from experiments.synthetic import baseline_comparison
from psi_vortex.baselines import StaticMLP, train_static_model
from psi_vortex import (
    AutoSymmetryDetector,
    EndToEndPipeline,
    LowRankRecurrentStudent,
    MatrixLSTMBlock,
    PsiXLSTMTeacher,
    RecurrentStudent,
    ScalarLSTMBlock,
    SequenceBatch,
    SequenceTrainer,
    Trajectory,
    contiguous_chunks,
    export_torchscript,
    fit_coupling,
    iter_batches,
    load_printed_memristor,
    make_windows,
    thermal_split,
    split_printed_memristor_sources,
)


def trajectory(name: str = "trajectory", length: int = 8, outputs: int = 1) -> Trajectory:
    time = torch.arange(length, dtype=torch.float32)[:, None]
    target = torch.cat([time * (index + 1) for index in range(outputs)], dim=1)
    return Trajectory(name, torch.cat((time, time.square()), dim=1), target, time, name)


class StateProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.received = []

    def forward(self, value, state=None):
        self.received.append(state)
        previous = value.new_zeros(1, value.shape[0], 1) if state is None else state
        output = value[..., :1] * self.weight + previous.transpose(0, 1)
        return output, output[:, -1:].transpose(0, 1), {"recurrent": output}


class RecurrenceAndScienceTests(unittest.TestCase):
    def test_learning_rate_sensitivity_overrides_both_training_stages(self):
        class ContextProbe:
            config = {
                "profile": "test",
                "device": "cpu",
                "seeds": [7],
                "learning_rate": 5e-4,
                "teacher_learning_rate": 8e-4,
                "student_learning_rate": 1e-3,
                "r2_accept": 0.9,
            }

            @staticmethod
            def checkpoint(model, label):
                return {}

        pipeline = SimpleNamespace(student=object())
        metrics = {"validation": {"mse": 1.0}, "test": {"mse": 1.0}}
        with (
            patch("experiments.structure.thermal_data", return_value=([], [], [], [])),
            patch("experiments.structure.make_pipeline", return_value=pipeline) as make,
            patch("experiments.structure.fit_pipeline", return_value=metrics) as fit,
            patch("experiments.structure.coupling_scores", return_value=[]),
            patch("experiments.structure.mean_estimate", return_value={}),
            patch("experiments.structure.source_record", return_value={}),
        ):
            rows = learning_rate_sensitivity(ContextProbe())

        expected = (1e-4, 1e-3, 1e-2)
        self.assertEqual([row["learning_rate"] for row in rows], list(expected))
        for call, learning_rate in zip(make.call_args_list, expected):
            modified = call.args[0]
            self.assertEqual(modified["learning_rate"], learning_rate)
            self.assertEqual(modified["teacher_learning_rate"], learning_rate)
            self.assertEqual(modified["student_learning_rate"], learning_rate)
        for call, learning_rate in zip(fit.call_args_list, expected):
            modified = call.args[-1]
            self.assertEqual(modified["teacher_learning_rate"], learning_rate)
            self.assertEqual(modified["student_learning_rate"], learning_rate)

    def test_sequence_contract_rejects_nonfinite_values(self):
        features = torch.randn(1, 4, 2)
        features[0, 2, 0] = float("nan")
        with self.assertRaises(ValueError):
            SequenceBatch(
                features,
                torch.randn(1, 4, 1),
                torch.arange(4.0).view(1, 4, 1),
                ("bad",),
            )
        time = torch.arange(4.0).view(1, 4, 1)
        time[0, 2, 0] = float("nan")
        with self.assertRaises(ValueError):
            SequenceBatch(torch.randn(1, 4, 2), torch.randn(1, 4, 1), time, ("bad",))

    def test_declared_seed_controls_static_model_initialization(self):
        items = [trajectory("seeded", 8)]
        first = train_static_model(StaticMLP(2, 4), items, epochs=1, seed=73)
        torch.manual_seed(999)
        second = train_static_model(StaticMLP(2, 4), items, epochs=1, seed=73)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right)

    def test_runner_refuses_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "old-result.txt").write_text("stale", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_experiments("configs/smoke.json", ["latent_recovery"], output=output)

    def test_runner_resume_skips_only_provenance_verified_groups(self):
        config = {
            "profile": "resume-test",
            "device": "cpu",
            "seeds": [0],
            "driver_train_sources": [1],
            "driver_validation_sources": [2],
            "driver_test_sources": [3],
            "teacher_hidden": 2,
            "teacher_blocks": 1,
            "student_hidden": 2,
            "teacher_epochs": 1,
            "student_epochs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "resume.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"

            def probe(_context):
                return [{"value": 1}]

            with patch.dict(EXPERIMENTS, {"resume_probe": probe}, clear=True):
                run_experiments(config_path, ["resume_probe"], output=output)
                manifest = output / "manifests" / "run_resume_probe.json"
                original = manifest.read_bytes()
                with patch.dict(
                    EXPERIMENTS,
                    {"resume_probe": unittest.mock.Mock(side_effect=AssertionError("rerun"))},
                    clear=True,
                ):
                    run_experiments(
                        config_path,
                        ["resume_probe"],
                        output=output,
                        resume=True,
                    )
                self.assertEqual(manifest.read_bytes(), original)
                progress = json.loads(
                    (output / "manifests" / "progress.json").read_text(encoding="utf-8")
                )
                self.assertEqual(progress["status"], "completed")
                self.assertEqual(progress["completed_groups"], ["resume_probe"])
                result = output / "raw_results" / "resume_probe.csv"
                result.write_text("value,execution_device\n2,cpu\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "raw-result hash mismatch"):
                    run_experiments(
                        config_path,
                        ["resume_probe"],
                        output=output,
                        resume=True,
                    )

    def test_runner_resume_rejects_source_tree_mismatch(self):
        config = {
            "profile": "resume-test",
            "device": "cpu",
            "seeds": [0],
            "driver_train_sources": [1],
            "driver_validation_sources": [2],
            "driver_test_sources": [3],
            "teacher_hidden": 2,
            "teacher_blocks": 1,
            "student_hidden": 2,
            "teacher_epochs": 1,
            "student_epochs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "resume.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"
            with patch.dict(
                EXPERIMENTS, {"resume_probe": lambda _context: [{"value": 1}]}, clear=True
            ):
                run_experiments(config_path, ["resume_probe"], output=output)
                runtime_path = output / "environment" / "runtime.json"
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                runtime["source_tree_sha256"] = "0" * 64
                runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source_tree_sha256"):
                    run_experiments(
                        config_path,
                        ["resume_probe"],
                        output=output,
                        resume=True,
                    )

    def test_runner_resume_rejects_corrupted_checkpoint(self):
        config = {
            "profile": "resume-test",
            "device": "cpu",
            "seeds": [0],
            "driver_train_sources": [1],
            "driver_validation_sources": [2],
            "driver_test_sources": [3],
            "teacher_hidden": 2,
            "teacher_blocks": 1,
            "student_hidden": 2,
            "teacher_epochs": 1,
            "student_epochs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "resume.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"

            def checkpoint_probe(context):
                model = RecurrentStudent(2, 2)
                return [{"value": 1, **context.checkpoint(model, "resume-probe")}]

            with patch.dict(
                EXPERIMENTS, {"resume_probe": checkpoint_probe}, clear=True
            ):
                run_experiments(config_path, ["resume_probe"], output=output)
                checkpoint = output / "checkpoints" / "resume-probe.pt"
                checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
                with self.assertRaisesRegex(ValueError, "checkpoint integrity mismatch"):
                    run_experiments(
                        config_path,
                        ["resume_probe"],
                        output=output,
                        resume=True,
                    )

    def test_final_inventory_contains_final_run_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            context = RunContext(
                {"profile": "test", "device": "cpu", "seeds": [0]},
                Path(directory),
            )
            (context.output / "manifests" / "run_probe.json").write_text(
                "{}", encoding="utf-8"
            )
            finalize(context, ["probe"], [])
            with (context.output / "manifests" / "artifact_inventory.csv").open(
                encoding="utf-8"
            ) as stream:
                paths = {row["relative_path"] for row in csv.DictReader(stream)}
            self.assertIn("manifests/run_completeness.csv", paths)
            self.assertIn("manifests/failures.json", paths)

    def test_circuit_latex_table_is_compact_and_escaped(self):
        frame = pd.DataFrame(
            [
                {
                    "stimulus": "held_out",
                    "openvaf_status": "passed",
                    "openvaf_version": "OpenVAF-reloaded",
                    "openvaf_executable": r"C:\tool path\openvaf-r.exe",
                    "ngspice_status": "available",
                    "ngspice_version_label": "ngspice-45.2\nextra",
                    "osdi_simulation_status": "passed",
                    "osdi_max_abs_error": 1e-8,
                    "osdi_mean_abs_error": 1e-9,
                    "max_abs_tolerance": 1e-3,
                    "circuit_claim_valid": True,
                }
            ]
        )
        table = _latex_ready(frame, "circuit_validation")
        self.assertNotIn("openvaf_executable", table.columns)
        self.assertNotIn("\n", table.loc[0, "ngspice"])
        latex = table.to_latex(index=False, escape=True)
        self.assertIn("OpenVAF-r", latex)

    def test_checkpoint_contains_architecture_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            context = RunContext(
                {"profile": "test", "device": "cpu", "seeds": [0]},
                Path(directory),
            )
            record = context.checkpoint(RecurrentStudent(2, 4, 3), "probe")
            payload = torch.load(
                context.output / record["checkpoint_path"], weights_only=True
            )
            self.assertEqual(payload["architecture"]["input_size"], 2)
            self.assertEqual(payload["architecture"]["hidden_size"], 4)
            self.assertEqual(payload["architecture"]["output_size"], 3)
            self.assertIn("source_tree_sha256", payload)
            self.assertIn("config_sha256", payload)

    def test_tbptt_absorbs_singleton_remainder_without_pointwise_call(self):
        batch = SequenceBatch(
            torch.randn(1, 9, 2),
            torch.randn(1, 9, 1),
            torch.arange(9, dtype=torch.float32).view(1, 9, 1),
            ("source",),
        )
        chunks = list(contiguous_chunks(batch, 4))
        self.assertEqual([item.features.shape[1] for _, item in chunks], [4, 5])
        rebuilt = torch.cat([item.features for _, item in chunks], dim=1)
        torch.testing.assert_close(rebuilt, batch.features)

    def test_source_bundle_integrity_and_experiment_coverage(self):
        from psi_vortex.verify import verify_repository

        report = verify_repository()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["registered_experiment_groups"], len(EXPERIMENTS))

    def test_printed_cycles_cannot_cross_physical_device_splits(self):
        root = Path(__file__).resolve().parents[1]
        trajectories = load_printed_memristor(
            root / "data" / "printed_memristor_training_data.csv"
        )
        train, validation, test = split_printed_memristor_sources(
            trajectories, [0, 1, 2], [3], [4]
        )
        identities = [
            {item.source_trajectory_id for item in split}
            for split in (train, validation, test)
        ]
        self.assertFalse(identities[0] & identities[1])
        self.assertFalse(identities[0] & identities[2])
        self.assertFalse(identities[1] & identities[2])
        self.assertTrue(all(item.features.shape[0] > 1 for item in trajectories))

    def test_all_recurrent_models_reject_pointwise_and_length_one(self):
        for model in (
            RecurrentStudent(2, 4),
            LowRankRecurrentStudent(2, 4, 2),
            PsiXLSTMTeacher(2, 8, num_blocks=4),
        ):
            with self.assertRaises(ValueError):
                model(torch.randn(5, 2))
            with self.assertRaises(ValueError):
                model(torch.randn(2, 1, 2))

    def test_canonical_teacher_topology_is_preserved(self):
        teacher = PsiXLSTMTeacher(2, hidden_size=32, num_blocks=4)
        self.assertEqual(teacher.block_widths, [16, 16, 32, 32])
        self.assertIsInstance(teacher.blocks[0], MatrixLSTMBlock)
        self.assertIsInstance(teacher.blocks[1], ScalarLSTMBlock)
        self.assertTrue(hasattr(teacher.blocks[1], "memory_proj"))
        self.assertTrue(hasattr(teacher.blocks[1], "norm"))
        self.assertEqual(teacher.fusion.in_features, 96)
        self.assertEqual(teacher.blocks[0].memory_size, 16)

    def test_scalar_memory_equation_matches_manual_step(self):
        torch.manual_seed(4)
        block = ScalarLSTMBlock(3, 5)
        value = torch.randn(2, 2, 3)
        output, (hidden, cell) = block(value)
        h0 = torch.zeros(2, 5)
        c0 = torch.zeros(2, 5)
        h1_raw, c1_raw = block.cell(value[:, 0], (h0, c0))
        gate, candidate = block.memory_proj(h1_raw).chunk(2, dim=-1)
        c1 = c1_raw + torch.sigmoid(gate) * torch.tanh(candidate)
        h1 = block.norm(h1_raw)
        self.assertTrue(torch.allclose(output[:, 0], h1, atol=1e-6))
        h2_raw, c2_raw = block.cell(value[:, 1], (h1, c1))
        gate, candidate = block.memory_proj(h2_raw).chunk(2, dim=-1)
        expected_cell = c2_raw + torch.sigmoid(gate) * torch.tanh(candidate)
        self.assertTrue(torch.allclose(cell, expected_cell, atol=1e-6))
        self.assertTrue(torch.allclose(hidden, block.norm(h2_raw), atol=1e-6))

    def test_teacher_fuses_every_block_sequence(self):
        teacher = PsiXLSTMTeacher(2, 8, num_blocks=4)
        output, state, auxiliary = teacher(torch.randn(2, 6, 2))
        self.assertEqual(output.shape, (2, 6, 1))
        self.assertEqual(len(state), 4)
        self.assertEqual(len(auxiliary["block_outputs"]), 4)
        self.assertEqual(auxiliary["recurrent"].shape, (2, 6, 8))

    def test_full_sequence_equals_threaded_contiguous_chunks(self):
        value = torch.randn(2, 8, 2)
        for model in (
            RecurrentStudent(2, 4),
            LowRankRecurrentStudent(2, 4, 2),
            PsiXLSTMTeacher(2, 8, num_blocks=4),
        ):
            model.eval()
            expected = model(value)[0]
            state = None
            pieces = []
            for part in (value[:, :4], value[:, 4:]):
                output, state, _ = model(part, state)
                pieces.append(output)
            self.assertTrue(
                torch.allclose(expected, torch.cat(pieces, dim=1), atol=1e-6),
                type(model).__name__,
            )

    def test_gradient_crosses_timesteps_and_order_matters(self):
        model = PsiXLSTMTeacher(2, 8, num_blocks=4)
        value = torch.randn(1, 5, 2, requires_grad=True)
        output = model(value)[0]
        output[:, -1].sum().backward()
        self.assertGreater(float(value.grad[:, :-1].abs().max()), 0)
        order = torch.tensor([2, 0, 4, 1, 3])
        permuted = model(value.detach()[:, order])[0]
        self.assertFalse(torch.allclose(output.detach()[:, order], permuted))

    def test_state_detaches_and_resets_between_independent_trajectories(self):
        probe = StateProbe()
        trainer = SequenceTrainer(probe, torch.optim.SGD(probe.parameters(), lr=0.01), 4)
        trainer.train_epoch(iter_batches([trajectory("a"), trajectory("b")], 1))
        self.assertIsNone(probe.received[0])
        self.assertIsNone(probe.received[2])
        self.assertIsNone(probe.received[1].grad_fn)
        self.assertIsNone(probe.received[3].grad_fn)

    def test_variable_lengths_are_bucketed_without_timestep_reordering(self):
        items = [trajectory("a", 5), trajectory("b", 8), trajectory("c", 5)]
        batches = list(iter_batches(items, 2, shuffle_trajectories=True))
        observed = {}
        for batch in batches:
            self.assertTrue(torch.all(batch.time[:, 1:] > batch.time[:, :-1]))
            for index, name in enumerate(batch.trajectory_ids):
                observed[name] = batch.time[index, :, 0].tolist()
        self.assertEqual(set(observed), {"a", "b", "c"})
        self.assertEqual(observed["a"], list(range(5)))

    def test_overlapping_source_windows_cannot_cross_splits(self):
        windows = make_windows(trajectory("source", 12), 8, stride=4)
        pipeline = EndToEndPipeline(2, teacher_hidden=8, teacher_blocks=2, student_hidden=4)
        with self.assertRaises(ValueError):
            pipeline.fit([windows[0]], [windows[1]], [trajectory("test")], teacher_epochs=0, student_epochs=0)

    def test_public_evaluation_supports_unequal_lengths(self):
        pipeline = EndToEndPipeline(2, teacher_hidden=8, teacher_blocks=2, student_hidden=4)
        result = pipeline.evaluate([trajectory("a", 4), trajectory("b", 7)])
        self.assertIn("mse", result)

    def test_low_rank_and_gru_exports_preserve_batch_and_streaming_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in (RecurrentStudent(2, 4), LowRankRecurrentStudent(2, 4, 2)):
                value = torch.randn(2, 5, 2)
                expected = model(value)[0]
                path = Path(directory) / f"{type(model).__name__}.pt"
                export_torchscript(model, value, path)
                loaded = torch.jit.load(str(path))
                self.assertTrue(torch.allclose(expected, loaded(value), atol=1e-6))
                state = loaded.initial_state(2)
                pieces = []
                for index in range(value.shape[1]):
                    output, state = loaded.step(value[:, index], state)
                    pieces.append(output[:, None])
                self.assertTrue(torch.allclose(expected, torch.cat(pieces, dim=1), atol=1e-6))
                with self.assertRaises(torch.jit.Error):
                    loaded(value[:, :1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_export_is_cpu_portable_without_moving_source_model(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in (
                RecurrentStudent(2, 4).cuda(),
                LowRankRecurrentStudent(2, 4, 2).cuda(),
            ):
                value = torch.randn(2, 5, 2)
                with torch.no_grad():
                    expected = model(value.cuda())[0].cpu()
                path = Path(directory) / f"cuda_{type(model).__name__}.pt"
                export_torchscript(model, value, path)
                self.assertTrue(next(model.parameters()).is_cuda)
                loaded = torch.jit.load(str(path))
                self.assertFalse(next(loaded.parameters()).is_cuda)
                self.assertTrue(torch.allclose(expected, loaded(value), atol=1e-6))

    def test_final_compression_rank_sweep_contains_only_valid_bottlenecks(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs" / "final.json").read_text(encoding="utf-8"))
        # The measured compression experiment has two input features and uses
        # student_hidden for every low-rank candidate.
        maximum_rank = 2 + config["student_hidden"]
        self.assertTrue(config["student_rank_sweep"])
        self.assertTrue(
            all(1 <= rank <= maximum_rank for rank in config["student_rank_sweep"])
        )
        self.assertEqual(config["compression_device"], "GO-PEI4")
        self.assertEqual(config["compression_training_mode"], "direct_supervised")
        self.assertEqual(config["compression_learning_rate"], 0.005)

    def test_invalid_compression_rank_fails_before_training(self):
        context = SimpleNamespace(
            config={"student_hidden": 8, "student_rank_sweep": [1, 16]}
        )
        with self.assertRaisesRegex(ValueError, "student_rank_sweep"):
            compression_fidelity(context)

    def test_invalid_compression_device_fails_before_loading_data(self):
        context = SimpleNamespace(
            config={
                "student_hidden": 8,
                "student_rank_sweep": [1, 2],
                "compression_device": "GO-NOT-DECLARED",
                "real_devices": ["GO-PEI4"],
            }
        )
        with (
            patch("experiments.measured.load_measured_cycles") as load,
            self.assertRaisesRegex(ValueError, "compression_device"),
        ):
            compression_fidelity(context)
        load.assert_not_called()

    def test_baseline_comparison_covers_every_alpha_method_and_seed(self):
        class ContextProbe:
            config = {
                "profile": "test",
                "device": "cpu",
                "seeds": [3, 5],
                "alphas": [0.05, 0.2],
                "student_epochs": 0,
                "student_hidden": 4,
                "batch_size": 1,
                "chunk_length": 4,
                "learning_rate": 1e-3,
                "r2_accept": 0.8,
            }

            @staticmethod
            def checkpoint(model, label):
                return {}

            @staticmethod
            def checkpoint_payload(payload, label):
                return {}

        fake_sindy = SimpleNamespace(
            active_terms=1,
            coefficients=np.zeros((1, 1)),
            powers=[(0,)],
            fit=lambda trajectories: fake_sindy,
        )
        with (
            patch(
                "experiments.synthetic.thermal_data",
                side_effect=lambda config, alpha: ([], [], [], []),
            ),
            patch("experiments.synthetic.train_static_model"),
            patch("experiments.synthetic.SINDyRegressor", return_value=fake_sindy),
            patch("experiments.synthetic.mean_estimate", return_value={}),
            patch("experiments.synthetic.coupling_scores", return_value=[]),
        ):
            rows = baseline_comparison(ContextProbe())

        observed = {
            (row["alpha_true"], row["method"], row["seed"]) for row in rows
        }
        expected = {
            (alpha, method, seed)
            for alpha in ContextProbe.config["alphas"]
            for method in ("mlp", "pikan", "sindy", "vanilla_lstm")
            for seed in ContextProbe.config["seeds"]
        }
        self.assertEqual(observed, expected)
        self.assertEqual(len(rows), len(expected))

    def test_write_rows_records_execution_device_and_cluster_count(self):
        with tempfile.TemporaryDirectory() as directory:
            context = RunContext.__new__(RunContext)
            context.config = {"device": "cuda"}
            context.output = Path(directory)
            architecture = json.dumps({"materialized_cluster_count": 7})
            destination = context.write_rows(
                "probe", [{"model_architecture": architecture}]
            )
            with destination.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["execution_device"], "cuda")
        self.assertEqual(row["cluster_count"], "7")

    def test_thermal_splits_are_source_disjoint_and_chronological(self):
        train, validation, test, _ = thermal_split(
            0.08, [1, 2], [3], [4], n_steps=64, stride=2
        )
        source_sets = [{item.source_trajectory_id for item in split} for split in (train, validation, test)]
        self.assertFalse(source_sets[0] & source_sets[1])
        self.assertFalse(source_sets[0] & source_sets[2])
        self.assertTrue(all(torch.all(item.time[1:] > item.time[:-1]) for item in train + validation + test))

    def test_latent_readout_and_gate(self):
        driver = np.linspace(0, 4, 30)
        response = 0.08 * driver + 0.3
        estimate = fit_coupling(response, driver, r2_threshold=0.8)
        self.assertAlmostEqual(estimate.value, 0.08, places=10)
        self.assertTrue(estimate.accepted)

    def test_latent_driver_band_excludes_near_zero_contamination(self):
        driver = np.linspace(-2, 2, 81)
        response = 0.08 * driver + 0.3
        response[np.abs(driver) <= 0.5] += 2.0 * driver[np.abs(driver) <= 0.5]
        estimate = fit_coupling(response, driver, driver_band=0.5)
        self.assertAlmostEqual(estimate.value, 0.08, places=10)
        self.assertEqual(estimate.observations, int(np.sum(np.abs(driver) > 0.5)))

    def test_symmetry_detector_distinguishes_odd_and_even(self):
        value = torch.linspace(-2, 2, 201)
        odd = AutoSymmetryDetector().detect(value, value**3)
        even = AutoSymmetryDetector().detect(value, value**2)
        self.assertEqual(odd.symmetry, "odd")
        self.assertEqual(even.symmetry, "even")

    def test_every_manifest_group_has_a_callable_runner(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "manifests" / "experiment_coverage.csv").open(encoding="utf-8") as stream:
            rows = list(__import__("csv").DictReader(stream))
        self.assertEqual({row["experiment_group"] for row in rows}, set(EXPERIMENTS))
        self.assertTrue(all(callable(value) for value in EXPERIMENTS.values()))

    def test_no_noncanonical_module_is_imported(self):
        root = Path(__file__).resolve().parents[1]
        banned = {"legacy_pointwise"}
        for path in list((root / "psi_vortex").glob("*.py")) + list((root / "experiments").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            self.assertFalse(imports & banned, f"{path}: {imports & banned}")


if __name__ == "__main__":
    unittest.main()
