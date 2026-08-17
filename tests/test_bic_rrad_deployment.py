from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from psi_vortex import (
    EndToEndPipeline,
    LowRankRecurrentStudent,
    PsiXLSTMTeacher,
    RRADLoss,
    RecurrentStudent,
    Trajectory,
    compression_report,
    compile_openvaf,
    generate_verilog_a,
    iter_batches,
    materialize_weight_clusters,
    simulate_osdi,
)
from psi_vortex.bic import exact_effective_dof
from psi_vortex.trainer import VortexTrainer


def trajectory(name: str, length: int = 8, outputs: int = 1):
    time = torch.arange(length, dtype=torch.float32)[:, None]
    features = torch.cat((time, time.square()), dim=1)
    targets = torch.cat([time * (index + 1) for index in range(outputs)], dim=1)
    return Trajectory(name, features, targets, time, name)


class BicRradDeploymentTests(unittest.TestCase):
    def test_pipeline_applies_bic_only_from_declared_start_epoch(self):
        class TrainerProbe:
            instance = None

            def __init__(self, *args, **kwargs):
                self.lambda_bic = None
                self.seen = []
                TrainerProbe.instance = self

            def train_epoch(self, batches):
                list(batches)
                self.seen.append(self.lambda_bic)

        pipeline = EndToEndPipeline(
            2, teacher_hidden=8, teacher_blocks=2, student_hidden=4
        )
        with patch("psi_vortex.pipeline.VortexTrainer", TrainerProbe):
            pipeline.fit(
                [trajectory("train", 8)],
                [trajectory("validation", 8)],
                [trajectory("test", 8)],
                teacher_epochs=0,
                student_epochs=4,
                lambda_bic=0.25,
                bic_start_epoch=2,
                cluster_candidates=(),
            )
        self.assertEqual(TrainerProbe.instance.seen, [0.0, 0.0, 0.25, 0.25])

    def test_rrad_boundary_cache_is_detached_between_tbptt_chunks(self):
        teacher = RecurrentStudent(2, 4)
        student = RecurrentStudent(2, 3)
        trainer = VortexTrainer(
            teacher,
            student,
            torch.optim.SGD(student.parameters(), lr=0.001),
            chunk_length=4,
            lambda_bic=0.0,
        )

        class BoundaryProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.previous = []

            def forward(self, student_y, teacher_y, time, student_hidden, teacher_hidden, previous):
                self.previous.append(previous)
                zero = student_y.sum() * 0
                return zero, {"statistics": zero}

            def aggregate(self, statistics):
                return sum(statistics)

        probe = BoundaryProbe()
        trainer.rrad = probe
        trainer.train_epoch(iter_batches([trajectory("chunked", 8)], 1))
        self.assertIsNone(probe.previous[0])
        for value in probe.previous[1].values():
            self.assertIsNone(value.grad_fn)

    def test_blocked_bic_matches_dense_value_and_gradient(self):
        weights = torch.tensor([-1.0, -0.9, 0.5, 0.55, 2.0], requires_grad=True)
        gamma = 0.2
        density = torch.exp(
            -((weights[:, None] - weights[None, :]).square()) / (2 * gamma * gamma)
        ).sum(dim=1)
        expected = (1 / density).sum()
        actual = exact_effective_dof(weights, gamma, block_size=2)
        expected_gradient = torch.autograd.grad(expected, weights, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(actual, weights)[0]
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(actual_gradient, expected_gradient, atol=1e-6))

    def test_rrad_chunk_statistics_equal_full_sequence(self):
        loss = RRADLoss(5, 3)
        time = torch.arange(6.0).view(1, 6, 1)
        student = torch.randn(1, 6, 1)
        teacher = torch.randn(1, 6, 1)
        student_hidden = torch.randn(1, 6, 3)
        teacher_hidden = torch.randn(1, 6, 5)
        full, _ = loss(student, teacher, time, student_hidden, teacher_hidden)
        statistics = []
        previous = None
        for selection in (slice(0, 3), slice(3, 6)):
            _, parts = loss(
                student[:, selection],
                teacher[:, selection],
                time[:, selection],
                student_hidden[:, selection],
                teacher_hidden[:, selection],
                previous,
            )
            statistics.append(parts["statistics"])
            previous = {
                "student": student[:, selection][:, -1:],
                "teacher": teacher[:, selection][:, -1:],
                "student_hidden": student_hidden[:, selection][:, -1:],
                "teacher_hidden": teacher_hidden[:, selection][:, -1:],
                "time": time[:, selection][:, -1:],
            }
        self.assertTrue(torch.allclose(full, loss.aggregate(statistics), atol=1e-6))

    def test_bic_counts_scalar_multi_output_elements(self):
        teacher = PsiXLSTMTeacher(2, 8, output_size=2, num_blocks=2)
        student = RecurrentStudent(2, 4, output_size=2)
        trainer = VortexTrainer(
            teacher, student, torch.optim.SGD(student.parameters(), lr=0.001), chunk_length=4
        )

        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.observations = None

            def forward(self, model, mse, observations):
                self.observations = observations
                return mse * 0

        trainer.bic = Probe()
        trainer.train_epoch(iter_batches([trajectory("multi", 8, 2)], 1))
        self.assertEqual(trainer.bic.observations, 16)

    def test_cluster_assignments_remain_tied(self):
        clustered = materialize_weight_clusters(RecurrentStudent(2, 4), 3)
        optimizer = torch.optim.SGD(clustered.parameters(), lr=0.1)
        clustered(torch.randn(2, 4, 2))[0].square().mean().backward()
        optimizer.step()
        expanded = torch.cat([value.flatten() for value in clustered.expanded_parameters().values()])
        self.assertLessEqual(torch.unique(expanded).numel(), 3)
        report = compression_report(clustered)
        self.assertEqual(report["effective_trainable_values"], 3)
        self.assertGreater(report["expanded_parameters"], 3)

    def test_validation_bic_uses_total_target_elements(self):
        pipeline = EndToEndPipeline(2, output_size=2, teacher_hidden=8, teacher_blocks=2, student_hidden=4)
        validation = [trajectory("validation-short", 2, 2), trajectory("validation-long", 6, 2)]
        captured = {}

        def fake_select(model, candidates, loss_fn, observations):
            captured["observations"] = observations
            captured["loss"] = float(loss_fn(model))
            return 0.0, candidates[0], materialize_weight_clusters(model, candidates[0])

        with patch("psi_vortex.pipeline.select_cluster_count", side_effect=fake_select):
            pipeline.fit(
                [trajectory("train", 8, 2)],
                validation,
                [trajectory("test", 8, 2)],
                teacher_epochs=0,
                student_epochs=0,
                cluster_candidates=(2,),
            )
        self.assertEqual(captured["observations"], 16)

    def test_verilog_a_generation_covers_both_student_types(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in (RecurrentStudent(2, 3), LowRankRecurrentStudent(2, 3, 2)):
                path = generate_verilog_a(
                    model, Path(directory) / f"{type(model).__name__}.va", sample_period=1e-4
                )
                text = path.read_text(encoding="utf-8")
                self.assertIn("sample_clock", text)
                self.assertIn("ddt(V(state_0))", text)
                self.assertIn("if (V(sample_clock) < 0.5)", text)
                self.assertNotIn("@(timer", text)
                self.assertNotIn("transition(", text)
                self.assertIn("endmodule", text)

    def test_external_osdi_preserves_order_and_resets_state(self):
        configured = os.environ.get("PSI_VORTEX_OPENVAF")
        compiler = (
            shutil.which(configured) if configured else None
        ) or shutil.which("openvaf") or shutil.which("openvaf-r")
        simulator = shutil.which("ngspice_con") or shutil.which("ngspice")
        if compiler is None or simulator is None:
            self.skipTest("OpenVAF and ngspice are required for external OSDI validation")

        model = RecurrentStudent(1, 2).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.2)
        values = np.asarray([[0.0], [1.0], [0.5], [-0.25], [0.75]], dtype=np.float32)

        def sampled(result):
            self.assertEqual(result["status"], "passed", result)
            return np.interp(result["sample_times"], result["times"], result["outputs"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = generate_verilog_a(model, root / "student.va", sample_period=2e-4)
            compiled = compile_openvaf(source)
            self.assertEqual(compiled["status"], "passed", compiled)
            first = sampled(
                simulate_osdi(
                    compiled["artifact"],
                    "psi_vortex_student",
                    values,
                    sample_period=2e-4,
                    work_directory=root / "first",
                )
            )
            second = sampled(
                simulate_osdi(
                    compiled["artifact"],
                    "psi_vortex_student",
                    values,
                    sample_period=2e-4,
                    work_directory=root / "second",
                )
            )
            reverse = sampled(
                simulate_osdi(
                    compiled["artifact"],
                    "psi_vortex_student",
                    values[::-1].copy(),
                    sample_period=2e-4,
                    work_directory=root / "reverse",
                )
            )

        with torch.no_grad():
            expected = model(torch.tensor(values).unsqueeze(0))[0][0, :, 0].numpy()
            reverse_expected = model(
                torch.tensor(values[::-1].copy()).unsqueeze(0)
            )[0][0, :, 0].numpy()
        np.testing.assert_allclose(first, expected, atol=1e-3, rtol=0)
        np.testing.assert_allclose(reverse, reverse_expected, atol=1e-3, rtol=0)
        np.testing.assert_allclose(first, second, atol=1e-8, rtol=0)
        self.assertGreater(float(np.max(np.abs(first - reverse))), 1e-3)


if __name__ == "__main__":
    unittest.main()
