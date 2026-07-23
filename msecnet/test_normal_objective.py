"""CPU tests for the normal aggregation and loss semantics."""
import unittest

import torch

from train import aggregate_point_normals, normal_losses, oriented_angular_error_deg


class NormalObjectiveTest(unittest.TestCase):
    def test_aggregation_is_invariant_to_vector_scale(self):
        vectors = torch.tensor([
            [100.0, 0.0, 0.0], [0.01, 0.0, 0.0],
            [0.0, 7.0, 0.0], [0.0, 0.1, 0.0],
        ])
        means, normals, _ = aggregate_point_normals(vectors, torch.tensor([2, 2]))
        self.assertTrue(torch.allclose(normals, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])))
        self.assertTrue(torch.allclose(means.norm(dim=1), torch.ones(2)))

    def test_oriented_loss_penalizes_a_reversed_normal(self):
        vectors = torch.tensor([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        loss, point_loss, patch_loss, prediction, _ = normal_losses(
            vectors, target, torch.tensor([2]), point_loss_weight=0.25
        )
        self.assertTrue(torch.allclose(prediction, torch.tensor([[-1.0, 0.0, 0.0]])))
        self.assertTrue(torch.allclose(point_loss, torch.tensor([2.0])))
        self.assertTrue(torch.allclose(patch_loss, torch.tensor([2.0])))
        self.assertTrue(torch.allclose(loss, torch.tensor([2.0])))
        self.assertTrue(torch.allclose(oriented_angular_error_deg(prediction, target), torch.tensor([180.0])))

    def test_consensus_exposes_opposing_point_directions(self):
        vectors = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        means, normal, _ = aggregate_point_normals(vectors, torch.tensor([3]))
        self.assertTrue(torch.allclose(normal, torch.tensor([[1.0, 0.0, 0.0]])))
        self.assertTrue(torch.allclose(means.norm(dim=1), torch.tensor([1.0 / 3.0])))


if __name__ == "__main__":
    unittest.main()
