"""CPU tests for PointNet++ ball preprocessing and the global normal head."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pointnet2_ball.data import BallNormalDataset, collate_fixed_points
from pointnet2_ball.model import build_model, farthest_point_sample
from pointnet2_ball.train import normal_loss


class PointNet2BallTest(unittest.TestCase):
    def test_dataset_centers_and_repeats_sparse_balls_to_fixed_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            cloud_dir = Path(temporary)
            np.savez_compressed(
                cloud_dir / "sample.npz",
                xyz=np.array([[1.02, 2.0, 3.0], [1.0, 2.04, 3.0]], dtype=np.float32),
                label=np.ones(2, dtype=np.uint8),
            )
            dataset = BallNormalDataset(
                ["sample.npz"], [[0.0, 0.0, 1.0]], cloud_dir,
                {"sample.npz": {"selection": "manual_center_ball_patch", "center_3d": [1.0, 2.0, 3.0], "ball_radius_m": 0.08}},
                0.08, 8, train=False, seed=1,
            )
            xyz, radial, normal, _ = dataset[0]
            self.assertEqual(xyz.shape, (8, 3))
            self.assertEqual(radial.shape, (8, 1))
            np.testing.assert_allclose(normal, [0, 0, 1])
            self.assertTrue(np.all(np.min(np.abs(xyz[:, 0, None] - np.array([0.0, 0.25])), axis=1) < 1e-6))

    def test_fps_and_model_emit_one_unit_vector_per_cloud(self):
        torch.manual_seed(3)
        xyz = torch.rand(2, 32, 3) - 0.5
        radial = xyz.norm(dim=-1, keepdim=True)
        indices = farthest_point_sample(xyz, 12)
        self.assertEqual(indices.shape, (2, 12))
        model = build_model({"sa1_points": 16, "sa2_points": 8, "sa1_k": 8, "sa2_k": 8, "dropout": 0.0})
        model.eval()
        normal = model.predict_normal(xyz, radial)
        self.assertEqual(normal.shape, (2, 3))
        self.assertTrue(torch.allclose(normal.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_directed_loss_penalizes_a_flipped_normal(self):
        target = torch.tensor([[0.0, 0.0, 1.0]])
        self.assertLess(float(normal_loss(target, target)), 1e-6)
        self.assertGreater(float(normal_loss(-target, target)), 1.99)


if __name__ == "__main__":
    unittest.main()
