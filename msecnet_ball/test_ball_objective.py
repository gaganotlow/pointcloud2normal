"""CPU tests for center-ball selection, anchoring, and radial weighting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from msecnet_ball.prepare_ball_dataset import ball_mask
from msecnet_ball.train import BallNormalDS, aggregate_point_normals, radial_weights


class BallObjectiveTest(unittest.TestCase):
    def test_ball_mask_includes_boundary_and_excludes_outside(self):
        xyz = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0801, 0.0, 0.0]], dtype=np.float32)
        self.assertEqual(ball_mask(xyz, np.zeros(3, dtype=np.float32), 0.08).tolist(), [True, True, False])

    def test_dataset_is_center_anchored_and_fixed_radius_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            cloud_dir = Path(tmp)
            np.savez_compressed(
                cloud_dir / "sample.npz",
                xyz=np.array([[1.02, 2.0, 3.0], [1.00, 2.04, 3.0]], dtype=np.float32),
                label=np.ones(2, dtype=np.uint8),
            )
            dataset = BallNormalDS(
                np.array(["sample.npz"]), np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                str(cloud_dir), 0, False,
                {"sample.npz": {"selection": "manual_center_ball_patch", "center_3d": [1.0, 2.0, 3.0], "ball_radius_m": 0.08}},
                0.08, seed=1,
            )
            xyz, radius, normal, _ = dataset[0]
            np.testing.assert_allclose(xyz, [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]], atol=1e-6)
            np.testing.assert_allclose(radius[:, 0], [0.25, 0.5], atol=1e-6)
            np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)

    def test_radial_weighting_prefers_near_center_direction(self):
        vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        radii = torch.tensor([[0.0], [1.0]])
        _, normal, _ = aggregate_point_normals(vectors, torch.tensor([2]), radial_weights(radii, beta=2.0))
        self.assertGreater(float(normal[0, 0]), float(normal[0, 1]))


if __name__ == "__main__":
    unittest.main()
