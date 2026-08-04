"""CPU-only checks for image crop and separate-modality packing."""
import unittest

import numpy as np

from .data import centered_rgb_crop, obb_crop_bounds, obb_rgb_crop, project_center, project_points_to_crop_grid


class RGBDataTest(unittest.TestCase):
    def test_project_center_uses_normalized_intrinsics(self):
        k_norm = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]], dtype=np.float32)
        self.assertEqual(project_center(np.array([0.0, 0.0, 2.0]), k_norm, 640, 480), (320.0, 240.0))

    def test_crop_pads_at_image_edge(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8); image[0, 0] = [1, 2, 3]
        k_norm = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0], [0, 0, 1]], dtype=np.float32)
        crop = centered_rgb_crop(image, np.array([0.0, 0.0, 1.0]), k_norm, 30, 20, 0.08, 2.0, 32)
        self.assertEqual(crop.shape, (32, 32, 3))

    def test_obb_crop_preserves_camera_orientation(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[30:50, 30:50] = [12, 34, 56]
        detection = {
            "corners": [[26, 32], [48, 26], [54, 48], [32, 54]],
            "confidence": 0.9, "class_id": 0,
        }
        crop = obb_rgb_crop(image, detection, 1.3, 48)
        self.assertEqual(crop.shape, (48, 48, 3))
        self.assertGreater(crop.mean(), 0)

    def test_projected_points_use_the_same_obb_crop_coordinates(self):
        detection = {"corners": [[5, 5], [15, 5], [15, 15], [5, 15]], "cx": 10, "cy": 10, "w": 10, "h": 10}
        left, top, side = obb_crop_bounds(detection, 1.0)
        intrinsics = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]], dtype=np.float32)
        grid = project_points_to_crop_grid(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), intrinsics, 20, 20, left, top, side)
        np.testing.assert_allclose(grid, [[0.03125, 0.03125]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
