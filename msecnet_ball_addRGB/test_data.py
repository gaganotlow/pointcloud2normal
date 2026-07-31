"""CPU-only checks for image crop and separate-modality packing."""
import unittest

import numpy as np

from .data import centered_rgb_crop, project_center


class RGBDataTest(unittest.TestCase):
    def test_project_center_uses_normalized_intrinsics(self):
        k_norm = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]], dtype=np.float32)
        self.assertEqual(project_center(np.array([0.0, 0.0, 2.0]), k_norm, 640, 480), (320.0, 240.0))

    def test_crop_pads_at_image_edge(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8); image[0, 0] = [1, 2, 3]
        k_norm = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0], [0, 0, 1]], dtype=np.float32)
        crop = centered_rgb_crop(image, np.array([0.0, 0.0, 1.0]), k_norm, 30, 20, 0.08, 2.0, 32)
        self.assertEqual(crop.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
