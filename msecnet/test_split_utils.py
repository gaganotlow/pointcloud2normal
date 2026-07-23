"""Tests for leakage-resistant dataset grouping and assignment."""
import unittest

from split_utils import build_group_split, generalization_group


class SplitUtilsTest(unittest.TestCase):
    def test_encoded_model_joins_direct_model(self):
        encoded = {
            "file": "td07__增加IR异常数据_车型51至车型59_凯迪拉克XT5_正常数据_color_20251127_103327_323.npz",
            "car_model": "车型51至车型59",
            "dataset": "testdept_20260707",
        }
        direct = {"file": "sample.npz", "car_model": "凯迪拉克XT5", "dataset": "fuelcap"}
        self.assertEqual(generalization_group(encoded)[0], generalization_group(direct)[0])

    def test_capture_sessions_are_grouped(self):
        first = {"file": "2026_03_07_14_17_50__v0.npz", "car_model": "prod_open_inside", "dataset": "prod_open_inside"}
        second = {"file": "2026_03_07_18_00_00__v0.npz", "car_model": "prod_open_inside", "dataset": "prod_open_inside"}
        other_day = {"file": "2026_03_08_14_17_50__v0.npz", "car_model": "prod_open_inside", "dataset": "prod_open_inside"}
        self.assertEqual(generalization_group(first)[0], generalization_group(second)[0])
        self.assertNotEqual(generalization_group(first)[0], generalization_group(other_day)[0])

    def test_whole_groups_never_cross_splits(self):
        counts = {"车型A": 12, "车型B": 10, "车型C": 9, "车型D": 8, "车型E": 7, "车型F": 6}
        rows = []
        for model, count in counts.items():
            rows.extend({"file": f"{model}_{index}.npz", "car_model": model, "dataset": "test"} for index in range(count))
        split, file_split, metadata = build_group_split(rows, seed=7, val_fraction=0.2, test_fraction=0.2)
        self.assertEqual(sum(len(names) for names in split.values()), len(rows))
        for model, count in counts.items():
            assignments = {file_split[f"{model}_{index}.npz"] for index in range(count)}
            self.assertEqual(len(assignments), 1)
        self.assertEqual(metadata["counts"], {name: len(split[name]) for name in ("train", "val", "test")})


if __name__ == "__main__":
    unittest.main()
