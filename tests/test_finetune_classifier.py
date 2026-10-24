import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from src.labelling import finetune_classifier


class FinetuneLabelTests(unittest.TestCase):
    def test_load_labelled_rows_strips_manual_labels_before_counting(self):
        labels = ["pigeon", "pigeon", "pigeon", "pigeon", "pigeon ", "human ", "cat"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "labels.csv"
            pd.DataFrame(
                {
                    "box": ["[0.5, 0.5, 0.4, 0.4]"] * len(labels),
                    "manual_label": labels,
                    "location_id": ["location"] * len(labels),
                    "image_file": [
                        f"image_{index}.jpg" for index in range(len(labels))
                    ],
                }
            ).to_csv(csv_path, index=False)
            indexed_paths = {
                ("location", f"image_{index}.jpg"): temp_path / f"image_{index}.jpg"
                for index in range(len(labels))
            }

            with (
                patch.object(finetune_classifier, "CSV_PATH", csv_path),
                patch.object(
                    finetune_classifier, "image_paths", return_value=indexed_paths
                ),
            ):
                rows = finetune_classifier.load_labelled_rows()

        self.assertEqual(rows["manual_label"].tolist(), ["pigeon"] * 5)


if __name__ == "__main__":
    unittest.main()
