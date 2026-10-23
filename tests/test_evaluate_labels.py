import unittest

import pandas as pd
from src.visualisation.evaluate_labels import (
    filter_baiting_classes,
    filter_visualization_classes,
)


class VisualizationClassFilterTests(unittest.TestCase):
    def test_keeps_only_requested_wildlife_classes(self):
        classes = [
            "roe deer",
            "wild boar",
            "pigeon",
            "badger",
            "marten",
            "fox",
            "hare",
            "raccoon",
            "crow",
            "horse",
            "none",
            "unknown",
        ]
        dataframe = pd.DataFrame({"class": classes})

        filtered = filter_visualization_classes(dataframe)

        self.assertEqual(filtered["class"].tolist(), classes[:9])

    def test_uses_manual_labels_and_falls_back_to_class_when_blank(self):
        dataframe = pd.DataFrame(
            {
                "class": ["horse", "roe deer", "fox", "crow"],
                "manual_label": ["roe deer", "", None, "horse"],
            }
        )

        filtered = filter_visualization_classes(dataframe)

        self.assertEqual(filtered["class"].tolist(), ["roe deer", "roe deer", "fox"])

    def test_baiting_filter_keeps_humans_and_target_species(self):
        dataframe = pd.DataFrame(
            {
                "class": ["horse", "human", "crow", "fox", "wild boar"],
                "manual_label": ["human", "", "roe deer", "wild boar", "squirrel"],
            }
        )

        filtered = filter_baiting_classes(dataframe)

        self.assertEqual(
            filtered["class"].tolist(), ["human", "human", "roe deer", "wild boar"]
        )


if __name__ == "__main__":
    unittest.main()
