import unittest

import pandas as pd
from src.visualisation.evaluate_labels import filter_visualization_classes


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


if __name__ == "__main__":
    unittest.main()
