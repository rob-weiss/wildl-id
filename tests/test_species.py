import json
import unittest
from importlib import import_module

species = import_module("src.labelling.species")


class RelevantSpeciesTests(unittest.TestCase):
    def test_relevant_species_come_from_config_file(self):
        configured = json.loads(species.CONFIG_PATH.read_text(encoding="utf-8"))

        self.assertEqual(species.CONFIG_PATH.name, "relevant_species.json")
        self.assertEqual(species.RECENT_SPECIES, frozenset(configured))
        self.assertEqual(
            species.RECENT_SPECIES,
            {"roe deer", "wild boar", "badger", "marten", "fox", "hare", "raccoon"},
        )


if __name__ == "__main__":
    unittest.main()
