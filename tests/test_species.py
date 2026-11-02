import json
import unittest
from importlib import import_module

species = import_module("src.labelling.species")


class RelevantSpeciesTests(unittest.TestCase):
    def test_game_species_come_from_the_approved_configurable_keep_list(self):
        configured = json.loads(
            species.CONFIG_PATH.with_name("game_species.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(species.GAME_SPECIES, frozenset(configured))
        self.assertEqual(
            species.GAME_SPECIES,
            {
                "roe deer",
                "red deer",
                "fallow deer",
                "wild boar",
                "fox",
                "badger",
                "hare",
                "marten",
                "raccoon",
                "mouflon",
                "chamois",
                "nutria",
                "crow",
                "pigeon",
            },
        )

    def test_normalize_species_keeps_crows_and_pigeons_separate_from_birds(self):
        for label in (
            "bird",
            "other bird",
            "jay",
            "owl",
            "song_bird",
            "song bird",
            "woodpecker",
            "buzzard",
            "falcon",
        ):
            with self.subTest(label=label):
                self.assertEqual(species.normalize_species(label), "bird")
        for label in ("crow", "pigeon", "squirrel"):
            with self.subTest(label=label):
                self.assertEqual(species.normalize_species(label), label)

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
