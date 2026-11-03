import json
import unittest
from importlib import import_module

species = import_module("src.labelling.species")


class RelevantSpeciesTests(unittest.TestCase):
    def test_class_emojis_match_approved_list(self):
        expected = {
            "badger": "\N{BADGER}",
            "bear": "\N{BEAR FACE}",
            "beaver": "\N{BEAVER}",
            "bird": "\N{BIRD}",
            "bison": "\N{BISON}",
            "cat": "\N{CAT}",
            "chamois": "\N{GOAT}",
            "cow": "\N{COW}",
            "crow": "\N{BIRD}\N{ZERO WIDTH JOINER}\N{BLACK LARGE SQUARE}",
            "dog": "\N{DOG}",
            "fallow deer": "\N{DEER}",
            "fox": "\N{FOX FACE}",
            "genet": "\N{CAT}",
            "goat": "\N{GOAT}",
            "hare": "\N{RABBIT}",
            "hedgehog": "\N{HEDGEHOG}",
            "horse": "\N{HORSE}",
            "ibex": "\N{GOAT}",
            "lynx": "\N{CAT}",
            "marmot": "\N{BEAVER}",
            "marten": "\N{OTTER}",
            "moose": "\N{MOOSE}",
            "mouflon": "\N{RAM}",
            "mouse": "\N{MOUSE}",
            "nutria": "\N{BEAVER}",
            "otter": "\N{OTTER}",
            "pigeon": "\N{DOVE OF PEACE}\N{VARIATION SELECTOR-16}",
            "raccoon": "\N{RACCOON}",
            "red deer": "\N{DEER}",
            "reindeer": "\N{DEER}",
            "roe deer": "\N{DEER}",
            "sheep": "\N{SHEEP}",
            "squirrel": "\N{CHIPMUNK}\N{VARIATION SELECTOR-16}",
            "wild boar": "\N{BOAR}",
            "wolf": "\N{WOLF FACE}",
            "wolverine": "\N{BADGER}",
            "human": "\N{STANDING PERSON}",
            "vehicle": "\N{AUTOMOBILE}",
            "none": "\N{NO ENTRY SIGN}",
            "unknown": "\N{BLACK QUESTION MARK ORNAMENT}",
        }
        self.assertEqual(species.LABEL_EMOJIS, expected)
        for label, emoji in expected.items():
            with self.subTest(label=label):
                self.assertEqual(species.species_emoji(label), emoji)

    def test_unknown_class_emoji_keeps_generic_fallback(self):
        self.assertEqual(species.species_emoji("new species"), "\N{PAW PRINTS}")

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
                "bird",
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
