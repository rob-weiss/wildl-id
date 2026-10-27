import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from PIL import Image

organizer = import_module("src.labelling.organize_images")
labeler = import_module("src.labelling.label_images")


class OrganizerFilenameTimestampTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.images_dir = self.root / "images"
        self.location_dir = self.data_dir / "Grenzweg"
        self.location_dir.mkdir(parents=True)
        self.images_dir.mkdir()
        self.csv_path = self.root / "detections.csv"
        self.logs_dir = self.root / "logs"
        self.patches = patch.multiple(
            organizer,
            REPO_DIR=self.root,
            DATA_DIR=self.data_dir,
            IMAGES_DIR=self.images_dir,
            LOG_DIR=self.logs_dir,
            CSV_PATH=self.csv_path,
            ANNOTATED_DIR=self.root / "annotated",
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def create_image(self, name):
        path = self.location_dir / name
        Image.new("RGB", (16, 12), (31, 82, 147)).save(path, format="JPEG")
        with Image.open(path) as image:
            return path, image.convert("RGB").tobytes()

    def exif_capture_time(self, path):
        with Image.open(path) as image:
            exif = image.getexif()
            exif_ifd = exif.get_ifd(0x8769)
        return exif_ifd.get(0x9003), exif_ifd.get(0x9004), exif.get(0x0132)

    def test_organizer_writes_standard_time_from_filename_and_labeler_reads_it(self):
        winter, winter_pixels = self.create_image("Aufnahme_260115_1200_CAM1.jpg")
        summer, summer_pixels = self.create_image("Aufnahme_260715_1200_CAM1.jpg")

        organizer.organize_images(workers=1)

        winter_target = self.images_dir / "2026" / "Grenzweg" / "image_0000001.jpg"
        summer_target = self.images_dir / "2026" / "Grenzweg" / "image_0000002.jpg"
        self.assertFalse(winter.exists())
        self.assertFalse(summer.exists())
        self.assertEqual(
            self.exif_capture_time(winter_target),
            ("2026:01:15 12:00:00", "2026:01:15 12:00:00", "2026:01:15 12:00:00"),
        )
        self.assertEqual(
            self.exif_capture_time(summer_target),
            ("2026:07:15 11:00:00", "2026:07:15 11:00:00", "2026:07:15 11:00:00"),
        )
        self.assertEqual(
            labeler.extract_exif_timestamp(winter_target), "2026-01-15T12:00:00"
        )
        self.assertEqual(
            labeler.extract_exif_timestamp(summer_target), "2026-07-15T11:00:00"
        )
        with Image.open(winter_target) as image:
            self.assertEqual(image.convert("RGB").tobytes(), winter_pixels)
        with Image.open(summer_target) as image:
            self.assertEqual(image.convert("RGB").tobytes(), summer_pixels)

    def test_dry_run_uses_filename_timestamp_without_modifying_metadata(self):
        source, _ = self.create_image("Aufnahme_260715_1200_CAM1.jpg")

        organizer.organize_images(dry_run=True, workers=1)

        self.assertTrue(source.exists())
        self.assertEqual(self.exif_capture_time(source), (None, None, None))
        self.assertFalse((self.images_dir / "2026" / "Grenzweg").exists())

    def test_unrecognized_filename_without_exif_stays_in_data(self):
        source, _ = self.create_image("camera-export.jpg")
        empty_folder = self.data_dir / "Unused" / "Empty"
        empty_folder.mkdir(parents=True)

        organizer.organize_images(workers=1)

        self.assertTrue(source.exists())
        self.assertEqual(self.exif_capture_time(source), (None, None, None))
        self.assertTrue(self.data_dir.exists())
        self.assertFalse(empty_folder.parent.exists())
        self.assertFalse((self.images_dir / "2026" / "Grenzweg").exists())

    def test_removes_data_tree_after_all_timestamped_images_are_organized(self):
        self.create_image("Aufnahme_261026_2141_ZWBTI.jpg")
        (self.data_dir / "Unused" / "Empty").mkdir(parents=True)

        organizer.organize_images(workers=1)

        self.assertFalse(self.data_dir.exists())
        self.assertTrue(
            (self.images_dir / "2026" / "Grenzweg" / "image_0000001.jpg").exists()
        )

    def test_dry_run_does_not_remove_empty_data_directories(self):
        empty_folder = self.data_dir / "Unused" / "Empty"
        empty_folder.mkdir(parents=True)

        organizer.organize_images(dry_run=True, workers=1)

        self.assertTrue(empty_folder.exists())


if __name__ == "__main__":
    unittest.main()
