import csv
import os
import tempfile
import unittest
from datetime import datetime
from importlib import import_module
from pathlib import Path
from threading import Barrier
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

    def duplicate_fixture(self, manual_labels=("", "")):
        source, _ = self.create_image("camera-export.jpg")
        directory = self.images_dir / "2026" / "Grenzweg"
        directory.mkdir(parents=True)
        first = directory / "image_0000001.jpg"
        second = directory / "image_0000002.jpg"
        first.write_bytes(source.read_bytes())
        second.write_bytes(source.read_bytes())
        source.unlink()
        organizer.write_exif_capture_time(second, datetime(2026, 1, 15, 12))
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["location_id", "image_file", "manual_label"]
            )
            writer.writeheader()
            for path, label in zip((first, second), manual_labels):
                writer.writerow(
                    dict(
                        location_id="Grenzweg", image_file=path.name, manual_label=label
                    )
                )
        annotated = self.root / "annotated" / f"Grenzweg_{second.name}"
        annotated.parent.mkdir()
        annotated.write_bytes(second.read_bytes())
        return first, second, annotated

    def test_duplicates_are_not_checked_by_default(self):
        first, second, _ = self.duplicate_fixture()

        organizer.organize_images(workers=1)

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_duplicate_check_ignores_metadata_and_backs_up_images_and_labels(self):
        first, second, annotated = self.duplicate_fixture()
        original_csv = self.csv_path.read_bytes()
        duplicate_bytes = second.read_bytes()

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertFalse(second.exists())
        self.assertFalse(annotated.exists())
        _, rows = organizer.read_csv(self.csv_path)
        self.assertEqual([row["image_file"] for row in rows], [first.name])
        backup = next(self.logs_dir.glob("duplicate_images_*"))
        self.assertEqual(
            (backup / second.relative_to(self.root)).read_bytes(), duplicate_bytes
        )
        self.assertEqual((backup / "labels.before.csv").read_bytes(), original_csv)
        self.assertTrue((backup / annotated.relative_to(self.root)).exists())

    def test_duplicate_check_dry_run_changes_nothing(self):
        first, second, annotated = self.duplicate_fixture()
        original_csv = self.csv_path.read_bytes()

        organizer.organize_images(dry_run=True, workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertTrue(annotated.exists())
        self.assertEqual(self.csv_path.read_bytes(), original_csv)
        self.assertFalse(self.logs_dir.exists())

    def test_duplicate_check_keeps_human_reviewed_copy(self):
        first, second, _ = self.duplicate_fixture(manual_labels=("", "fox"))

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        _, rows = organizer.read_csv(self.csv_path)
        self.assertEqual(rows[0]["manual_label"], "fox")

    def test_duplicate_check_preserves_conflicting_human_labels(self):
        first, second, _ = self.duplicate_fixture(manual_labels=("fox", "badger"))
        original_csv = self.csv_path.read_bytes()

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(self.csv_path.read_bytes(), original_csv)

    def test_duplicate_check_preserves_identical_images_from_other_locations(self):
        first, second, _ = self.duplicate_fixture()
        other = self.images_dir / "2026" / "Other" / second.name
        other.parent.mkdir(parents=True)
        second.rename(other)

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertTrue(other.exists())

    def test_duplicate_check_removes_staged_copy_before_organizing(self):
        first, _, _ = self.duplicate_fixture()
        staged, _ = self.create_image("Aufnahme_260115_1200_CAM1.jpg")

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertFalse(staged.exists())
        self.assertEqual(list(self.images_dir.rglob("*.jpg")), [first])

    def test_staged_copy_with_keeper_filename_preserves_keeper_csv_row(self):
        first, _, _ = self.duplicate_fixture()
        staged = self.location_dir / first.name
        staged.write_bytes(first.read_bytes())

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertFalse(staged.exists())
        _, rows = organizer.read_csv(self.csv_path)
        self.assertEqual([row["image_file"] for row in rows], [first.name])

    def test_duplicate_check_keeps_different_pixels_with_same_dimensions(self):
        first, second, _ = self.duplicate_fixture()
        Image.new("RGB", (16, 12), "red").save(second, format="JPEG")

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_duplicate_check_preserves_distinct_high_bit_depth_pixels(self):
        directory = self.images_dir / "2026" / "Grenzweg"
        directory.mkdir(parents=True)
        first = directory / "image_0000001.png"
        second = directory / "image_0000002.png"
        Image.new("I;16", (16, 12), 1024).save(first)
        Image.new("I;16", (16, 12), 2048).save(second)

        organizer.organize_images(workers=1, check_duplicates=True)

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_duplicate_hashes_run_concurrently_with_identical_results(self):
        self.duplicate_fixture()
        expected = organizer.find_duplicate_images(workers=1, write_cache=False)
        barrier = Barrier(2)
        original_open = Image.open

        def open_together(path):
            barrier.wait(timeout=5)
            return original_open(path)

        with patch.object(organizer.Image, "open", side_effect=open_together):
            actual = organizer.find_duplicate_images(workers=2)

        self.assertEqual(actual, expected)

    def test_organizer_passes_workers_to_duplicate_scan(self):
        with patch.object(organizer, "find_duplicate_images", return_value=[]) as scan:
            organizer.organize_images(dry_run=True, workers=2, check_duplicates=True)

        scan.assert_called_once_with(workers=2, write_cache=False)

    def test_parallel_duplicate_scan_skips_unreadable_images(self):
        first, second, _ = self.duplicate_fixture()
        broken = first.parent / "broken.jpg"
        broken.write_bytes(b"not an image")

        groups = organizer.find_duplicate_images(workers=2)

        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]), {first, second})
        self.assertTrue(broken.exists())

    def test_hash_cache_reuses_unlabelled_image_hashes_without_decoding(self):
        image, _ = self.create_image("camera-export.jpg")
        organizer.find_duplicate_images(workers=1)

        with patch.object(organizer, "duplicate_image_digest") as digest:
            self.assertEqual(organizer.find_duplicate_images(workers=2), [])

        digest.assert_not_called()
        cache = self.logs_dir / "image_hashes.csv"
        _, rows = organizer.read_csv(cache)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_path"], image.relative_to(self.root).as_posix())
        self.assertEqual(len(rows[0]["sha256"]), 64)
        self.assertEqual(rows[0]["size"], str(image.stat().st_size))
        self.assertEqual(rows[0]["mtime_ns"], str(image.stat().st_mtime_ns))

    def test_hash_cache_invalidates_size_and_modification_time(self):
        image, _ = self.create_image("camera-export.jpg")
        organizer.find_duplicate_images(workers=1)
        original_digest = organizer.duplicate_image_digest
        stat = image.stat()
        image.write_bytes(image.read_bytes() + b"padding")
        os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        with patch.object(
            organizer, "duplicate_image_digest", wraps=original_digest
        ) as digest:
            organizer.find_duplicate_images(workers=1)
        digest.assert_called_once_with(image)

        stat = image.stat()
        os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with patch.object(
            organizer, "duplicate_image_digest", wraps=original_digest
        ) as digest:
            organizer.find_duplicate_images(workers=1)
        digest.assert_called_once_with(image)

    def test_hash_cache_rebuilds_invalid_entries_and_old_versions(self):
        image, _ = self.create_image("camera-export.jpg")
        cache = self.logs_dir / "image_hashes.csv"
        original_digest = organizer.duplicate_image_digest
        for field, value in (("hash_version", "old"), ("sha256", "invalid")):
            with self.subTest(field=field):
                organizer.find_duplicate_images(workers=1)
                fields, rows = organizer.read_csv(cache)
                rows[0][field] = value
                organizer.rewrite_csv(cache, rows, fields)

                with patch.object(
                    organizer, "duplicate_image_digest", wraps=original_digest
                ) as digest:
                    organizer.find_duplicate_images(workers=1)

                digest.assert_called_once_with(image)

    def test_hash_cache_discards_deleted_paths(self):
        first, second, _ = self.duplicate_fixture()
        organizer.find_duplicate_images(workers=1)
        second.unlink()

        organizer.find_duplicate_images(workers=1)

        _, rows = organizer.read_csv(self.logs_dir / "image_hashes.csv")
        self.assertEqual(
            [row["image_path"] for row in rows],
            [first.relative_to(self.root).as_posix()],
        )

    def test_dry_run_reads_cache_without_updating_it(self):
        self.duplicate_fixture()
        organizer.find_duplicate_images(workers=1)
        cache = self.logs_dir / "image_hashes.csv"
        before = cache.read_bytes(), cache.stat().st_mtime_ns

        with patch.object(organizer, "duplicate_image_digest") as digest:
            organizer.organize_images(dry_run=True, workers=1, check_duplicates=True)

        digest.assert_not_called()
        self.assertEqual((cache.read_bytes(), cache.stat().st_mtime_ns), before)

    def test_hash_cache_skips_files_changed_during_hashing(self):
        image, _ = self.create_image("camera-export.jpg")
        original_digest = organizer.duplicate_image_digest

        def change_during_hash(path):
            result = original_digest(path)
            path.write_bytes(path.read_bytes() + b"changed")
            return result

        with patch.object(
            organizer, "duplicate_image_digest", side_effect=change_during_hash
        ):
            organizer.find_duplicate_images(workers=1)

        _, rows = organizer.read_csv(self.logs_dir / "image_hashes.csv")
        self.assertEqual(rows, [])

    def test_build_hash_cache_only_does_not_organize_or_remove_duplicates(self):
        first, second, _ = self.duplicate_fixture()
        original_csv = self.csv_path.read_bytes()

        with patch(
            "sys.argv", ["organize_images.py", "--build-hash-cache", "--workers", "2"]
        ):
            organizer.main()

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(self.csv_path.read_bytes(), original_csv)
        _, rows = organizer.read_csv(self.logs_dir / "image_hashes.csv")
        self.assertEqual(len(rows), 2)

    def test_hash_cache_revalidates_hits_after_other_files_finish_hashing(self):
        first, second, _ = self.duplicate_fixture()
        organizer.find_duplicate_images(workers=1)
        stat = second.stat()
        os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        original_digest = organizer.duplicate_image_digest

        def change_cached_image(path):
            result = original_digest(path)
            Image.new("RGB", (16, 12), "red").save(first, format="JPEG")
            return result

        with patch.object(
            organizer, "duplicate_image_digest", side_effect=change_cached_image
        ):
            groups = organizer.find_duplicate_images(workers=1)

        self.assertEqual(groups, [])
        _, rows = organizer.read_csv(self.logs_dir / "image_hashes.csv")
        self.assertEqual(
            [row["image_path"] for row in rows],
            [second.relative_to(self.root).as_posix()],
        )

    def test_build_hash_cache_rejects_dry_run(self):
        with (
            patch(
                "sys.argv", ["organize_images.py", "--build-hash-cache", "--dry-run"]
            ),
            self.assertRaises(SystemExit) as error,
        ):
            organizer.main()

        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.logs_dir.exists())


if __name__ == "__main__":
    unittest.main()
