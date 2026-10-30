import csv
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from nicegui import run, ui
from nicegui.testing.user import User
from PIL import Image
from web import app as web_app

review_module = import_module("src.labelling.review_app")


def call_cloud_image_url(image_path, images_root, base_url):
    mapper = getattr(review_module, "cloud_image_url", None)
    return mapper(image_path, images_root, base_url) if mapper else None


class FakeVisualisationRunStore:
    def __init__(self, current_directory=None, result=None, error=None):
        self.current_directory = current_directory
        self.result = result
        self.error = error
        self.is_running = False
        self.generate_calls = 0

    async def generate(self, csv_path):
        self.generate_calls += 1
        if self.error is not None:
            raise self.error
        self.current_directory = self.result
        return self.result


class WebAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.csv_path = self.root / "labels.csv"
        self.csv_patch = patch.object(
            web_app, "DEFAULT_CSV", self.csv_path, create=True
        )
        self.csv_patch.start()
        self.addCleanup(self.csv_patch.stop)

    @asynccontextmanager
    async def http_client(self):
        try:
            async with web_app.app.router.lifespan_context(web_app.app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=web_app.app),
                    base_url="http://test",
                ) as client:
                    yield client
        finally:
            run.reset()

    @asynccontextmanager
    async def app_user(self, path):
        async with self.http_client() as client:
            response = await client.get(path)
            self.assertEqual(response.headers.get("X-NiceGUI-Content"), "page")
            user = User(client)
            await user.open(path)
            yield user

    def patch_run_store(self, store):
        run_store_patch = patch.object(
            web_app, "visualisation_runs", store, create=True
        )
        run_store_patch.start()
        self.addCleanup(run_store_patch.stop)

    def write_csv(self, fieldnames, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_valid_review_data(self):
        self.write_csv(
            ["location_id", "image_file", "class"],
            [{"location_id": "camera1", "image_file": "photo.png", "class": "fox"}],
        )
        annotated_images = self.root / "images"
        annotated_images.mkdir()
        Image.new("RGB", (16, 16), "green").save(annotated_images / "camera1_photo.png")

    def test_cloud_image_url_encodes_nested_path_segments(self):
        images_root = self.root / "images"
        image_path = images_root / "2026" / "München" / "fox 1.jpg"

        self.assertEqual(
            call_cloud_image_url(
                image_path, images_root, "https://cdn.example.test/images/"
            ),
            "https://cdn.example.test/images/2026/M%C3%BCnchen/fox%201.jpg",
        )

    def test_cloud_image_url_returns_none_without_base_url(self):
        images_root = self.root / "images"

        self.assertIsNone(
            call_cloud_image_url(images_root / "photo.jpg", images_root, None)
        )

    def test_cloud_image_url_returns_none_for_annotated_fallback(self):
        images_root = self.root / "images"
        annotated_path = self.root / "labels" / "images" / "camera1_photo.png"

        self.assertIsNone(
            call_cloud_image_url(
                annotated_path, images_root, "https://cdn.example.test/images"
            )
        )

    async def test_review_page_passes_cloud_image_base_url(self):
        self.write_valid_review_data()
        reviewer = Mock()
        base_url = "https://cdn.example.test/images"

        with (
            patch.dict(os.environ, {"WILDL_ID_IMAGE_BASE_URL": base_url}),
            patch.object(web_app, "ReviewApp", return_value=reviewer) as factory,
        ):
            async with self.app_user("/review") as user:
                await user.should_see(kind=ui.image)

        self.assertGreater(factory.call_count, 0)
        for call_args in factory.call_args_list:
            self.assertEqual(call_args.args, (self.csv_path,))
            self.assertEqual(call_args.kwargs, {"image_base_url": base_url})

    async def test_review_viewer_uses_cloud_url_and_keeps_local_image_path(self):
        self.write_valid_review_data()
        images_root = self.root / "source-images"
        local_image = images_root / "2026" / "camera1" / "photo.png"
        local_image.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16), "green").save(local_image)
        base_url = "https://cdn.example.test/images"
        created_reviewers = []
        review_app_class = web_app.ReviewApp

        def create_reviewer(csv_path, image_base_url=None):
            reviewer = review_app_class(csv_path, image_base_url=image_base_url)
            created_reviewers.append(reviewer)
            return reviewer

        with (
            patch.dict(os.environ, {"WILDL_ID_IMAGE_BASE_URL": base_url}),
            patch.object(web_app, "ReviewApp", side_effect=create_reviewer),
            patch.object(review_module, "IMAGES_DIR", images_root),
            patch.object(
                review_module.LabelReview, "original_path", return_value=local_image
            ),
        ):
            async with self.app_user("/review") as user:
                await user.should_see("Wildlife label review")

        reviewer = created_reviewers[-1]
        self.assertIn(
            "https://cdn.example.test/images/2026/camera1/photo.png",
            str(reviewer.image._props),
        )
        self.assertEqual(reviewer.current_image_path, local_image)
        self.assertEqual(reviewer.image_size, (16, 16))

    async def test_review_page_shows_labels_and_hides_remove_empty(self):
        self.write_valid_review_data()

        async with self.app_user("/review") as user:
            await user.should_see("Wildlife label review")
            await user.should_see("fox")
            await user.should_not_see("Remove empty")

    async def test_page_shows_missing_csv_error(self):
        async with self.app_user("/review") as user:
            await user.should_see("Review data unavailable")

    async def test_page_shows_invalid_csv_schema_error(self):
        self.write_csv(["invalid"], [{"invalid": "value"}])

        async with self.app_user("/review") as user:
            await user.should_see("Review data unavailable")

    async def test_page_shows_unreadable_csv_error(self):
        self.csv_path.mkdir()

        async with self.app_user("/review") as user:
            await user.should_see("Review data unavailable")

    async def test_home_links_to_review_and_visualisations(self):
        async with self.app_user("/") as user:
            await user.should_see("Manual review")
            await user.should_see("Gallery")
            await user.should_see("Visualisations")

    async def test_gallery_page_applies_smart_filter(self):
        self.write_csv(
            ["location_id", "image_file", "timestamp", "class"],
            [
                {
                    "location_id": "camera1",
                    "image_file": "fox.jpg",
                    "timestamp": "2025-01-01T00:00:00",
                    "class": "fox",
                },
                {
                    "location_id": "camera1",
                    "image_file": "boar.jpg",
                    "timestamp": "2025-01-02T00:00:00",
                    "class": "wild boar",
                },
            ],
        )
        with patch.dict(
            os.environ,
            {"WILDL_ID_IMAGE_BASE_URL": "https://cdn.example.test/images"},
        ):
            async with self.app_user("/gallery") as user:
                await user.should_see("2 images · all images")
                user.find("Smart filter").type("only foxes").trigger("keydown.enter")
                await user.should_see("1 images · only 🦊 fox")

    async def test_gallery_page_reports_missing_csv(self):
        async with self.app_user("/gallery") as user:
            await user.should_see("Gallery data unavailable")

    async def test_homepage_shows_brand_logo(self):
        async with self.app_user("/") as user:
            await user.should_see(kind=ui.image)

    async def test_page_declares_brand_tab_icon(self):
        async with self.http_client() as client:
            response = await client.get("/")
            favicon = await client.get("/favicon.ico")

        self.assertIn('href="/favicon.ico"', response.text)
        self.assertNotIn("/assets/wildl-id-icon.svg", response.text)
        self.assertEqual(favicon.status_code, 200)
        self.assertIn("image/svg+xml", favicon.headers["content-type"])
        self.assertEqual(
            favicon.content,
            (web_app.ASSETS_DIRECTORY / "wildl-id-icon.svg").read_bytes(),
        )

    async def test_visualisations_page_does_not_generate_on_open(self):
        run_store = FakeVisualisationRunStore()
        self.patch_run_store(run_store)

        async with self.app_user("/visualisations") as user:
            await user.should_see("Generate visualisations")

        self.assertEqual(run_store.generate_calls, 0)

    async def test_successful_run_shows_summary_and_selected_chart(self):
        output_dir = self.root / "results"
        output_dir.mkdir()
        (output_dir / "summary_statistics.txt").write_text(
            "24 images analysed", encoding="utf-8"
        )
        (output_dir / "01_species_distribution.svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
        )
        run_store = FakeVisualisationRunStore(result=output_dir)
        self.patch_run_store(run_store)

        async with self.app_user("/visualisations") as user:
            user.find("Generate visualisations").click()
            await user.should_see("Summary statistics")
            await user.should_see("01_species_distribution.svg")
            await user.should_see(kind=ui.image)
            user.find("Summary statistics").click()
            await user.should_see("24 images analysed")

        self.assertEqual(run_store.generate_calls, 1)

    async def test_failed_run_shows_error_and_preserves_previous_chart(self):
        output_dir = self.root / "previous-results"
        output_dir.mkdir()
        (output_dir / "summary_statistics.txt").write_text(
            "previous successful run", encoding="utf-8"
        )
        (output_dir / "01_species_distribution.svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
        )
        run_store = FakeVisualisationRunStore(
            current_directory=output_dir,
            error=RuntimeError("fixture generation failure"),
        )
        self.patch_run_store(run_store)

        async with self.app_user("/visualisations") as user:
            user.find("Generate visualisations").click()
            await user.should_see(
                "Visualisations failed; see container logs for details."
            )
            await user.should_not_see("fixture generation failure")
            await user.should_see("previous successful run")
            await user.should_see("01_species_distribution.svg")

        self.assertEqual(run_store.generate_calls, 1)

    async def test_health_returns_ok_json(self):
        async with self.http_client() as client:
            response = await client.get("/health")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    async def test_brand_logo_and_tab_icon_are_served(self):
        async with self.http_client() as client:
            logo = await client.get("/assets/wildl-id-logo.svg")
            icon = await client.get("/assets/wildl-id-icon.svg")

        self.assertEqual(logo.status_code, 200)
        self.assertEqual(icon.status_code, 200)
        self.assertIn("image/svg+xml", logo.headers["content-type"])
        self.assertIn("image/svg+xml", icon.headers["content-type"])
        self.assertIn("<svg", logo.text)
        self.assertIn("<svg", icon.text)
        for animal in ("\U0001f98c", "\U0001f9a1", "\U0001f986"):
            self.assertIn(animal, logo.text)
            self.assertIn(animal, icon.text)

    async def test_unknown_route_returns_404(self):
        async with self.http_client() as client:
            response = await client.get("/missing")

            self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
