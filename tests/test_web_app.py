import csv
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from nicegui.testing.user import User
from PIL import Image
from web import app as web_app


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
        async with web_app.app.router.lifespan_context(web_app.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=web_app.app),
                base_url="http://test",
            ) as client:
                yield client

    @asynccontextmanager
    async def review_user(self):
        async with self.http_client() as client:
            response = await client.get("/")
            self.assertEqual(response.headers.get("X-NiceGUI-Content"), "page")
            user = User(client)
            await user.open("/")
            yield user

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

    async def test_review_page_shows_labels_and_hides_remove_empty(self):
        self.write_valid_review_data()

        async with self.review_user() as user:
            await user.should_see("Wildlife label review")
            await user.should_see("fox")
            await user.should_not_see("Remove empty")

    async def test_page_shows_missing_csv_error(self):
        async with self.review_user() as user:
            await user.should_see("Review data unavailable")

    async def test_page_shows_invalid_csv_schema_error(self):
        self.write_csv(["invalid"], [{"invalid": "value"}])

        async with self.review_user() as user:
            await user.should_see("Review data unavailable")

    async def test_page_shows_unreadable_csv_error(self):
        self.csv_path.mkdir()

        async with self.review_user() as user:
            await user.should_see("Review data unavailable")

    async def test_health_returns_ok_json(self):
        async with self.http_client() as client:
            response = await client.get("/health")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    async def test_unknown_route_returns_404(self):
        async with self.http_client() as client:
            response = await client.get("/missing")

            self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
