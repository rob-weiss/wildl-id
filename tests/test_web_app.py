import unittest

from fastapi.testclient import TestClient
from web.app import app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_home_returns_wildl_id_html(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertIn("<title>wildl-id</title>", response.text)
        self.assertNotIn("repository", response.text.lower())
        self.assertNotIn("device", response.text.lower())

    def test_health_returns_ok_json(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_unknown_route_returns_404(self):
        response = self.client.get("/missing")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
