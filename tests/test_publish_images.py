import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from pathlib import Path
from unittest.mock import patch

publisher = import_module("src.labelling.publish_images")
build_force_upload_command = publisher.build_force_upload_command
build_sync_command = publisher.build_sync_command
publish_images = publisher.publish_images


class PublishImagesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.images_directory = Path(self.temporary_directory.name) / "images"
        self.images_directory.mkdir()

    def stack_output_result(self, outputs=None):
        if outputs is None:
            outputs = {
                "ImagesBucketName": "wildl-id-images-test",
                "CloudFrontDistributionId": "E123EXAMPLE",
            }
        response = {
            "Stacks": [
                {
                    "Outputs": [
                        {"OutputKey": key, "OutputValue": value}
                        for key, value in outputs.items()
                    ]
                }
            ]
        }
        return subprocess.CompletedProcess([], 0, json.dumps(response), "")

    def test_sync_command_mirrors_image_prefix_with_deletions_and_checksums(self):
        command = build_sync_command(Path("/tmp/images"), "image-bucket", False)

        self.assertEqual(
            command,
            [
                "aws",
                "s3",
                "sync",
                "/tmp/images/",
                "s3://image-bucket/images/",
                "--delete",
                "--checksum-algorithm",
                "SHA256",
            ],
        )

    def test_sync_command_dry_run_adds_aws_cli_dryrun_flag(self):
        command = build_sync_command(Path("/tmp/images"), "image-bucket", True)

        self.assertIn("--dryrun", command)
        self.assertIn("--delete", command)

    def test_force_upload_command_targets_only_the_requested_image(self):
        image = self.images_directory / "2026" / "Camera A" / "fox 1.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")

        command = build_force_upload_command(
            self.images_directory, "image-bucket", "2026/Camera A/fox 1.jpg"
        )

        self.assertEqual(
            command,
            [
                "aws",
                "s3",
                "cp",
                str(image),
                "s3://image-bucket/images/2026/Camera A/fox 1.jpg",
                "--checksum-algorithm",
                "SHA256",
            ],
        )

    def test_force_upload_rejects_paths_outside_images(self):
        with self.assertRaises(ValueError):
            build_force_upload_command(
                self.images_directory, "image-bucket", "../outside.jpg"
            )

    @patch("src.labelling.publish_images.subprocess.run")
    def test_dry_run_never_invalidates_cloudfront(self, run):
        run.side_effect = [
            self.stack_output_result(),
            subprocess.CompletedProcess([], 0, "dryrun: upload", ""),
        ]

        output = StringIO()
        with redirect_stdout(output):
            publish_images(
                "wildl-id-images",
                "wildl-id",
                "eu-central-1",
                self.images_directory,
                dry_run=True,
                force_upload=None,
            )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn("--dryrun", commands[1])
        self.assertFalse(any("cloudfront" in command for command in commands))
        self.assertNotIn('"Stacks"', output.getvalue())

    @patch("src.labelling.publish_images.subprocess.run")
    def test_missing_stack_outputs_stop_before_sync(self, run):
        run.return_value = self.stack_output_result({"ImagesBucketName": "bucket"})

        with self.assertRaisesRegex(RuntimeError, "CloudFrontDistributionId"):
            publish_images(
                "wildl-id-images",
                "wildl-id",
                "eu-central-1",
                self.images_directory,
                dry_run=True,
                force_upload=None,
            )

        self.assertEqual(run.call_count, 1)

    @patch("src.labelling.publish_images.subprocess.run")
    def test_failed_sync_does_not_invalidate_cloudfront(self, run):
        run.side_effect = [
            self.stack_output_result(),
            subprocess.CalledProcessError(1, ["aws", "s3", "sync"]),
        ]

        with self.assertRaises(subprocess.CalledProcessError):
            publish_images(
                "wildl-id-images",
                "wildl-id",
                "eu-central-1",
                self.images_directory,
                dry_run=False,
                force_upload=None,
            )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertFalse(any("cloudfront" in command for command in commands))

    @patch("src.labelling.publish_images.subprocess.run")
    def test_successful_sync_invalidates_cloudfront(self, run):
        run.side_effect = [
            self.stack_output_result(),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]

        publish_images(
            "wildl-id-images",
            "wildl-id",
            "eu-central-1",
            self.images_directory,
            dry_run=False,
            force_upload=None,
        )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[2][:3], ["aws", "cloudfront", "create-invalidation"])
        self.assertIn("E123EXAMPLE", commands[2])
        self.assertIn("/*", commands[2])


if __name__ == "__main__":
    unittest.main()
