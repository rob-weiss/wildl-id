import csv
import hashlib
import imaplib
import json
import os
import tempfile
import unittest
from datetime import datetime
from email.header import Header
from email.message import EmailMessage
from importlib import import_module
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

downloader = import_module("src.downloading.download_email_images")


class FakeIMAP4:
    def __init__(
        self,
        mailboxes,
        capabilities=("IMAP4rev1", "UIDPLUS", "MOVE", "SPECIAL-USE"),
        fail_moves=(),
        fail_uid_expunge=False,
        trash_name="Trash",
        login_failure=False,
        login_exception=None,
    ):
        self.mailboxes = mailboxes
        self.selected = "INBOX"
        self.uidvalidity = 812
        self.capabilities = capabilities
        self.fail_moves = set(fail_moves)
        self.fail_uid_expunge = fail_uid_expunge
        self.trash_name = trash_name
        self.login_failure = login_failure
        self.login_exception = login_exception

    def login(self, username, password):
        if self.login_exception:
            raise self.login_exception
        if (
            self.login_failure
            or username != "camera@example.test"
            or password != "test-password"
        ):
            return "NO", [b"authentication failed"]
        return "OK", [b"logged in"]

    def list(self):
        return "OK", [
            b'(\\HasNoChildren \\Inbox) "/" "INBOX"',
            f'(\\HasNoChildren \\Trash) "/" "{self.trash_name}"'.encode(),
        ]

    def capability(self):
        return "OK", [" ".join(self.capabilities).encode()]

    def select(self, mailbox, readonly=False):
        self.selected = mailbox.strip('"')
        if self.selected not in self.mailboxes:
            return "NO", [b"mailbox not found"]
        return "OK", [str(len(self.mailboxes[self.selected])).encode()]

    def response(self, code):
        if code == "UIDVALIDITY":
            return "UIDVALIDITY", [str(self.uidvalidity).encode()]
        return code, [None]

    def uid(self, command, *args):
        command = command.upper()
        mailbox = self.mailboxes[self.selected]
        if command == "SEARCH":
            criteria = " ".join(self._text(arg) for arg in args)
            start = 1
            if " UID " in f" {criteria.upper()} ":
                range_value = criteria.upper().split("UID", 1)[1].strip().split()[0]
                start = int(range_value.split(":", 1)[0])
            uids = [uid for uid in sorted(mailbox, key=int) if int(uid) >= start]
            if not uids and mailbox:
                uids = [max(mailbox, key=int)]
            return "OK", [" ".join(uids).encode()]
        if command == "FETCH":
            uid = self._text(args[0])
            message = mailbox[uid]
            flags = " ".join(sorted(message["flags"]))
            header = f"{uid} (UID {uid} FLAGS ({flags}))".encode()
            return "OK", [(header, message["raw"]), b")"]
        if command == "STORE":
            uid = self._text(args[0])
            operation = self._text(args[1]).upper()
            flag = self._text(args[2]).strip("() ")
            if operation.startswith("+FLAGS"):
                mailbox[uid]["flags"].add(flag)
            elif operation.startswith("-FLAGS"):
                mailbox[uid]["flags"].discard(flag)
            return "OK", [b"stored"]
        if command == "MOVE":
            uid = self._text(args[0])
            if uid in self.fail_moves:
                return "NO", [b"move failed"]
            destination = self._text(args[1]).strip('"')
            message = mailbox.pop(uid)
            target = self.mailboxes[destination]
            new_uid = str(max((int(value) for value in target), default=0) + 1)
            target[new_uid] = message
            return "OK", [b"moved"]
        if command == "COPY":
            uid = self._text(args[0])
            destination = self._text(args[1]).strip('"')
            message = mailbox[uid]
            target = self.mailboxes[destination]
            new_uid = str(max((int(value) for value in target), default=0) + 1)
            target[new_uid] = {"raw": message["raw"], "flags": set(message["flags"])}
            return "OK", [b"copied"]
        if command == "EXPUNGE":
            if self.fail_uid_expunge:
                return "NO", [b"UID EXPUNGE unsupported"]
            uid = self._text(args[0])
            mailbox.pop(uid, None)
            return "OK", [b"expunged"]
        if command == "LOGOUT":
            return "OK", [b"logged out"]
        raise AssertionError(f"Unexpected IMAP UID command: {command} {args!r}")

    def logout(self):
        return "BYE", [b"logged out"]

    @staticmethod
    def _text(value):
        if isinstance(value, bytes):
            return value.decode()
        return str(value)


def make_message(
    subject,
    image=None,
    filename="camera.jpg",
    sender="info@supersim.camera",
    recipient="wildlife@robinweiss.de",
    original_recipient=None,
    encoded_subject=False,
):
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = (
        Header(subject, "utf-8").encode() if encoded_subject else subject
    )
    if original_recipient:
        message["X-Original-To"] = original_recipient
    if image is not None:
        message.add_attachment(
            image,
            maintype="image",
            subtype="jpeg",
            filename=filename,
        )
    return message.as_bytes()


def stored_message(raw, seen=False):
    flags = {"\\Seen"} if seen else set()
    return {"raw": raw, "flags": flags}


class EmailImageDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.images_dir = self.root / "images"
        self.data_dir.mkdir()
        self.images_dir.mkdir()
        self.config_path = self.root / "mail.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "host": "imap.example.test",
                    "port": 993,
                    "username": "camera@example.test",
                    "recipient": "wildlife@robinweiss.de",
                }
            ),
            encoding="utf-8",
        )
        self.state_path = self.root / "state.json"
        self.patches = patch.multiple(
            downloader,
            REPO_DIR=self.root,
            DATA_DIR=self.data_dir,
            IMAGES_DIR=self.images_dir,
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_downloader_reuses_organizer_pixel_cache_for_legacy_state(self):
        organizer = import_module("src.labelling.organize_images")
        content = BytesIO()
        Image.new("RGB", (16, 12), "green").save(content, format="JPEG")
        image_bytes = content.getvalue()
        image_path = self.images_dir / "2026" / "Grenzweg" / "image_0000001.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(image_bytes)
        stat = image_path.stat()
        relative = image_path.relative_to(self.root).as_posix()
        cache_path = self.root / "logs" / "image_hashes.csv"
        cache_path.parent.mkdir()
        with cache_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["image_path", "size", "mtime_ns", "hash_version", "sha256"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": relative,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "hash_version": "rgba-sha256-v1",
                    "sha256": organizer.duplicate_image_digest(image_path).hex(),
                }
            )
        legacy_state = {
            "files": {
                str(image_path.resolve()): {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "hash_version": 1,
                }
            }
        }

        with patch.object(downloader, "_sha256") as hash_file:
            by_location, _ = downloader._image_index(legacy_state)

        hash_file.assert_not_called()
        expected = organizer.duplicate_image_digest(image_path).hex()
        self.assertIn(expected, by_location["Grenzweg"])
        self.assertEqual(
            legacy_state["files"][str(image_path.resolve())]["hash_version"],
            downloader.DOWNLOAD_HASH_VERSION,
        )

    def test_downloader_rehashes_when_shared_pixel_cache_metadata_is_stale(self):
        organizer = import_module("src.labelling.organize_images")
        content = BytesIO()
        Image.new("RGB", (16, 12), "green").save(content, format="JPEG")
        image_path = self.images_dir / "2026" / "Grenzweg" / "image_0000001.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(content.getvalue())
        stat = image_path.stat()
        cache_path = self.root / "logs" / "image_hashes.csv"
        cache_path.parent.mkdir()
        with cache_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["image_path", "size", "mtime_ns", "hash_version", "sha256"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": image_path.relative_to(self.root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "hash_version": "rgba-sha256-v1",
                    "sha256": organizer.duplicate_image_digest(image_path).hex(),
                }
            )
        Image.new("RGB", (16, 12), "red").save(image_path, format="JPEG")

        with patch.object(downloader, "_sha256", wraps=downloader._sha256) as hash_file:
            downloader._image_index({"files": {}})

        hash_file.assert_called_once_with(image_path)

    def run_download(
        self,
        mailboxes=None,
        server=None,
        keychain_password="test-password",
        allow_failures=False,
    ):
        server = server or FakeIMAP4(mailboxes)
        result = downloader.download_email_images(
            config_path=self.config_path,
            state_path=self.state_path,
            imap_factory=lambda host, port: server,
            keychain_reader=lambda service, account: keychain_password,
        )
        if not allow_failures:
            self.assertEqual(result.failures, (), result.failures)
        return result, server

    def test_scans_read_and_unread_messages_and_parses_variable_camera_numbers(self):
        first = b"first camera image"
        second = b"second camera image"
        older = b"older camera image"
        mailboxes = {
            "INBOX": {
                "2": stored_message(
                    make_message(
                        "Kamera: Jochen 14 Madenhäule Nord", second, "north.jpg"
                    ),
                    seen=True,
                ),
                "5": stored_message(
                    make_message(
                        "Kamera: Jochen 06 Madenhäule Nord", first, "south.jpg"
                    )
                ),
            },
            "Trash": {
                "17": stored_message(
                    make_message("Kamera: Jochen 3 Grenzweg", older, "older.jpg"),
                    seen=True,
                )
            },
        }

        result, server = self.run_download(mailboxes)

        location = self.data_dir / "Madenhäule Nord"
        self.assertEqual((location / "north.jpg").read_bytes(), second)
        self.assertEqual((location / "south.jpg").read_bytes(), first)
        self.assertEqual((self.data_dir / "Grenzweg" / "older.jpg").read_bytes(), older)
        self.assertEqual(result.saved_images, 3)
        self.assertTrue(result.pending_organization)
        self.assertEqual(server.mailboxes["INBOX"], {})
        self.assertEqual(len(server.mailboxes["Trash"]), 3)
        self.assertTrue(
            all(
                "\\Seen" in message["flags"]
                for message in server.mailboxes["Trash"].values()
            )
        )

    def test_decodes_rfc2047_subject_before_extracting_location(self):
        image = b"encoded subject image"
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message(
                        "Kamera: Jochen 21 Madenhäule Nord",
                        image,
                        encoded_subject=True,
                    )
                )
            },
            "Trash": {},
        }

        result, _ = self.run_download(mailboxes)

        self.assertEqual(result.saved_images, 1)
        self.assertEqual(
            (self.data_dir / "Madenhäule Nord" / "camera.jpg").read_bytes(), image
        )

    def test_matches_original_recipient_alias_and_rejects_unrelated_mail(self):
        image = b"alias delivered image"
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message(
                        "Kamera: Jochen 06 Grenzweg",
                        image,
                        original_recipient="wildlife@robinweiss.de",
                        recipient="camera-archive@example.test",
                    )
                ),
                "2": stored_message(
                    make_message(
                        "Kamera: Jochen 06 Grenzweg",
                        b"wrong sender",
                        sender="other@example.test",
                    )
                ),
                "3": stored_message(
                    make_message(
                        "Kamera: Jochen 06 Grenzweg",
                        b"wrong recipient",
                        recipient="other@example.test",
                    )
                ),
                "4": stored_message(
                    make_message("Monthly camera report", b"wrong subject")
                ),
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes)

        self.assertEqual(
            (self.data_dir / "Grenzweg" / "camera.jpg").read_bytes(), image
        )
        self.assertEqual(result.saved_images, 1)
        self.assertEqual(set(server.mailboxes["INBOX"]), {"2", "3", "4"})

    def test_existing_organized_image_prevents_duplicate_download(self):
        image = b"same bytes as an organized image"
        organized = self.images_dir / "2025" / "Grenzweg" / "image_0000001.jpg"
        organized.parent.mkdir(parents=True)
        organized.write_bytes(image)
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message("Kamera: Jochen 6 Grenzweg", image, "camera.jpg")
                )
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes)

        self.assertFalse((self.data_dir / "Grenzweg" / "camera.jpg").exists())
        self.assertEqual(result.saved_images, 0)
        self.assertEqual(result.duplicates, 1)
        self.assertEqual(server.mailboxes["INBOX"], {})
        self.assertEqual(len(server.mailboxes["Trash"]), 1)

    def test_organized_exif_timestamp_prevents_duplicate_with_legacy_cache(self):
        organizer = import_module("src.labelling.organize_images")
        content = BytesIO()
        Image.new("RGB", (640, 480), "green").save(content, format="JPEG")
        image = content.getvalue()
        organized = self.images_dir / "2026" / "Grenzweg" / "image_0000001.jpg"
        organized.parent.mkdir(parents=True)
        organized.write_bytes(image)
        organizer.write_exif_capture_time(organized, datetime(2026, 10, 27, 12, 30))
        stat = organized.stat()
        self.state_path.write_text(
            json.dumps(
                {
                    "mailboxes": {},
                    "files": {
                        str(organized.resolve()): {
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                            "sha256": hashlib.sha256(
                                organized.read_bytes()
                            ).hexdigest(),
                        }
                    },
                }
            )
        )
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message(
                        "Kamera: Jochen 6 Grenzweg",
                        image,
                        "Aufnahme_261027_1230_test.jpg",
                    )
                )
            },
            "Trash": {},
        }

        result, _ = self.run_download(mailboxes)

        self.assertEqual(result.saved_images, 0)
        self.assertEqual(result.duplicates, 1)
        self.assertFalse(result.pending_organization)
        self.assertFalse((self.data_dir / "Grenzweg").exists())

    def test_existing_staged_image_prevents_duplicate_and_requests_organization(self):
        image = b"same bytes as a staged image"
        staged = self.data_dir / "Grenzweg" / "existing.jpg"
        staged.parent.mkdir()
        staged.write_bytes(image)
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message("Kamera: Jochen 6 Grenzweg", image, "camera.jpg")
                )
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes)

        self.assertEqual(result.saved_images, 0)
        self.assertEqual(result.duplicates, 1)
        self.assertTrue(result.pending_organization)
        self.assertFalse((self.data_dir / "Grenzweg" / "camera.jpg").exists())
        self.assertEqual(server.mailboxes["INBOX"], {})

    def test_distinct_attachments_with_same_filename_are_preserved(self):
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message("Kamera: Jochen 6 Grenzweg", b"image one", "same.jpg")
                ),
                "2": stored_message(
                    make_message("Kamera: Jochen 6 Grenzweg", b"image two", "same.jpg")
                ),
            },
            "Trash": {},
        }

        result, _ = self.run_download(mailboxes)

        files = list((self.data_dir / "Grenzweg").glob("*.jpg"))
        self.assertEqual(result.saved_images, 2)
        self.assertEqual(len(files), 2)
        self.assertEqual(
            {path.read_bytes() for path in files}, {b"image one", b"image two"}
        )

    def test_messages_without_matching_image_attachment_remain_unchanged(self):
        mailboxes = {
            "INBOX": {
                "1": stored_message(make_message("Kamera: Jochen 6 Grenzweg")),
                "2": stored_message(
                    make_message("Monthly camera report", b"not matching")
                ),
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes)

        self.assertEqual(result.saved_images, 0)
        self.assertFalse(result.pending_organization)
        self.assertEqual(set(server.mailboxes["INBOX"]), {"1", "2"})
        self.assertFalse(self.data_dir.joinpath("Grenzweg").exists())

    def test_uidvalidity_change_rescans_mailbox_from_first_uid(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "8": stored_message(
                        make_message(
                            "Kamera: Jochen 6 Grenzweg", b"first image", "first.jpg"
                        )
                    )
                },
                "Trash": {},
            }
        )
        first, _ = self.run_download(server=server)
        self.assertEqual(first.saved_images, 1)

        server.uidvalidity += 1
        server.mailboxes["INBOX"]["1"] = stored_message(
            make_message("Kamera: Jochen 6 Grenzweg", b"new uid space", "second.jpg")
        )
        second, _ = self.run_download(server=server)

        self.assertEqual(second.saved_images, 1)
        self.assertTrue((self.data_dir / "Grenzweg" / "second.jpg").exists())

    def test_completed_trash_uids_are_not_reprocessed_each_poll(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "1": stored_message(
                        make_message("Kamera: Jochen 6 Grenzweg", b"single image")
                    )
                },
                "Trash": {},
            }
        )

        first, _ = self.run_download(server=server)
        second, _ = self.run_download(server=server)
        third, _ = self.run_download(server=server)

        self.assertEqual(first.saved_images, 1)
        self.assertEqual(second.duplicates, 1)
        self.assertEqual(third.duplicates, 0)

    def test_discovers_nonstandard_trash_mailbox_by_special_use_flag(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "1": stored_message(
                        make_message(
                            "Kamera: Jochen 6 Grenzweg", b"move to special trash"
                        )
                    )
                },
                "Deleted Messages": {},
            },
            trash_name="Deleted Messages",
        )

        result, _ = self.run_download(server=server)

        self.assertEqual(result.failures, ())
        self.assertEqual(server.mailboxes["INBOX"], {})
        self.assertEqual(len(server.mailboxes["Deleted Messages"]), 1)

    def test_changed_file_metadata_invalidates_cached_hash(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "1": stored_message(
                        make_message("Kamera: Jochen 6 Grenzweg", b"camera image")
                    )
                },
                "Trash": {},
            }
        )
        first, _ = self.run_download(server=server)
        self.assertEqual(first.saved_images, 1)

        existing = self.data_dir / "Grenzweg" / "camera.jpg"
        cached_files = json.loads(self.state_path.read_text(encoding="utf-8"))["files"]
        self.assertIn(str(existing.resolve()), cached_files)
        original_mtime = existing.stat().st_mtime_ns
        existing.write_bytes(b"edited image")
        os.utime(existing, ns=(original_mtime + 1_000_000, original_mtime + 1_000_000))
        second, _ = self.run_download(server=server)

        self.assertEqual(second.saved_images, 1)
        recovered = list((self.data_dir / "Grenzweg").glob("camera_*.jpg"))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].read_bytes(), b"camera image")

    def test_authentication_failure_does_not_expose_password(self):
        server = FakeIMAP4(
            {"INBOX": {}, "Trash": {}},
            login_failure=True,
        )

        result, _ = self.run_download(
            server=server,
            keychain_password="wrong-password",
            allow_failures=True,
        )

        self.assertIn("IMAP authentication failed", " ".join(result.failures))
        self.assertNotIn("wrong-password", " ".join(result.failures))

    def test_server_login_exception_is_sanitized(self):
        server = FakeIMAP4(
            {"INBOX": {}, "Trash": {}},
            login_exception=imaplib.IMAP4.error(
                "AUTHENTICATIONFAILED password=do-not-log"
            ),
        )

        result, _ = self.run_download(server=server, allow_failures=True)

        self.assertIn("IMAP authentication failed", " ".join(result.failures))
        self.assertNotIn("do-not-log", " ".join(result.failures))

    def test_default_imap_connection_uses_a_finite_timeout(self):
        server = FakeIMAP4({"INBOX": {}, "Trash": {}})
        with patch.object(
            downloader.imaplib, "IMAP4_SSL", return_value=server
        ) as factory:
            result = downloader.download_email_images(
                config_path=self.config_path,
                state_path=self.state_path,
                keychain_reader=lambda service, account: "test-password",
            )

        self.assertEqual(result.failures, ())
        factory.assert_called_once_with("imap.example.test", 993, timeout=30)

    def test_unsafe_location_does_not_write_outside_data(self):
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message("Kamera: Jochen 6 ../outside", b"unsafe path")
                )
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes)

        self.assertEqual(result.saved_images, 0)
        self.assertEqual(set(server.mailboxes["INBOX"]), {"1"})
        self.assertFalse((self.root / "outside").exists())

    def test_symlinked_location_cannot_redirect_download_outside_data(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.data_dir / "Grenzweg").symlink_to(outside, target_is_directory=True)
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message("Kamera: Jochen 6 Grenzweg", b"symlink escape")
                )
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes, allow_failures=True)

        self.assertIn("symlink", " ".join(result.failures).lower())
        self.assertFalse((outside / "camera.jpg").exists())
        self.assertEqual(set(server.mailboxes["INBOX"]), {"1"})

    def test_unsafe_attachment_filename_is_not_written(self):
        mailboxes = {
            "INBOX": {
                "1": stored_message(
                    make_message(
                        "Kamera: Jochen 6 Grenzweg",
                        b"unsafe filename",
                        "../outside.jpg",
                    )
                )
            },
            "Trash": {},
        }

        result, server = self.run_download(mailboxes, allow_failures=True)

        self.assertIn("unsafe filename", " ".join(result.failures))
        self.assertEqual(set(server.mailboxes["INBOX"]), {"1"})
        self.assertFalse((self.root / "outside.jpg").exists())

    def test_failed_move_retries_staged_image_without_losing_it(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "1": stored_message(
                        make_message("Kamera: Jochen 6 Grenzweg", b"retry me")
                    )
                },
                "Trash": {},
            },
            capabilities=("IMAP4rev1", "MOVE"),
            fail_moves={"1"},
        )

        failed, _ = self.run_download(server=server, allow_failures=True)

        self.assertTrue(failed.failures)
        self.assertTrue(failed.pending_organization)
        self.assertTrue((self.data_dir / "Grenzweg" / "camera.jpg").exists())
        self.assertIn("1", server.mailboxes["INBOX"])

        server.fail_moves.clear()
        retried, _ = self.run_download(server=server)

        self.assertEqual(retried.failures, ())
        self.assertEqual(retried.duplicates, 1)
        self.assertTrue(retried.pending_organization)
        self.assertEqual(server.mailboxes["INBOX"], {})

    def test_uidplus_fallback_moves_only_matching_uid(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "1": stored_message(
                        make_message("Kamera: Jochen 6 Grenzweg", b"move me")
                    ),
                    "2": stored_message(make_message("monthly report", b"leave me")),
                },
                "Trash": {},
            },
            capabilities=("IMAP4rev1", "UIDPLUS"),
        )

        result, _ = self.run_download(server=server, allow_failures=True)

        self.assertEqual(result.failures, ())
        self.assertEqual(set(server.mailboxes["INBOX"]), {"2"})
        self.assertEqual(len(server.mailboxes["Trash"]), 1)

    def test_failed_uid_expunge_preserves_source_and_unrelated_messages(self):
        server = FakeIMAP4(
            {
                "INBOX": {
                    "1": stored_message(
                        make_message("Kamera: Jochen 6 Grenzweg", b"keep retryable")
                    ),
                    "2": stored_message(make_message("monthly report", b"unrelated")),
                },
                "Trash": {},
            },
            capabilities=("IMAP4rev1", "UIDPLUS"),
            fail_uid_expunge=True,
        )

        result, _ = self.run_download(server=server, allow_failures=True)

        self.assertIn("UID EXPUNGE", " ".join(result.failures))
        self.assertEqual(set(server.mailboxes["INBOX"]), {"1", "2"})
        self.assertNotIn("\\Deleted", server.mailboxes["INBOX"]["1"]["flags"])
        self.assertEqual(len(server.mailboxes["Trash"]), 1)

    def test_missing_keychain_password_fails_without_exposing_a_secret(self):
        result, server = self.run_download(
            {"INBOX": {}, "Trash": {}},
            keychain_password="",
            allow_failures=True,
        )

        self.assertIn("Keychain", " ".join(result.failures))
        self.assertNotIn("test-password", " ".join(result.failures))
        self.assertEqual(server.mailboxes["INBOX"], {})


if __name__ == "__main__":
    unittest.main()
