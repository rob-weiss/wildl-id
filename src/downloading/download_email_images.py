"""Download matching camera images from IMAP mailboxes."""

import csv
import hashlib
import imaplib
import json
import mimetypes
import os
import re
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from email import message_from_bytes, policy
from email.utils import getaddresses
from imaplib import IMAP4
from pathlib import Path

from PIL import Image

try:
    from ..labelling.pixel_hash import HASH_VERSION, pixel_digest_bytes
except ImportError:
    from labelling.pixel_hash import HASH_VERSION, pixel_digest_bytes

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
IMAGES_DIR = REPO_DIR / "images"
CONFIG_PATH = Path.home() / ".config/wildl-id/mail-downloader.json"
STATE_PATH = (
    Path.home() / "Library/Application Support/wildl-id/mail-downloader/state.json"
)
KEYCHAIN_SERVICE = "wildl-id-mail-downloader"
SENDER_ADDRESS = "info@supersim.camera"
RECIPIENT_ADDRESS = "wildlife@robinweiss.de"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAP_TIMEOUT_SECONDS = 30
DOWNLOAD_HASH_VERSION = 2
SUBJECT_PATTERN = re.compile(r"^Kamera:\s*Jochen\s+\d+\s+(.+?)\s*$", re.IGNORECASE)
RECIPIENT_HEADERS = (
    "To",
    "Cc",
    "Delivered-To",
    "X-Original-To",
    "Envelope-To",
    "X-Envelope-To",
    "X-Rcpt-To",
)


@dataclass(frozen=True)
class DownloadResult:
    saved_images: int
    duplicates: int
    pending_organization: bool
    failures: tuple[str, ...]


class MessageProcessingError(RuntimeError):
    def __init__(self, error, saved_images, duplicates, pending_organization):
        super().__init__(str(error))
        self.error = error
        self.saved_images = saved_images
        self.duplicates = duplicates
        self.pending_organization = pending_organization


def _read_keychain_password(service, account):
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-w",
                "-s",
                service,
                "-a",
                account,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.rstrip("\r\n")


def _read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("mail configuration must be a JSON object")
    host = config.get("host")
    username = config.get("username")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("mail configuration requires a non-empty host")
    if not isinstance(username, str) or not username.strip():
        raise ValueError("mail configuration requires a non-empty username")
    port = config.get("port", 993)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("mail configuration port must be between 1 and 65535")
    return {**config, "host": host.strip(), "username": username.strip(), "port": port}


def _load_state(path):
    if not path.exists():
        return {"mailboxes": {}, "files": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("mail downloader state must be a JSON object")
    state.setdefault("mailboxes", {})
    state.setdefault("files", {})
    if not isinstance(state["mailboxes"], dict) or not isinstance(state["files"], dict):
        raise ValueError("mail downloader state has an invalid structure")
    return state


def _write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _status_ok(status):
    if isinstance(status, bytes):
        status = status.decode("ascii", errors="replace")
    return str(status).upper() == "OK"


def _decode_mailbox(raw):
    line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    match = re.match(r'^\(([^)]*)\)\s+(?:NIL|"(?:\\.|[^"])*")\s+(.+)$', line)
    if not match:
        return set(), line.strip().strip('"')
    flags = {flag.casefold() for flag in match.group(1).split()}
    name = match.group(2).strip()
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return flags, name


def _mailboxes(connection, config):
    status, data = connection.list()
    if not _status_ok(status):
        raise RuntimeError("could not list IMAP mailboxes")
    listed = [_decode_mailbox(item) for item in data or () if item]
    names = [name for _, name in listed]
    inbox = config.get("inbox_mailbox", "INBOX")
    if inbox not in names:
        raise RuntimeError("configured Inbox mailbox was not found")

    override = config.get("trash_mailbox")
    if override:
        trash = override if override in names else None
    else:
        trash = next((name for flags, name in listed if "\\trash" in flags), None)
        if trash is None:
            common_names = {
                "trash",
                "bin",
                "deleted items",
                "deleted messages",
                "inbox.trash",
                "inbox/bin",
            }
            trash = next(
                (name for name in names if name.casefold() in common_names), None
            )
    if trash is None:
        raise RuntimeError("Trash/Bin mailbox was not found; configure trash_mailbox")
    return inbox, trash


def _uidvalidity(connection):
    response = connection.response("UIDVALIDITY")
    if not response or not response[1]:
        raise RuntimeError("IMAP server did not provide UIDVALIDITY")
    value = response[1][0]
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    return str(value)


def _capabilities(connection):
    status, data = connection.capability()
    if not _status_ok(status):
        return set()
    raw = b" ".join(item for item in data or () if isinstance(item, bytes))
    return {item.decode("ascii", errors="ignore").upper() for item in raw.split()}


def _mailbox_snapshot(connection, mailbox, state):
    status, _ = connection.select(_quote_mailbox(mailbox), readonly=False)
    if not _status_ok(status):
        raise RuntimeError(f"could not select IMAP mailbox {mailbox!r}")
    validity = _uidvalidity(connection)
    mailbox_state = state["mailboxes"].get(mailbox, {})
    if mailbox_state.get("uidvalidity") != validity:
        last_uid = 0
    else:
        last_uid = int(mailbox_state.get("last_uid", 0))
    status, data = connection.uid("SEARCH", None, "UID", f"{last_uid + 1}:*")
    if not _status_ok(status):
        raise RuntimeError(f"could not search IMAP mailbox {mailbox!r}")
    uids = [uid for uid in _uid_values(data) if int(uid) > last_uid]
    return validity, last_uid, sorted(set(uids), key=int)


def _header_addresses(message, names):
    values = []
    for name in names:
        values.extend(str(value) for value in message.get_all(name, ()))
    return {address.casefold() for _, address in getaddresses(values) if address}


def _message_location(message):
    subject = str(message.get("Subject", ""))
    match = SUBJECT_PATTERN.fullmatch(subject.strip())
    if not match:
        return None
    location = unicodedata.normalize("NFC", match.group(1).strip())
    if (
        not location
        or location in {".", ".."}
        or "/" in location
        or "\\" in location
        or any(ord(char) < 32 for char in location)
    ):
        return None
    return location


def _matches(message):
    sender_addresses = _header_addresses(message, ("From",))
    recipient_addresses = _header_addresses(message, RECIPIENT_HEADERS)
    location = _message_location(message)
    return (
        SENDER_ADDRESS.casefold() in sender_addresses
        and RECIPIENT_ADDRESS.casefold() in recipient_addresses
        and location is not None,
        location,
    )


def _image_attachments(message):
    attachments = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_maintype().casefold() != "image":
            continue
        filename = part.get_filename()
        if not filename:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            raise ValueError("image attachment had no content")
        attachments.append((filename, payload, part.get_content_type().casefold()))
    return attachments


def _safe_filename(filename, content_type):
    filename = unicodedata.normalize("NFC", str(filename)).strip()
    if not filename or filename in {".", ".."}:
        raise ValueError("image attachment has an unsafe filename")
    if "/" in filename or "\\" in filename or any(ord(char) < 32 for char in filename):
        raise ValueError("image attachment has an unsafe filename")
    suffix = Path(filename).suffix.lower()
    if not suffix:
        suffix = mimetypes.guess_extension(content_type, strict=False) or ""
        filename += suffix
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image format: {suffix or content_type}")
    return filename, suffix


def _content_digest(content):
    try:
        return pixel_digest_bytes(content).hex()
    except (OSError, ValueError, ImportError, Image.DecompressionBombError):
        return hashlib.sha256(content).hexdigest()


def _sha256(path):
    return _content_digest(path.read_bytes())


def _location_for_existing(path, root, kind):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if kind == "data" and len(parts) >= 2:
        return unicodedata.normalize("NFC", parts[0])
    if kind == "images" and len(parts) >= 3:
        return unicodedata.normalize("NFC", parts[1])
    return None


def _shared_pixel_hash_cache():
    cache_path = REPO_DIR / "logs" / "image_hashes.csv"
    entries = {}
    try:
        with cache_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    relative = Path(row["image_path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        continue
                    path = REPO_DIR / relative
                    resolved = path.resolve()
                    if not resolved.is_relative_to(REPO_DIR.resolve()):
                        continue
                    stat = path.stat()
                    digest = bytes.fromhex(row["sha256"])
                    if (
                        row["hash_version"] != HASH_VERSION
                        or len(digest) != 32
                        or int(row["size"]) != stat.st_size
                        or int(row["mtime_ns"]) != stat.st_mtime_ns
                    ):
                        continue
                    entries[str(resolved)] = {
                        "hash_version": DOWNLOAD_HASH_VERSION,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": digest.hex(),
                    }
                except (KeyError, OSError, TypeError, ValueError):
                    continue
    except (OSError, UnicodeError, csv.Error):
        return {}
    return entries


def _image_index(state):
    old_cache = state["files"]
    old_cache.update(_shared_pixel_hash_cache())
    new_cache = {}
    by_location = {}
    by_hash = {}
    roots = ((DATA_DIR, "data"), (IMAGES_DIR, "images"))
    for root, kind in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.rglob("*"):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue
            location = _location_for_existing(path, root, kind)
            if not location:
                continue
            key = str(path.resolve())
            stat = path.stat()
            cached = old_cache.get(key, {})
            if (
                cached.get("hash_version") == DOWNLOAD_HASH_VERSION
                and cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and isinstance(cached.get("sha256"), str)
            ):
                digest = cached["sha256"]
            else:
                digest = _sha256(path)
            new_cache[key] = {
                "hash_version": DOWNLOAD_HASH_VERSION,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
                "location": location,
                "kind": kind,
            }
            by_location.setdefault(location, {}).setdefault(digest, set()).add(key)
            by_hash[key] = (location, digest, kind)
    state["files"] = new_cache
    return by_location, by_hash


def _write_attachment(
    location, filename, content, digest, existing, by_hash, file_cache
):
    target_dir = DATA_DIR / location
    if DATA_DIR.is_symlink() or target_dir.is_symlink():
        raise ValueError("download destination must not use a symlink")
    target_dir.mkdir(parents=True, exist_ok=True)
    data_root = DATA_DIR.resolve()
    if not target_dir.resolve().is_relative_to(data_root):
        raise ValueError("download destination escapes data directory")
    target = target_dir / filename
    if target.is_symlink():
        raise ValueError("download target must not be a symlink")
    if target.exists():
        suffix = target.suffix
        target = target.with_name(f"{target.stem}_{digest[:12]}{suffix}")
    if target.is_symlink():
        raise ValueError("download target must not be a symlink")
    if target.exists():
        if _sha256(target) == digest:
            return target, False
        target = target.with_name(f"{target.stem}_{digest[12:24]}{target.suffix}")
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    key = str(target.resolve())
    stat = target.stat()
    file_cache[key] = {
        "hash_version": DOWNLOAD_HASH_VERSION,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
        "location": location,
        "kind": "data",
    }
    by_hash[key] = (location, digest, "data")
    existing.setdefault(location, {}).setdefault(digest, set()).add(key)
    return target, True


def _uid_values(data):
    return [
        uid.decode("ascii")
        for row in data or ()
        if isinstance(row, bytes)
        for uid in row.split()
    ]


def _fetch_message(connection, uid):
    status, data = connection.uid("FETCH", uid, "(UID FLAGS BODY.PEEK[])")
    if not _status_ok(status):
        raise RuntimeError("could not fetch message")
    raw = next((item[1] for item in data or () if isinstance(item, tuple)), None)
    if not isinstance(raw, bytes):
        raise RuntimeError("IMAP server returned no message body")
    return message_from_bytes(raw, policy=policy.default)


def _quote_mailbox(name):
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _mark_seen(connection, uid):
    status, _ = connection.uid("STORE", uid, "+FLAGS.SILENT", r"(\Seen)")
    if not _status_ok(status):
        raise RuntimeError("could not mark message as read")


def _move_message(connection, capabilities, uid, source, destination):
    if source == destination:
        return
    quoted_destination = _quote_mailbox(destination)
    if "MOVE" in capabilities:
        status, _ = connection.uid("MOVE", uid, quoted_destination)
        if _status_ok(status):
            return
    if "UIDPLUS" not in capabilities:
        raise RuntimeError("server lacks safe UID MOVE or UID EXPUNGE support")
    status, _ = connection.uid("COPY", uid, quoted_destination)
    if not _status_ok(status):
        raise RuntimeError("could not copy message to Trash/Bin")
    status, _ = connection.uid("STORE", uid, "+FLAGS.SILENT", r"(\Deleted)")
    if not _status_ok(status):
        raise RuntimeError("could not mark source message for deletion")
    status, _ = connection.uid("EXPUNGE", uid)
    if _status_ok(status):
        return
    connection.uid("STORE", uid, "-FLAGS.SILENT", r"(\Deleted)")
    raise RuntimeError("server rejected UID EXPUNGE; source message was preserved")


def _process_message(
    connection,
    capabilities,
    uid,
    source,
    trash,
    existing,
    by_hash,
    file_cache,
):
    message = _fetch_message(connection, uid)
    matches, location = _matches(message)
    if not matches:
        return 0, 0, False
    attachments = _image_attachments(message)
    if not attachments:
        return 0, 0, False

    saved = duplicates = 0
    pending_organization = False
    try:
        for filename, content, content_type in attachments:
            filename, _ = _safe_filename(filename, content_type)
            digest = _content_digest(content)
            found = existing.get(location, {}).get(digest, set())
            if found:
                duplicates += 1
                pending_organization |= any(
                    by_hash[path][2] == "data" for path in found if path in by_hash
                )
                continue
            target, was_saved = _write_attachment(
                location, filename, content, digest, existing, by_hash, file_cache
            )
            if was_saved:
                saved += 1
                pending_organization = True

        _mark_seen(connection, uid)
        _move_message(connection, capabilities, uid, source, trash)
    except Exception as error:
        raise MessageProcessingError(
            error, saved, duplicates, pending_organization
        ) from error
    return saved, duplicates, pending_organization


def download_email_images(
    config_path: Path | None = None,
    state_path: Path | None = None,
    *,
    imap_factory: Callable[[str, int], IMAP4] | None = None,
    keychain_reader: Callable[[str, str], str] | None = None,
) -> DownloadResult:
    config_path = Path(config_path or CONFIG_PATH).expanduser()
    state_path = Path(state_path or STATE_PATH).expanduser()
    try:
        config = _read_config(config_path)
        state = _load_state(state_path)
        password_reader = keychain_reader or _read_keychain_password
        password = password_reader(KEYCHAIN_SERVICE, config["username"])
        if not password:
            raise RuntimeError("mail password is missing from macOS Keychain")
        connection: IMAP4
        if imap_factory is None:
            connection = imaplib.IMAP4_SSL(
                config["host"], config["port"], timeout=IMAP_TIMEOUT_SECONDS
            )
        else:
            connection = imap_factory(config["host"], config["port"])
        try:
            status, _ = connection.login(config["username"], password)
        except imaplib.IMAP4.error:
            raise RuntimeError("IMAP authentication failed") from None
        if not _status_ok(status):
            raise RuntimeError("IMAP authentication failed")
        inbox, trash = _mailboxes(connection, config)
        capabilities = _capabilities(connection)
        by_location, by_hash = _image_index(state)

        snapshots = {}
        for mailbox in dict.fromkeys((inbox, trash)):
            snapshots[mailbox] = _mailbox_snapshot(connection, mailbox, state)

        saved = duplicates = 0
        pending_organization = False
        failures = []
        for mailbox, (validity, previous_uid, uids) in snapshots.items():
            select_status, _ = connection.select(
                _quote_mailbox(mailbox), readonly=False
            )
            if not _status_ok(select_status) or _uidvalidity(connection) != validity:
                failures.append(f"{mailbox}: mailbox changed during download scan")
                continue
            first_failure = None
            for uid in uids:
                try:
                    message_saved, message_duplicates, needs_organization = (
                        _process_message(
                            connection,
                            capabilities,
                            uid,
                            mailbox,
                            trash,
                            by_location,
                            by_hash,
                            state["files"],
                        )
                    )
                    saved += message_saved
                    duplicates += message_duplicates
                    pending_organization |= needs_organization
                except Exception as error:
                    if isinstance(error, MessageProcessingError):
                        saved += error.saved_images
                        duplicates += error.duplicates
                        pending_organization |= error.pending_organization
                        reported_error = error.error
                    else:
                        reported_error = error
                    failures.append(
                        f"{mailbox} UID {uid}: "
                        f"{type(reported_error).__name__}: {reported_error}"
                    )
                    if first_failure is None:
                        first_failure = int(uid)
            if first_failure is None:
                last_uid = max((int(uid) for uid in uids), default=previous_uid)
            else:
                last_uid = min(previous_uid, first_failure - 1)
            state["mailboxes"][mailbox] = {
                "uidvalidity": validity,
                "last_uid": last_uid,
            }
            _write_state(state_path, state)
        return DownloadResult(saved, duplicates, pending_organization, tuple(failures))
    except Exception as error:
        return DownloadResult(0, 0, False, (f"{type(error).__name__}: {error}",))
    finally:
        if "connection" in locals():
            try:
                connection.logout()
            except Exception:
                pass
