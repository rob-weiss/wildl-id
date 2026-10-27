# Email Image Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the website downloader with scheduled IMAP image downloads that feed the existing organiser.

**Architecture:** A standard-library IMAP module scans Inbox and Trash/Bin, downloads matching attachments into `data/<location>/`, and tracks UID and image hashes outside the repository. The CLI runs the existing organiser when staged images need processing; a versioned LaunchAgent template and installer provide one-minute polling.

**Tech Stack:** Python 3.10+, `imaplib`, `email`, `hashlib`, `json`, `plistlib`, `subprocess`, `unittest`; no new dependency.

**Spec:** `designs/email-image-downloader-design.md`

## Global Constraints

- `--all` remains limited to organising, labelling, and visualising.
- Use Python's standard-library IMAP client; do not add a mail dependency.
- Never put credentials in the repository, LaunchAgent plist, or logs.
- The first run scans all messages; later runs use per-mailbox UID checkpoints and reset them if `UIDVALIDITY` changes.
- macOS cannot run the job while asleep; launchd runs a missed interval after the Mac wakes.
- Keep Python support at `>=3.10, <3.15`.
- Keep documentation and user-facing prose in British English.

## Review Focus

- Encoded subjects, variable numeric camera IDs, and unsafe or multi-word locations must match or reject safely; test subject decoding and location sanitisation in Task 1.
- Read mail, alias delivery headers, and old Trash/Bin messages must be considered on the first scan; test all mailbox and recipient paths in Task 1.
- Identical image bytes must not duplicate after the organiser renames and moves files; test data and organised-image deduplication in Task 1.
- Partial downloads or failed moves must remain retryable without expunging unrelated messages; test cursor, flags, and UID-scoped move fallback in Task 1.
- Missing Keychain entries and IMAP authentication failures must fail clearly without exposing passwords; test error handling in Task 1 and the CLI result in Task 2.

---

### Task 1: IMAP Download Core

**Files:**

- Create: `src/downloading/download_email_images.py`
- Test: `tests/test_download_email_images.py`

**Interfaces:**

- Produces `DownloadResult(saved_images: int, duplicates: int, pending_organization: bool, failures: tuple[str, ...])`.
- Produces `download_email_images(config_path: Path | None = None, state_path: Path | None = None, *, imap_factory: Callable[[str, int], IMAP4] | None = None, keychain_reader: Callable[[str, str], str] | None = None) -> DownloadResult`.
- Reads `~/.config/wildl-id/mail-downloader.json`; reads the password from Keychain service `wildl-id-mail-downloader` and account equal to the configured username.
- Stores UID cursors and a path/size/mtime/SHA-256 cache under `~/Library/Application Support/wildl-id/mail-downloader/state.json`.

- [ ] **Step 1: Write failing unit tests** for variable numeric subject IDs, Unicode and multi-word locations, sender/alias recipient matching, Inbox and `\\Trash` discovery, read and unread UIDs, image MIME extraction, filename collisions, and content deduplication across `data/` and `images/`. Add `src/` to `sys.path` in this test module, matching the CLI's import setup.
- [ ] **Step 2: Run the focused tests** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p 'test_download_email_images.py' -v`; confirm they fail before implementation.
- [ ] **Step 3: Implement the standard-library IMAP module** with injectable IMAP and Keychain readers, atomic file/state writes, path sanitisation, per-mailbox UID checkpoints, and cache invalidation when file metadata changes.
- [ ] **Step 4: Add failure tests** for UIDVALIDITY reset, interrupted download retry, failed move retry, safe UID-scoped COPY/EXPUNGE fallback, unrelated message preservation, and missing credentials without secret output.
- [ ] **Step 5: Run the focused tests** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p 'test_download_email_images.py' -v`; confirm all pass.
- [ ] **Step 6: Commit** the module and its tests as `feat: download camera images from email`.

### Task 2: CLI and Existing Organiser

**Files:**

- Modify: `wildl_id.py`
- Test: `tests/test_wildl_id.py`

**Interfaces:**

- Consumes `download_email_images()` and `DownloadResult` from Task 1.
- Adds a `download_images()` action that reports failures, calls `organize_images()` when `pending_organization` is true, and returns success only when download and organization succeed.
- Adds `--download` and menu choice 5; preserves choices 1-4 and moves Exit to choice 6.

- [ ] **Step 1: Write failing CLI tests** for `--download`, menu choice 5, organizer invocation for new or pending staged images, no organizer call without matching mail, and non-zero status on download or organizer failure.
- [ ] **Step 2: Run focused CLI tests** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p 'test_wildl_id.py' -v`; confirm the new cases fail.
- [ ] **Step 3: Wire the download action** into the menu and parser; keep it out of `run_all()` and avoid running the organiser twice when flags are combined.
- [ ] **Step 4: Run focused CLI tests** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p 'test_wildl_id.py' -v`; confirm all pass.
- [ ] **Step 5: Commit** CLI and tests as `feat: add email download command`.

### Task 3: macOS LaunchAgent and Setup Docs

**Files:**

- Create: `macos/launchd/com.robinweiss.wildl-id-mail-downloader.plist.template`
- Create: `macos/launchd/install.py`
- Modify: `README.md`
- Test: `tests/test_launchd_install.py`

**Interfaces:**

- The installer accepts `install` and `uninstall`, uses `plistlib` to render the checked-in template, and writes `~/Library/LaunchAgents/com.robinweiss.wildl-id-mail-downloader.plist`.
- The LaunchAgent runs the current project interpreter and `wildl_id.py --download`, with `RunAtLoad = true` and `StartInterval = 60`.

- [ ] **Step 1: Write failing installer tests** for template rendering, absolute project/interpreter paths, the 60-second interval, RunAtLoad, install/uninstall behavior, and absence of credentials.
- [ ] **Step 2: Run installer tests** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p 'test_launchd_install.py' -v`; confirm they fail.
- [ ] **Step 3: Implement the plist template and installer** using `plistlib`; keep generated paths and Keychain data outside the repository.
- [ ] **Step 4: Document setup** in README: IMAP config fields, Keychain item setup, install/uninstall commands, output folders, one-minute polling, wake behavior, and organizer cleanup of videos/corrupt images.
- [ ] **Step 5: Run the full suite** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -v`; confirm all pass.
- [ ] **Step 6: Commit** installer, template, tests, and docs as `feat: schedule email image downloads`.

## Runtime Files Not Version-Controlled

- `data/<location>/**` contains staged email images; `.gitignore` excludes `data/`.
- `images/**` contains downloaded and organised photos; `.gitignore` excludes `images/`.
- `~/.config/wildl-id/mail-downloader.json` contains IMAP host, username, and optional mailbox settings.
- The Keychain item `wildl-id-mail-downloader` contains the mail password; it is not a repository file.
- `~/Library/Application Support/wildl-id/mail-downloader/state.json` contains mailbox cursors and the image hash cache.
- `~/Library/LaunchAgents/com.robinweiss.wildl-id-mail-downloader.plist` is generated from the versioned template.
- `~/Library/Logs/wildl-id/` contains LaunchAgent output logs.

## Final Verification

- [ ] Run `uv lock --check`.
- [ ] Run `uv run --locked python -m compileall -q src wildl_id.py tests`.
- [ ] Run `PYTHONPATH=. uv run --locked python -m unittest discover -s tests`.
- [ ] Confirm `--all` remains independent of mail access and generated/runtime files remain outside Git.
