# Email Image Downloader Design

## Goal

Replace the former website-based image downloader with an email-based downloader
for images sent by `info@supersim.camera` to `wildlife@robinweiss.de`.

## User Workflow

- Add a downloader action to the interactive menu and restore the `--download`
  command-line option, backed by the email downloader.
- Download attachments into `data/<location>/`, then run the existing organiser
  when the run saves new images. The organiser remains the only implementation
  of EXIF sorting, numbering, and CSV updates; final images go to
  `images/<year>/<location>/`.
- Keep `--all` limited to organising, labelling, and visualising. Downloading
  requires network access and mail credentials, so it remains an explicit,
  independent action.
- Parse location from subjects of the form `Kamera: Jochen <number> <location>`.
- Add a macOS LaunchAgent that checks at login and every 60 seconds.

## Mail Processing

- Use Python's standard-library IMAP client; do not add a mail dependency.
- Scan every message in `INBOX` and the server's trash/bin mailbox, including
  messages already marked read. Discover the mailbox with the IMAP `\\Trash`
  special-use flag when available, with common mailbox names as fallbacks and
  configuration for providers with a different name. The first run scans all
  messages; later runs use per-mailbox UID checkpoints and reset them if
  `UIDVALIDITY` changes.
- Download only image attachments from messages whose sender, recipient, and
  subject match the configured criteria. Check recipient headers including
  `To`, `Cc`, and delivery/original-recipient headers so alias delivery works.
- After every image attachment is saved or confirmed as a duplicate, mark the
  message read and move it to the server's trash/bin mailbox. Leave messages
  with no matching image attachments unchanged. Keep messages with any download
  or move failure available for retry. Use UID-scoped move/delete operations;
  never expunge unrelated messages.
- Prevent duplicate files by comparing SHA-256 content hashes within each
  location, both against existing files in `data/` and organized images, and
  during the current run. Keep distinct images with colliding attachment
  filenames by adding a stable hash suffix.
- Validate and sanitize location and attachment names before writing files.
- Run the existing organiser when new files were saved or matching attachments
  are already waiting in `data/` from an interrupted run. This intentionally
  preserves its current cleanup behavior: it deletes videos and corrupt images
  under `data/`.

## Configuration and Scheduling

- Store IMAP host, username, and optional mailbox overrides in a per-user config
  file outside the repository. Store the password in macOS Keychain and UID
  checkpoints in per-user application data; never put credentials in the
  repository, LaunchAgent plist, or logs.
- Keep the LaunchAgent template and installer in the project repository under
  `macos/launchd/`. The installer generates the machine-specific plist in
  `~/Library/LaunchAgents`; schedule source and setup logic remain versioned.
- Run checks when the agent loads and at 60-second intervals. macOS cannot run
  the job while asleep; launchd runs a missed interval after the Mac wakes.
- Keep scheduling optional: `--download` remains usable manually.

## Tests and Documentation

- Add focused tests with a mocked IMAP server for mailbox selection, read and
  unread messages, header and subject filtering, attachment extraction,
  duplicate handling, UID checkpoint recovery, and read/move behavior after
  successful downloads and on failure.
- Update the main CLI tests for the menu action, `--download`, organizer
  invocation after new downloads or pending staged files, and failure status.
  Verify `--all` remains independent of mail access.
- Document manual setup, Keychain configuration, launchd installation, and the
  sleep limitation in the README or project guide.

## Non-Goals

- Do not retain or restore the website downloader.
- Do not make email access a prerequisite for the existing full analysis
  pipeline.
- Do not attempt to run the Mac while it is asleep. Use an always-on server or
  device if checks must happen during sleep.

## Acceptance Criteria

- The old website downloader is replaced by the email downloader at the CLI and
  interactive menu entry points.
- Matching image attachments from Inbox and Trash/Bin are saved under
  `data/<location>/`, regardless of read status or the numeric camera
  identifier in the subject.
- Repeated messages or attachments do not create duplicate files after
  organizing moves images out of `data/`.
- Successfully handled messages are marked read and moved to Trash/Bin. Failed
  downloads remain available for retry; unrelated messages are never expunged.
- New downloads or pending staged images trigger the existing organiser; runs
  without matching mail do not.
- Credentials remain outside tracked files. LaunchAgent source is versioned in
  the repository.
- The LaunchAgent checks at startup and every minute while macOS is awake, then
  catches up after wake.
- Focused tests pass, and documentation describes configuration and scheduling.
