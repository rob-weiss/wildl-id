# Web Manual Labelling Design

## Purpose

Make the existing wildlife manual reviewer usable as wildl-id's first web feature.
Initially, the user reviews the existing default dataset on the Mac through the
local Docker app. The existing CSV remains the source of truth; labels and
backups stay on the host. The same deployment can later run against a dataset
placed on the Pi.

This feature amends the container starter's original no-data-access boundary:
only the explicit default dataset mounts described below are introduced. No
uploads, arbitrary path selection, public access, or AI processing are added.

## Approved Scope

- Reuse the existing NiceGUI reviewer and its label, custom-label, undo,
  bounding-box, zoom, and review-filter behaviour.
- Serve the reviewer at `/` and retain `/health`.
- Use the repository's existing default CSV and image locations.
- Allow CSV and sibling backup writes; mount annotated and original images and
  models read-only.
- Hide the destructive "Remove empty" control in the web app.
- Keep internet, LAN, and remote access disabled.

Dataset upload, a dataset picker, multi-user sessions, application login,
"Remove empty", model inference, and Hailo/AI HAT+ 2 support remain out of scope.

## Architecture

Replace the standalone FastAPI page with the existing NiceGUI application and
its FastAPI app. Register `/health` on that app. Pass a root-page builder to
`ui.run(root=...)` so the reviewer UI is created through NiceGUI's supported
startup path; run Uvicorn on container address `0.0.0.0:8080`, with reload and
browser auto-open disabled. Host publishing remains `127.0.0.1:8080` only.

The root-page builder creates `ReviewApp` with the default CSV, builds the
existing interface, then hides its "Remove empty" button. Do not copy the
reviewer implementation into `web/`; keep its existing ownership and tests.

The image now needs the repository root as its Docker build context so it can
include selected files from `src/labelling/`. Replace the web-only ignore file
with a root `.dockerignore` allowlist. Include `web/Dockerfile`, web project
metadata, lockfile, and entrypoint; `src/labelling/review_app.py`,
`review_labels.py`, `remove_empty_images.py`, `organize_images.py`,
`pixel_hash.py`, and `review_zoom.js`; plus the Panzoom vendor JS and licence.
Exclude `.git`, tests, images, labels, models, data, logs, virtual environments,
secrets, and every other repository file. Continue to install from
`web/pyproject.toml` and `web/uv.lock`, never the ML-heavy root project.

Add only reviewer runtime dependencies to the isolated web project: NiceGUI,
Matplotlib, Pillow, piexif, and tqdm, alongside FastAPI and Uvicorn. Do not add
PyTorch, torchvision, OpenCV, EasyOCR, or model runtimes. Keep the existing
`httpx2` test extra.

## Host Data And Permissions

Use these default paths, which the current reviewer already expects:

- CSV directory: `labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/`.
- Annotated images: the `images/` child of that CSV directory.
- Original images: repository `images/`.
- Latest validation filter data: repository `models/`.

Mount the CSV directory read-write because the reviewer creates its one-time
backup and atomically replaces the CSV in that directory. Mount its nested
annotated-images directory read-only to override the parent mount's write
permission. Mount original images and models read-only. Do not mount `data/`,
logs, credentials, the Docker socket, or the full repository as a runtime
volume. Do not copy host images, labels, models, or backups into the image.

Run the container under the host's numeric UID and GID supplied through Compose
variables. Document a start command that obtains both with `id`; this lets the
non-root process update the host CSV and create its backup on both macOS and
Linux. Keep the container root filesystem read-only, with `/tmp` as its only
other writable area. Do not change host permissions recursively.

## Behaviour And Failure Handling

The existing reviewer remains responsible for CSV parsing, image lookup,
filters, edits, and persistence. Preserve its save safeguards: create the
`.before-review.bak` backup before the first save, write via a temporary file
and atomic replacement, and reject a save if another process changed the CSV
since the reviewer loaded it. Show the existing user-facing save error; do not
silently overwrite concurrent changes. Label undo remains session-local, while
each accepted edit is saved immediately.

If the default CSV is missing, unreadable, or has invalid required columns,
show a clear setup error on `/` instead of an unhandled traceback. `/health`
continues to report server liveness, not dataset readiness. Missing or unreadable
images use the reviewer's existing error state. Startup errors remain visible in
container logs.

The owner is the only intended user. Loopback-only publishing is not
per-user authentication: other local processes or accounts may reach the app.
No external clients can reach it through the configured host port.

## Verification And Acceptance

1. Tests use temporary CSVs and images, never the owner's real dataset.
2. Test that `/` builds the reviewer, hides "Remove empty", and reports missing
   or invalid default data clearly; `/health` returns HTTP 200; unknown routes
   return HTTP 404.
3. Run existing reviewer tests and the full repository suite without changing
   their expected CSV, undo, box, and conflict behaviour.
4. Check the web lockfile and run web tests in isolated Python 3.13 CI without
   installing the root ML dependencies; preserve the existing CI matrix.
5. Validate Compose and build the image for the host's native architecture.
6. Run the container and verify local HTTP routes, healthy status, non-root UID,
   read-only root, read-only image/model mounts, writable CSV directory, and
   host publishing exactly on `127.0.0.1:8080`.
7. Verify the CSV backup and label edit using a disposable host fixture mounted
   into a test container; do not alter the real dataset during automated tests.
8. Confirm the Docker build context contains only the explicit allowlist and no
   host dataset, models, secrets, or ML dependencies.

Do not claim Pi hardware or architecture validation without testing on the Pi.
Do not enable LAN or internet access as part of this feature.

## Future Compatibility

The same mount contract can point at the Pi's local dataset when the container
moves there. Supporting alternate datasets, browser uploads, invited users, or
public access requires a separate design and security review. No database or
external storage service is introduced now.
