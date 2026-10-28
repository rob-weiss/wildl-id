# Web Manual Labelling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the existing wildlife label reviewer to the local web container using the owner's Mac dataset without copying or exposing that data.

**Architecture:** Keep `web/` as the isolated Python environment and run the existing NiceGUI reviewer inside its current FastAPI app. Build from a strict repository-root allowlist; bind-mount the default CSV, annotations, originals, and models with the approved permissions.

**Tech Stack:** Python 3.13, NiceGUI, FastAPI, Uvicorn, Docker Compose, `uv`, unittest, NiceGUI `user_simulation`.

**Spec:** `docs/superpowers/specs/2026-10-28-web-manual-labelling-design.md`

## Global Constraints

- Host-publish only `127.0.0.1:8080`; NiceGUI listens on container `0.0.0.0:8080`.
- Keep web dependencies and `web/uv.lock` separate from the ML-heavy root environment.
- Never install PyTorch, torchvision, OpenCV, EasyOCR, or model runtimes in the web image.
- Mount only the specified CSV, annotated images, originals, and models; do not copy them into the image.
- CSV directory is writable; nested annotated images, original images, and models are read-only.
- Run with host numeric UID/GID, read-only container root, `/tmp` as writable scratch, all capabilities dropped, and no-new-privileges.
- Preserve the existing CSV backup, atomic save, external-change rejection, label undo, box undo, and reviewer modes.
- Hide `Remove empty`; add no uploads, dataset picker, login, LAN/internet ingress, or AI integration.
- Automated tests use disposable fixtures, never the owner's real labels or images.

## Review Focus

- Missing, unreadable, malformed, or schema-invalid CSV; render a useful error page without crashing the service. Test each invalid-dataset case in Task 2.
- CSV changed by another process after page load; preserve rejection and existing label-review regression tests in Task 2.
- Nested annotated-image mount accidentally remains writable through the parent CSV mount; inspect Docker mount `RW` flags and prove a disposable write fails in Task 3.
- Host UID/GID absent or mismatched; Compose must fail clearly when unset, and a disposable container must save a label and backup as the supplied UID/GID in Task 3.
- Build allowlist accidentally contains host data, models, secrets, or the root ML project; inspect the resolved context and image in Task 1.

---

### Task 1: Add Reviewer Dependencies And Strict Build Context

**Files:**
- Modify: `web/pyproject.toml`
- Modify: `web/uv.lock`
- Modify: `web/Dockerfile`
- Delete: `web/.dockerignore`
- Create: `.dockerignore`
- Modify: `compose.yaml`

**Interfaces:**
- Produces a root-context image build that still installs exclusively from `web/pyproject.toml` and `web/uv.lock`.
- Compose uses build context `.` and Dockerfile `web/Dockerfile`; host data is not yet mounted in this task.
- The root `.dockerignore` allows only web metadata/entrypoint, the named reviewer Python modules, `review_zoom.js`, and Panzoom JS/licence assets from the spec.

- [ ] **Step 1: Add isolated runtime dependencies** in `web/pyproject.toml`: `nicegui`, `matplotlib`, `pillow`, `piexif`, and `tqdm`, keeping FastAPI and Uvicorn. Set the test extra to `httpx>=0.28,<1.0` and `httpx2>=2.0,<3.0`; NiceGUI's `user_simulation` imports `httpx`, while Starlette prefers `httpx2`.
- [ ] **Step 2: Regenerate and verify the web lockfile** with `uv lock --project web`, then `uv lock --project web --check`. Expected: the isolated lock resolves without changing the root `uv.lock`.
- [ ] **Step 3: Replace `web/.dockerignore` with a root `.dockerignore` allowlist**. Unignore each parent directory before its allowed children. Include exactly the files listed in the spec; ensure `images/`, `labels/`, `models/`, `data/`, `logs/`, tests, `.git`, secrets, and virtual environments remain excluded.
- [ ] **Step 4: Update `web/Dockerfile`** to use repository-root paths in `COPY`, install only the locked web project, include the listed reviewer modules/assets, and set `MPLCONFIGDIR=/tmp/matplotlib`. Keep the official Python 3.13 slim runtime and unprivileged user.
- [ ] **Step 5: Point Compose at the root build context** and run `docker compose config --quiet` and `docker compose build`. Expected: valid configuration and a successful native-architecture image build containing no host dataset.
- [ ] **Step 6: Commit** the dependency and build-context changes.

### Task 2: Mount The Existing Reviewer In The Web App

**Files:**
- Modify: `web/app.py`
- Modify: `web/Dockerfile`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- Produces a FastAPI `app` with `GET /health` returning `{"status": "ok"}`, `build_review_page(csv_path: Path = DEFAULT_CSV) -> None`, and an explicit NiceGUI `@ui.page("/")` route mounted with `ui.run_with(app, mount_path="/")`.
- The root builder creates `ReviewApp`, calls `build()`, then hides `remove_empty_button`; it catches `OSError`, `ValueError`, and `csv.Error` from dataset loading and renders a clear setup error.
- Uvicorn serves `app:app` on container address `0.0.0.0:8080`, with one worker and reload disabled.

- [ ] **Step 1: Write tests** using `unittest.IsolatedAsyncioTestCase`, `httpx.AsyncClient`, and NiceGUI's `User` interaction helper under the app lifespan. Name cases `test_review_page_shows_labels_and_hides_remove_empty`, `test_page_shows_missing_csv_error`, `test_page_shows_invalid_csv_schema_error`, `test_page_shows_unreadable_csv_error`, `test_health_returns_ok_json`, and `test_unknown_route_returns_404`. Use a directory at the CSV path to exercise an `IsADirectoryError` without relying on OS permission bits.
- [ ] **Step 2: Run the focused web tests** with `uv run --project web --locked --extra test --python 3.13 python -m unittest discover -s tests -p test_web_app.py`. Expected: tests fail because the starter page does not build the reviewer.
- [ ] **Step 3: Implement `build_review_page(csv_path: Path = DEFAULT_CSV) -> None`** in `web/app.py`, using the existing `ReviewApp` and leaving its save, undo, box, and mode logic unchanged. Register `/health` on the outer FastAPI app.
- [ ] **Step 4: Register the explicit `/` page** with `@ui.page("/")` and mount NiceGUI using `ui.run_with(app, mount_path="/")`; retain Uvicorn serving `app:app` on `0.0.0.0:8080` with one worker and reload disabled. Keep unknown paths as HTTP 404.
- [ ] **Step 5: Run the focused web tests** and existing reviewer tests. Expected: temporary valid data displays; invalid data shows a useful error; health is 200; unknown paths are 404; reviewer tests retain existing behavior.
- [ ] **Step 6: Commit** the web integration and route tests.

### Task 3: Add Safe Host Mounts And Verify Container Writes

**Files:**
- Modify: `compose.yaml`

**Interfaces:**
- Consumes the working root-context image and reviewer defaults from Tasks 1 and 2.
- Requires `HOST_UID` and `HOST_GID` at runtime, sets Compose `user` to those numeric IDs, and mounts only the four spec paths.
- Uses `./labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified:/app/labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified:rw`, then overlays its `images/` child as `ro`; mounts `./images:/app/images:ro` and `./models:/app/models:ro`.

- [ ] **Step 1: Add required UID/GID interpolation** to Compose with actionable required-variable messages; do not recursively change host ownership or permissions.
- [ ] **Step 2: Add only the four approved bind mounts**, with the annotated image child mounted read-only after its writable parent. Keep port binding, read-only root, `/tmp`, health check, dropped capabilities, and bounded logs.
- [ ] **Step 3: Validate Compose** with `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose config --quiet`. Also verify configuration fails with a clear message when either variable is missing.
- [ ] **Step 4: Build and start the container** with `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build -d`. Verify local `/`, `/health`, healthy status, host binding exactly `127.0.0.1:8080`, runtime UID/GID, read-only root, and mount `RW` flags using `docker inspect`.
- [ ] **Step 5: Verify persistence in a disposable container**: create a temporary minimal CSV fixture, mount it at the default CSV path with its temporary `images/` child read-only, run `LabelReview.choose("fox")` under the supplied numeric UID/GID, then assert the CSV contains the label and `.before-review.bak` contains original bytes. Confirm writing an annotated image fails. Remove only the temporary fixture after checks.
- [ ] **Step 6: Run the full repository suite** and commit Compose mount changes after verification.

### Task 4: Document The Dataset Workflow And CI

**Files:**
- Modify: `web/README.md`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Docs use the fixed Mac dataset mounts and explicitly require host UID/GID in the start command.
- CI's Python 3.13 web job installs the isolated test extra, checks `web/uv.lock`, and runs web tests; existing Python 3.10/3.14 root matrix stays intact.

- [ ] **Step 1: Update `web/README.md`** with the default dataset paths, required `HOST_UID=$(id -u) HOST_GID=$(id -g)` start command, local browser URL, health check, logs, update, and stop commands. State CSV/backup writes, read-only images/models, no external access, and Pi browser locality.
- [ ] **Step 2: Update the Python 3.13 CI job** to install any direct test-only dependencies, verify the web lock, and run `tests/test_web_app.py` without installing the root ML environment.
- [ ] **Step 3: Run final checks**: web lock check; isolated Python 3.13 web tests; root lock check; compile check; complete repository suite; workflow YAML parse; Compose validation; image build and disposable mount verification from Task 3.
- [ ] **Step 4: Review the final diff** for data leakage in the build context, broad mounts, non-loopback publishing, or unintended deletion controls. Commit docs and CI together.

## Stop Conditions

- Stop after local native-architecture container verification; do not enable LAN or internet access.
- Never test writes against the owner's real CSV or images; use disposable fixtures only.
- If Docker is unavailable, report application-test results separately from unverified container permissions and health.
- Pi hardware and architecture remain unverified until tested on the Pi.
