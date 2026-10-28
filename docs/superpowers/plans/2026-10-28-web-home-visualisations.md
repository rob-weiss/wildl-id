# Web Home And Visualisations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local home menu and an asynchronous visualization workflow that displays the existing analysis outputs without writing to the host repository.

**Architecture:** Keep `evaluate_labels.main()` as the analysis engine, adding optional CSV/output paths while retaining CLI defaults. A temporary-run manager executes it through NiceGUI's worker pool, preserves the last successful run, and serves one selected SVG at a time.

**Tech Stack:** Python 3.13, NiceGUI, FastAPI, Matplotlib, pandas, seaborn, Astral, Ephem, Docker Compose, unittest.

**Spec:** `docs/superpowers/specs/2026-10-28-web-home-visualisations-design.md`

## Global Constraints

- Keep host access bound to `127.0.0.1:8080`; do not enable LAN or internet access.
- Preserve the manual reviewer at `/review` and add a navigation-only home page at `/`.
- Visualisations run only after an explicit user action and use the mounted default CSV.
- Run visualization generation off the NiceGUI event loop; prevent concurrent runs.
- Write outputs only under `/tmp/wildl-id-visualisations/`; never modify host `docs/diagrams/`.
- Preserve the previous successful run if a new run fails; remove temporary failed output.
- Keep only one selected SVG loaded in the page; show `summary_statistics.txt`.
- Increase Compose `/tmp` from 16 MiB to 64 MiB; do not add a writable host mount.
- Keep web dependencies isolated; do not add ML runtimes or access to `data/`.
- CLI calls without explicit paths must retain current default CSV and `docs/diagrams/` behavior.

## Review Focus

- Missing, malformed, empty, or schema-invalid CSV; show a useful error and preserve previous charts. Test through generator and UI tasks.
- Visualizer returns `False` after partial output or raises mid-run; remove only failed output and retain prior successful run. Test in Task 3.
- Two rapid clicks start overlapping Matplotlib work; reject the second run and keep the page responsive. Test in Task 3.
- Host `docs/diagrams/` or the real CSV changes during visualization; test that success, failure, and rerun touch only disposable input/output paths. Test in Task 2.
- Temporary storage fills or cannot create a run directory; show an error, keep old results, and leave the UI usable. Test in Task 3.

---

### Task 1: Add Visualization Runtime To The Isolated Image

**Files:**
- Modify: `web/pyproject.toml`
- Modify: `web/uv.lock`
- Modify: `.dockerignore`
- Modify: `web/Dockerfile`

**Interfaces:**
- Produces an image containing `src/visualisation/evaluate_labels.py` and its direct runtime dependencies, while excluding `docs/diagrams/` and all host data.
- Continues to install the isolated web project only; root ML dependencies remain absent.

- [ ] **Step 1: Add direct dependencies** to `web/pyproject.toml`: `astral>=3.2`, `ephem>=4.2.1`, `numpy>=1.23.2`, `pandas>=2.3.3`, `pytz`, and `seaborn>=0.13.2`. Keep existing FastAPI, NiceGUI, Matplotlib, Pillow, and test extras.
- [ ] **Step 2: Regenerate and verify the lockfile** with `uv lock --project web` and `uv lock --project web --check`. Expected: dependencies resolve in `web/uv.lock`; root `uv.lock` remains unchanged.
- [ ] **Step 3: Add only `src/visualisation/` and `evaluate_labels.py`** to the root `.dockerignore` allowlist. Do not allow `docs/`, `docs/diagrams/`, or any additional host data.
- [ ] **Step 4: Copy the visualizer module** into `/app/src/visualisation/` in `web/Dockerfile`.
- [ ] **Step 5: Build and inspect** with `docker compose build` and `docker run --rm --entrypoint sh wildl-id-web -c 'find /app/src/visualisation -maxdepth 1 -type f -print'`. Expected: only the visualizer source is added; no diagrams or dataset files enter the image.
- [ ] **Step 6: Commit** the dependency and build-context changes.

### Task 2: Make The Existing Generator Path-Injectable

**Files:**
- Modify: `src/visualisation/evaluate_labels.py`
- Modify: `tests/test_evaluate_labels.py`

**Interfaces:**
- Changes `main() -> bool` to `main(csv_path: Path | None = None, output_dir: Path | None = None) -> bool`.
- With omitted arguments, retains current default CSV and `docs/diagrams/` paths. Supplied paths control all input/output for web runs and tests.

- [ ] **Step 1: Add tests** named `test_main_uses_custom_paths_and_manual_labels`, `test_missing_custom_csv_preserves_existing_output`, and `test_invalid_custom_csv_preserves_existing_output`. Use a temporary 28-day CSV with 4 rows per day, cycling `roe deer`, `wild boar`, `human`, and `horse`; set `manual_label="fox"` on horse rows. Include `location_id`, `timestamp`, `timestamp_exif`, `class`, `manual_label`, `lighting`, and varying `temperature_celsius` columns.
- [ ] **Step 2: Run the focused visualizer tests** with `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_evaluate_labels.py`. Expected: custom-path tests fail because `main` has no path parameters.
- [ ] **Step 3: Implement the optional path parameters** in `main`; resolve defaults exactly as today, validate the supplied CSV before clearing only the supplied output directory, and keep the `__main__` call unchanged.
- [ ] **Step 4: Assert successful outputs** in the custom-path test: it returns `True`, writes `summary_statistics.txt` and at least one SVG in the temporary output directory, reflects `fox` in species counts, and does not create or alter host `docs/diagrams/`.
- [ ] **Step 5: Run all visualizer tests**. Expected: custom success and failure cases pass; existing manual-label filters and default-output safety tests remain green.
- [ ] **Step 6: Commit** the generator path injection and tests.

### Task 3: Manage Temporary Visualization Runs

**Files:**
- Create: `web/visualisations.py`
- Create: `tests/test_web_visualisations.py`

**Interfaces:**
- Produces `VisualisationRunStore(output_root: Path = Path("/tmp/wildl-id-visualisations"))` with `async generate(csv_path: Path) -> Path | None`, `current_directory: Path | None`, and `is_running: bool`.
- The store calls `evaluate_labels.main(csv_path=..., output_dir=...)` through `nicegui.run.io_bound`; injected output roots and a patched generator support isolated tests.

- [ ] **Step 1: Write tests** named `test_success_replaces_previous_run`, `test_false_result_removes_partial_output_and_keeps_previous`, `test_exception_removes_failed_output_and_keeps_previous`, `test_second_run_is_rejected_while_first_is_active`, and `test_run_directory_creation_failure_keeps_previous_run`. Use temporary roots and generator functions that create a summary/SVG, return `False`, or raise; patch `tempfile.mkdtemp` to raise `OSError` in the last test.
- [ ] **Step 2: Run the focused runner tests** with `uv run --project web --locked --extra test --python 3.13 python -m unittest discover -s tests -p test_web_visualisations.py`. Expected: FAIL because the run store does not exist.
- [ ] **Step 3: Implement the run store** with a single-run lock, unique `tempfile.mkdtemp` directories, `run.io_bound`, success verification, and cleanup rules from the spec. Return `None` for a false generator result; re-raise generation exceptions after cleaning the failed directory.
- [ ] **Step 4: Verify concurrency and cleanup tests**. Expected: only one run executes at a time; successful output replaces and removes old output; failures preserve prior output and remove their own partial directory.
- [ ] **Step 5: Commit** the run store and tests.

### Task 4: Add Menu And Visualization Pages

**Files:**
- Modify: `web/app.py`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- `/` is a compact menu linking to `/review` and `/visualisations`.
- `/review` builds the existing reviewer and still hides `Remove empty`.
- `/visualisations` shows a Generate action; opening the page never starts generation. Success shows the summary and SVG selector; selection displays that SVG using `ui.image(Path)`.
- Generation uses the Task 3 store, displays a spinner, disables the action while running, and reports errors without discarding previous results.

- [ ] **Step 1: Write tests** named `test_home_links_to_review_and_visualisations`, `test_review_page_remains_available_at_review`, `test_visualisations_page_does_not_generate_on_open`, `test_successful_run_shows_summary_and_selected_chart`, and `test_failed_run_shows_error_and_preserves_previous_chart`. Use the existing NiceGUI `User`/ASGI lifespan test harness and temporary CSV/output fixtures.
- [ ] **Step 2: Run the focused web tests** with `uv run --project web --locked --extra test --python 3.13 python -m unittest discover -s tests -p test_web_app.py`. Expected: menu and visualisation cases fail; existing reviewer route behavior remains testable.
- [ ] **Step 3: Move the reviewer route** from `@ui.page("/")` to `@ui.page("/review")` and add the two-link home menu at `/`.
- [ ] **Step 4: Add `/visualisations`** with no automatic generation, a disabled/spinner state during `VisualisationRunStore.generate`, summary display, and one selected `ui.image(Path)` chart.
- [ ] **Step 5: Run focused web and visualization runner tests**. Expected: all three pages behave as specified; health and unknown-route responses remain unchanged.
- [ ] **Step 6: Commit** the menu, page, and tests.

### Task 5: Bound Temporary Storage And Document The Menu

**Files:**
- Modify: `compose.yaml`
- Modify: `web/README.md`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Compose grants the existing `/tmp` tmpfs a 64 MiB limit; no new host mount is added.
- Documentation links the two local pages and describes transient output behavior.

- [ ] **Step 1: Raise `/tmp` tmpfs** from `size=16m` to `size=64m` without changing the existing CSV or image/model mounts.
- [ ] **Step 2: Update `web/README.md`** to document the main menu, `/review`, `/visualisations`, manual Generate action, in-browser chart selection, the 64 MiB temporary limit, and that results disappear when the container stops. Keep localhost-only access and existing UID/GID startup commands.
- [ ] **Step 3: Update the Python 3.13 CI job** to discover both web test files using `-p 'test_web*.py'`; retain the web lock check and existing Python 3.10/3.14 root matrix.
- [ ] **Step 4: Run final checks**: web lock check; visualizer tests; run-store tests; Python 3.13 web tests; root lock check; compile check; complete repository suite; workflow YAML parse; Compose validation; image build; and container route/health smoke checks with UID/GID supplied.
- [ ] **Step 5: Review the final diff** for writes to host `docs/diagrams/`, unbounded storage, concurrent Matplotlib calls, non-loopback publishing, or ML dependencies. Commit docs, CI, and Compose changes.

## Stop Conditions

- Keep access limited to `127.0.0.1`; do not enable LAN or internet access.
- Do not modify the real CSV, labels, images, models, or host `docs/diagrams/` during visualization tests.
- On generation failure, preserve previous successful results and remove only the failed temporary output.
- Do not claim Pi hardware or architecture validation without testing on the Pi.
