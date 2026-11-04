# EXIF Metadata Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove OCR and the duplicate OCR timestamp from the metadata pipeline, use EXIF as the sole capture-time source, and make canonical image metadata authoritative for gallery lighting.

**Architecture:** Keep `timestamp_exif` as the only stored capture time, remove OCR implementation and OCR-only packages, and make label-data composition plus direct gallery CSV loading use `labels/image_metadata.csv` for lighting and temperature. Existing temperature values remain until the separate Open-Meteo enrichment plan replaces them.

**Tech Stack:** Python 3.14, pandas, Pillow, uv, `unittest`.

**Spec:** [2026-11-04-exif-weather-metadata-and-ui-design.md](../specs/2026-11-04-exif-weather-metadata-and-ui-design.md)

## Global Constraints

- “OCR code, OCR-only dependencies, configuration, and documentation are removed.”
- “EXIF `timestamp_exif` is the sole capture-time field.”
- “The OCR-derived `timestamp` field and its fallback paths are removed from schemas and tracked CSVs.”
- “Gallery lighting is sourced from `labels/image_metadata.csv`, not a duplicate lighting value in model prediction CSVs.”
- “Avoid rewriting human labels, boxes, model classes, confidence values, or `reviewed` statuses.”
- “Do not overwrite or discard the unrelated local edits in `labels/labels_deepfaune_finetuned_20261103-091800_018b8ca6/predictions.csv` or `tests/test_web_app.py`.”

## Review Focus

- Missing or malformed EXIF timestamp: no fallback to the removed `timestamp` field; test in `tests/test_visits.py` or existing consumer tests.
- Conflicting canonical metadata and prediction lighting: gallery must use the metadata value; test direct prediction-CSV loading with temporary metadata.
- A metadata row missing from `image_metadata.csv`: keep prediction identity and nonmetadata fields intact, with lighting left blank rather than sourced from prediction data.
- Human labels, boxes, model classes, confidence, and `reviewed` values during schema migration: compare unchanged fields and image-key sets in migration tests.
- OCR symbols and optional dependencies absent after update: verify imports/search and `uv lock --check`.

---

### Task 1: Remove OCR and the legacy timestamp field

**Files:**
- Modify: `src/labelling/label_images.py`
- Modify: `src/labelling/label_data.py`
- Modify: `src/labelling/finetune_classifier.py`
- Modify: `src/labelling/review_labels.py`
- Modify: `src/alerting/push_alerts.py`
- Modify: `web/predictions.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`
- Modify: `docs/project-guide.md`
- Modify: `docs/data-licence.md`
- Modify: `web/README.md`
- Test: `tests/test_label_images.py`
- Test: `tests/test_label_data.py`
- Test: consumer tests covering prediction time, visit time, and alerts

**Interfaces:**
- Consumes: existing EXIF extraction in `label_images.extract_exif_timestamp` and `src.visits.exif_datetime`.
- Produces: `METADATA_COLUMNS` and `PREDICTION_COLUMNS` without `timestamp`; label processing writes `timestamp_exif` directly from EXIF; consumers use no timestamp fallback.

- [ ] **Step 1: Write failing EXIF-only and schema tests**

Replace OCR-specific tests with assertions that schemas omit `timestamp`, EXIF capture values are retained, and input rows containing a conflicting legacy `timestamp` do not affect visit ordering, predictions, review order, or alert timestamps. Add an assertion that `image_metadata.csv` has exactly the metadata schema without the legacy field.

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `uv run --locked --python 3.14 python -m unittest tests.test_label_images tests.test_label_data tests.test_visits tests.test_web_predictions tests.test_push_alerts`

Expected: FAIL on the current legacy schema, OCR symbols, and fallback behavior.

- [ ] **Step 3: Remove OCR implementation and schema/fallback paths**

Delete OCR parsers, framework detection/imports, runtime flags, OCR failure handling, and `resolve_exif_timestamp` from `src/labelling/label_images.py`; write the EXIF extraction result directly to `timestamp_exif`. Remove `timestamp` from `METADATA_COLUMNS`, `PREDICTION_COLUMNS`, dtypes, and processing logic. Remove timestamp fallback reads from `src/labelling/finetune_classifier.py`, `src/labelling/review_labels.py`, `src/alerting/push_alerts.py`, and `web/predictions.py`. Remove EasyOCR and macOS Vision dependencies, regenerate `uv.lock` with `uv lock`, and update documentation to describe EXIF-only timestamps and weather temperatures.

- [ ] **Step 4: Migrate tracked CSV schemas without changing unrelated values**

Remove `timestamp` from `labels/image_metadata.csv` and both tracked model prediction CSVs. Preserve image identities, ordering, every other field, and all existing `reviewed` values. Leave weather temperature values for the Open-Meteo task.

- [ ] **Step 5: Run focused tests and repository checks**

Run the focused command from Step 2, `uv lock --check`, and:
`uv run --locked --python 3.14 python -m compileall -q src wildl_id.py tests`

Expected: all tests pass, lockfile is current, compilation succeeds, and an OCR search finds no runtime implementation or dependency references outside explicitly historical design notes.

- [ ] **Step 6: Commit**

```bash
git add src pyproject.toml uv.lock README.md docs/project-guide.md docs/data-licence.md web/README.md tests labels/image_metadata.csv labels/labels_deepfaune_pretrained/predictions.csv labels/labels_deepfaune_finetuned_20261103-091800_018b8ca6/predictions.csv
git commit -m "refactor(metadata): remove OCR and use EXIF timestamps"
```

### Task 2: Make image metadata authoritative in the gallery

**Files:**
- Modify: `web/gallery.py`
- Test: `tests/test_gallery.py`
- Test: `tests/test_web_app.py`

**Interfaces:**
- Consumes: `label_data.read_records` and canonical metadata schema from Task 1.
- Produces: `gallery.load_rows(csv_path)` that overlays `lighting` and `temperature_celsius` from `labels/image_metadata.csv` for matching `(location_id, image_file)` keys, including direct model-prediction CSV inputs.

- [ ] **Step 1: Write failing canonical-lighting tests**

Add a direct-prediction-CSV gallery test where the prediction row says `bright`, the matching metadata row says `dark`, and the loaded row must be `dark`. Add a missing-metadata-row assertion that leaves `lighting` empty and preserves the prediction row.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run --project web --locked --extra test --python 3.14 python -m unittest tests.test_gallery tests.test_web_app`

Expected: FAIL because direct prediction CSVs currently bypass canonical metadata.

- [ ] **Step 3: Overlay canonical metadata in gallery loading**

For direct prediction CSVs, read metadata from the sibling `labels/image_metadata.csv` and apply canonical `lighting` and `temperature_celsius` by normalized image identity. A matching metadata row is authoritative even when either value is blank; if no metadata row exists, set gallery `lighting` blank instead of falling back to prediction lighting. Preserve all prediction identity, label, box, confidence, and review fields.

- [ ] **Step 4: Run gallery and web tests**

Run: `uv run --project web --locked --extra test --python 3.14 python -m unittest tests.test_gallery tests.test_web_app`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/gallery.py tests/test_gallery.py tests/test_web_app.py
git commit -m "fix(gallery): use canonical image metadata"
```
