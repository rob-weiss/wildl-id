# Open-Meteo Temperature Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OCR-read temperatures in tracked metadata and prediction CSVs with Open-Meteo hourly weather estimates matched by camera and EXIF capture hour.

**Architecture:** Add a repeatable, dry-run-capable batch command that reads camera coordinates and canonical metadata, fetches UTC hourly `temperature_2m` from Open-Meteo Historical Forecast, and updates metadata and existing prediction CSVs by image identity. Reuse the repository’s atomic CSV writer; report unresolved rows and preserve all non-temperature fields.

**Tech Stack:** Python 3.14, standard-library `urllib`, pandas only if already needed by the shared data layer, `unittest`.

**Spec:** [2026-11-04-exif-weather-metadata-and-ui-design.md](../specs/2026-11-04-exif-weather-metadata-and-ui-design.md)

## Global Constraints

- “`temperature_celsius` is populated from the Open-Meteo Historical Forecast API's hourly `temperature_2m` matched to the camera location and EXIF capture hour.”
- “A repeatable batch command performs the network enrichment; image processing does not gain a runtime network dependency.”
- “If weather data is unavailable for a particular hour, that value remains blank and the batch reports the number of unresolved rows.”
- “API and data errors are surfaced rather than silently presented as successful enrichment.”
- “The batch updates metadata and prediction CSVs by `(location_id, image_file)`, retaining all unrelated columns and values.”
- “In particular, existing `reviewed` fields in a worktree-modified prediction CSV must be preserved.”
- “Do not overwrite or discard the unrelated local edits in `labels/labels_deepfaune_finetuned_20261103-091800_018b8ca6/predictions.csv` or `tests/test_web_app.py`.”

## Review Focus

- Camera without configured coordinates: leave its weather values blank and count them unresolved; test the batch report.
- Missing or invalid EXIF time: do not query or match weather by another timestamp; leave the row unresolved.
- Naive camera-local EXIF times in a DST fold/gap: use `src.visits.exif_datetime`; unresolved times remain blank.
- Missing hour or null temperature from a valid Open-Meteo response: leave that image blank and include it in the unresolved count.
- HTTP errors, malformed JSON, or mismatched hourly arrays: raise/report an explicit failure and do not write partial output.

---

### Task 1: Add tested Open-Meteo hourly lookup and batch update

**Files:**
- Create: `src/enrich_weather.py`
- Test: `tests/test_enrich_weather.py`
- Modify: `README.md`
- Modify: `docs/project-guide.md`

**Interfaces:**
- Consumes: `src.visits.exif_datetime(row)`, `src.labelling.label_data.read_records`, `write_records`, `image_key`, and `config/camera_locations.json`.
- Produces:
  - `fetch_hourly_temperature(latitude: float, longitude: float, start_date: date, end_date: date) -> dict[datetime, float]`, keyed by UTC hour.
  - `enrich_temperatures(repo_dir: Path, dry_run: bool = False) -> tuple[int, int]`, returning `(updated_count, unresolved_count)`.
  - `python -m src.enrich_weather [--dry-run]`, whose network or schema failures exit nonzero and do not write CSVs.

- [ ] **Step 1: Write failing lookup and batch tests**

Test that `fetch_hourly_temperature` requests `hourly=temperature_2m` with `timezone=UTC`, parses hourly timestamps to UTC-aware keys, omits null hourly values, and rejects malformed array lengths or invalid non-null temperatures. Test batch joins by image identity, floors capture instants to the UTC hour, ignores missing/invalid EXIF times, leaves missing camera/hour values blank, preserves every non-temperature field including `reviewed`, and performs no writes in dry-run mode. Assert API errors leave all CSV bytes unchanged.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run --locked --python 3.14 python -m unittest tests.test_enrich_weather`

Expected: FAIL because the weather module and batch command do not exist.

- [ ] **Step 3: Implement the Open-Meteo request and UTC-hour parser**

Use `https://historical-forecast-api.open-meteo.com/v1/forecast`, query a single location with `hourly=temperature_2m` and `timezone=UTC`, and validate that the returned `hourly.time` and `hourly.temperature_2m` arrays have matching lengths. Treat null temperatures as unavailable hours; surface HTTP, JSON, schema, and invalid non-null temperature failures.

- [ ] **Step 4: Implement safe metadata and prediction updates**

Load cameras from `config/camera_locations.json`, derive per-camera UTC date ranges from valid EXIF instants, fetch hourly weather, and join exact UTC-hour keys. Update `temperature_celsius` in canonical metadata and mirror values into every existing `labels/*/predictions.csv` by `(location_id, image_file)`. Define `updated_count` as unique metadata rows matched to a weather value and `unresolved_count` as unique metadata rows without a valid camera/time/hour match. Leave unmatched metadata rows blank and prediction rows without a metadata key unchanged; never alter non-temperature values. Prepare and validate all output rows before writing; write each CSV atomically with `label_data.write_records`. `--dry-run` prints counts without writes.

- [ ] **Step 5: Add operator documentation**

Document `uv run python -m src.enrich_weather --dry-run` and `uv run python -m src.enrich_weather`, the hourly weather-estimate meaning, unresolved-row reporting, and rerun safety.

- [ ] **Step 6: Run focused and full verification**

Run `uv run --locked --python 3.14 python -m unittest tests.test_enrich_weather`, then:
`uv lock --check`
`uv run --locked --python 3.14 python -m compileall -q src wildl_id.py tests`
`PYTHONPATH=. uv run --locked --python 3.14 python -m unittest discover -s tests`

Expected: all checks pass. Run the batch in dry-run mode against repository CSVs and verify it reports counts without changing file hashes.

- [ ] **Step 7: Commit implementation, then separately run the approved migration**

```bash
git add src/enrich_weather.py tests/test_enrich_weather.py README.md docs/project-guide.md
git commit -m "feat(metadata): add Open-Meteo temperature enrichment"
```

After review, run the batch first with `--dry-run`; then run the write mode as a separate explicit operation. Verify image identities and all non-temperature prediction fields—including the existing `reviewed` edits—are unchanged, and report updated/unresolved counts.
