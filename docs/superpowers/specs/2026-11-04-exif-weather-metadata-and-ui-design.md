# EXIF, weather metadata, and UI consistency

## Goal

Use one trustworthy capture-time source and one consistent weather source in the
wildlife data pipeline, while applying the requested small UI corrections.
Preserve model and human-review data, including unrelated changes already in the
worktree.

## Approved behavior

- The Review navigation toggle uses the same Material chevron icon and behavior
  as every other page.
- Each row in the landing-page Longest visits table shows the number of images
  in that visit.
- The camera-map column header reads `Sightings in the last 24h`.
- Gallery lighting is sourced from `labels/image_metadata.csv`, not a duplicate
  lighting value in model prediction CSVs.
- `misclassified` continues to mean a pending model error. A model `class` of
  `none` is an explicit no-accepted-detection result from the detector. When a
  human label identifies an animal for that image, this is a false negative and
  should remain in the review queue. Validation records with an `incorrect`
  status but no stored predicted label remain identifiable as unresolved errors.
- OCR code, OCR-only dependencies, configuration, and documentation are
  removed. EXIF `timestamp_exif` is the sole capture-time field. The OCR-derived
  `timestamp` field and its fallback paths are removed from schemas and tracked
  CSVs.
- `temperature_celsius` is populated from the Open-Meteo Historical Forecast
  API's hourly `temperature_2m` matched to the camera location and EXIF capture
  hour. Its historical forecast coverage from 2022 includes this dataset's
  2024–2026 captures. A repeatable batch command performs the network
  enrichment; image processing does not gain a runtime network dependency.
- If weather data is unavailable for a particular hour, that value remains
  blank and the batch reports the number of unresolved rows. API and data
  errors are surfaced rather than silently presented as successful enrichment.
- The batch updates metadata and prediction CSVs by `(location_id, image_file)`,
  retaining all unrelated columns and values. In particular, existing `reviewed`
  fields in a worktree-modified prediction CSV must be preserved.

## Components and data flow

1. `src/labelling/label_images.py` records the EXIF timestamp and calculated
   lighting but does not run OCR. The shared label-data schemas no longer
   contain the OCR timestamp.
2. A small reusable weather-enrichment module/command reads camera coordinates
   from `config/camera_locations.json`, reads EXIF times from
   `labels/image_metadata.csv`, and retrieves hourly Open-Meteo historical
   temperatures. Requests are grouped by camera and date range; returned
   hourly values are joined by camera-local capture hour.
3. Enriched temperature values are written atomically to the metadata CSV and
   mirrored into existing model-prediction CSVs by image identity. Other
   prediction and review fields remain unchanged.
4. `src/labelling/label_data.py` composes canonical metadata into human-label
   rows. `web/gallery.py` also overlays canonical lighting for direct
   prediction-CSV galleries, so all gallery sources follow the same rule.
5. Existing visit grouping already exposes `Visit.image_count`; the home table
   adds that property without changing visit selection or ranking.

## Data migration and compatibility

- Update the metadata and prediction column definitions, writers, and readers
  to remove `timestamp`. Capture-time consumers use `timestamp_exif` only.
- Update the tracked metadata and prediction CSV headers and values. Preserve
  row identity, ordering where practical, and every non-target prediction
  value. Update temperature columns from Open-Meteo; do not infer temperature
  from image pixels or OCR.
- Avoid rewriting human labels, boxes, model classes, confidence values, or
  `reviewed` statuses.
- Remove OCR-specific tests and replace them with tests proving EXIF-only
  output and that no OCR implementation is invoked.
- Document that capture times are EXIF-only and temperatures are weather
  estimates from Open-Meteo, distinct from camera-sensor readings.

## Testing

- Add focused tests before code changes for the Review toggle, visit image
  counts, label text, canonical gallery lighting, prediction-CSV preservation,
  EXIF-only schemas, and weather-hour joins including unavailable values.
- Verify the batch against a small deterministic fixture using a mocked HTTP
  response; verify malformed or failed API responses are reported.
- Run the focused tests, the complete unittest suite, compile checks, and
  `uv lock --check`.
- Validate every migrated CSV for expected headers, stable row identities,
  unchanged non-target fields (including `reviewed`), and temperature
  consistency between metadata and prediction files.

## Risks and boundaries

- Open-Meteo returns gridded hourly weather estimates, not camera sensor
  temperatures; use a clear label in UI/docs where the value is described.
- Historical coverage or an API outage can leave individual hours unresolved;
  report those rows and keep the operation safely repeatable.
- OCR was disabled by default, but its code, optional dependencies, tests, and
  the legacy timestamp column still exist. Remove all of these surfaces, not
  only the currently disabled call.
- Do not overwrite or discard the unrelated local edits in
  `labels/labels_deepfaune_finetuned_20261103-091800_018b8ca6/predictions.csv`
  or `tests/test_web_app.py`.
