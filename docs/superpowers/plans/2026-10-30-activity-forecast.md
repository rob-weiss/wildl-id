# Wildlife Activity Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast 24 hourly unique-image counts per wildlife species and camera,
comparing a small deep TCN with a seasonal baseline and Poisson regression.

**Architecture:** Validate camera observations and coverage before making hourly
training examples. Join shared Holzgerlingen weather and deterministic astronomy
at explicit forecast origins. Fit and compare models chronologically, exporting
expected counts, a training-reference index, and provenance through a standalone CLI.

**Tech Stack:** Python 3.10-compatible code, pandas, NumPy, scikit-learn, PyTorch,
Astral, PyEphem, unittest; add only `openmeteo-requests` and resolved dependencies.

**Spec:** `docs/superpowers/specs/2026-10-30-activity-forecast-design.md`.

## Global Constraints

- Each image contributes at most once per species; five images with several roe
  deer each count as five, not the number of animals or detection boxes.
- Support the nine species selected by the existing visualisation script and
  honour manual class corrections.
- Start with seven days of recent context; forecast the next 24 elapsed hours.
- One shared weather location in Holzgerlingen, Germany; camera locations remain
  distinct model inputs.
- Unknown observation coverage is not zero sightings. Require reviewed coverage
  for training and planned operation for the forecast horizon.
- No web integration, automatic scheduling, independent-visit grouping,
  model-selection framework, or calibrated prediction intervals in this version.
- No edits to existing labelled data, camera EXIF, or visualisation output.
- Follow repository unittest conventions and Python 3.10 compatibility.
- Use British English in documentation and user-facing output.
- Fit transforms, support thresholds, and normalisation references on training
  data only. Do not claim neural, weather, or lunar improvement without evidence.

## Review Focus

- A CSV containing multiple detections and corrected classes must not double-count
  an image/species or discard another species in the same image (Task 1).
- A stale forecast initialised before issuance but published afterwards must be
  excluded; archived acquisition time differs from historical availability (Task 2).
- Partial ingestion and DST transitions must not admit future observations or
  change a 24-hour output into 23 or 25 rows (Task 3).
- A rare species, all-zero training target, or unseen location must produce a
  labelled fallback, not an unstable fit or missing row (Tasks 4 and 5).
- Neural checkpoint reloads must preserve species/site order and feature scaling;
  an incompatible checkpoint must fail clearly (Tasks 5 and 6).

## File Ownership And Contracts

Create `src/forecasting/activity.py`, `weather.py`, `neural.py`, and `predictor.py`.
Keep the new package compatible with the repository's existing `src` imports and
module CLI invocation: `python -m src.forecasting.predictor`.
Create `tests/test_activity_forecast.py` using synthetic DataFrames, temporary
directories, and mocked network clients. Add `docs/activity-forecast.md` with
configuration schemas, runnable commands, attribution, and limitations.
Modify only `pyproject.toml` and `uv.lock` for the weather dependency.

Use frozen dataclasses in `activity.py` for shared contracts:

- `SiteConfig(location_id: str, latitude: float, longitude: float,
  local_timezone: str, timestamp_timezone: str, ingestion_delay_hours: int)`.
- `ForecastConfig(sites: tuple[SiteConfig, ...], weather_latitude: float,
  weather_longitude: float, weather_model: str = "icon_d2",
  publication_delay_hours: int = 3, max_weather_age_hours: int = 12)`.
- `OriginBatch(origin: pd.Timestamp, location_id: str, species: tuple[str, ...],
  history: np.ndarray, history_mask: np.ndarray, future: pd.DataFrame,
  targets: np.ndarray | None, metadata: dict[str, object])`.
  Array order is time then species/features. Targets have shape `(24, 9)`;
  species order is sorted `VISUALIZATION_CLASSES`, saved in every artefact.

Configuration is JSON with `weather` and `sites` objects. `weather` records
`name: "Holzgerlingen, Germany"`, explicit latitude/longitude, model, and delays;
do not guess coordinates or copy Renningen. Validate latitude/longitude ranges,
unique NFC-normalised site IDs, supported timezones, and nonnegative integer
delays. Accept timestamp timezones `UTC+01:00` or valid IANA names. Coverage CSV
columns are `location_id,start_utc,end_utc`, with hour-aligned half-open intervals.

`future` contains a UTC `valid_time` column, horizon 1 through 24, cyclic local
hour/year, solar elevation/day length, optionally moon illumination/altitude, and
named target-hour weather columns when requested. History consists of sorted
species `log1p` counts followed by named weather columns; store the exact feature
names and masks. Require complete count context in this version. All arrays must
be finite. Serialise schema version, configuration, orders, training cutoff,
scalers, training maxima, support, and model hyperparameters with model state.

## Task 1: Validated Hourly Observations And Audit

**Files:** Create `src/forecasting/activity.py` and
`tests/test_activity_forecast.py`.

**Interfaces:** Produce `load_config(path: Path) -> ForecastConfig`,
`load_sightings(path: Path, config: ForecastConfig) -> pd.DataFrame`,
`load_coverage(path: Path, config: ForecastConfig) -> pd.DataFrame`,
`hourly_counts(sightings: pd.DataFrame, coverage: pd.DataFrame,
config: ForecastConfig) -> pd.DataFrame`, and
`audit(sightings: pd.DataFrame, coverage: pd.DataFrame | None) -> dict[str, object]`.
Hourly rows have `location_id`, `valid_time`, `species`, and `count`.

- [ ] **1. Write failing unittest cases.** Assert five distinct roe-deer images
  produce count 5 even with duplicate boxes; an image with fox and roe deer counts
  once for each. Assert blank manual labels fall back and corrections take effect.
  Assert covered empty hours count zero, uncovered hours have no target, conflicting
  image timestamps raise `ValueError`, and audit works without coverage. Test
  equivalent decomposed/composed site names, missing image names, invalid timestamps,
  unknown sites, invalid coordinates, and overlapping/non-aligned intervals.
- [ ] **2. Run the tests and confirm failure for missing forecasting interfaces.**
  Run `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_activity_forecast.py`.
- [ ] **3. Implement the interfaces minimally.** Reuse the existing
  `filter_visualization_classes` and `VISUALIZATION_CLASSES` without invoking
  visualisation `main`. Prefer nonblank `timestamp_exif`, then `timestamp`; require
  one source. Preserve aware offsets, localise naive input according to site
  configuration, and reject DST ambiguity. Produce UTC bins and audit counts,
  dates, duplicate contributions, uncovered sightings, and positive-hour support.
- [ ] **4. Rerun the focused command; require all new tests to pass.** Also run
  `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_evaluate_labels.py`.
- [ ] **5. Review and commit only the new module and tests:**
  `feat(forecast): validate hourly sighting data`.

## Task 2: Shared Holzgerlingen Weather And Archived Runs

**Files:** Create `src/forecasting/weather.py`; extend the existing forecast test
file; update `pyproject.toml` and `uv.lock`.

**Interfaces:** Consume `ForecastConfig`. Produce
`WeatherSnapshot(frame: pd.DataFrame, metadata: dict[str, object])`,
`WeatherClient(config: ForecastConfig, cache_dir: Path, transport: object | None = None)`,
`WeatherClient.history(start: pd.Timestamp, end: pd.Timestamp) -> WeatherSnapshot`,
`WeatherClient.forecast(origin: pd.Timestamp, archived: bool) -> WeatherSnapshot`,
and `align_weather(snapshot: WeatherSnapshot, hours: pd.DatetimeIndex) -> pd.DataFrame`.
The transport defaults to `openmeteo_requests.Client` and is injectable for tests.
Keep raw instantaneous and accumulated variable timestamps until alignment.

- [ ] **1. Write failing tests with an injected fake client.** Assert all sites
  request the same configured weather coordinates, correct six variables and
  model, and cache keys differ by run/model/coordinate/period. At 06:00 UTC with a
  three-hour assumed publication delay, exclude a 06:00 run and select the 03:00
  ICON-D2 cycle. Assert precipitation stamped 01:00 joins count bin 00:00, and the
  final target bin obtains its end-boundary precipitation. Test absent variables,
  mismatched lengths/units, invalid cache metadata, unavailable archives, and stale
  runs. Historical fetch time must not masquerade as original publication time.
- [ ] **2. Run focused unittest discovery and confirm the new cases fail.**
- [ ] **3. Add `openmeteo-requests` using `uv add openmeteo-requests` and implement
  the adapter.** Use model-specific three-hour run cycles for ICON-D2; support only
  this model initially and reject others explicitly instead of inventing cycle
  rules. Pin variable names: `temperature_2m`, `relative_humidity_2m`,
  `pressure_msl`, `precipitation`, `wind_speed_10m`, `cloud_cover`. Store structured
  metadata and numerical arrays atomically, distinguish live and archived cache
  identities, and bound network timeout/retries. Report source, run initialisation,
  acquisition time, and assumed publication delay. Never silently substitute models.
- [ ] **4. Rerun focused tests and `uv lock --check`; require success.**
- [ ] **5. Review and commit module, tests, and dependency changes:**
  `feat(forecast): acquire provenance-aware weather`.

## Task 3: Leakage-Safe Examples And Astronomy

**Files:** Extend `activity.py` and the forecast test file.

**Interfaces:** Consume Task 1 counts/configuration and Task 2 snapshots. Produce
`build_origin(counts: pd.DataFrame, config: ForecastConfig, location_id: str,
origin: pd.Timestamp, weather_history: WeatherSnapshot | None,
weather_forecast: WeatherSnapshot | None, include_moon: bool = False,
require_targets: bool = False) -> OriginBatch` and
`eligible_origins(counts: pd.DataFrame, config: ForecastConfig,
start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex`.
Daily origins are 00:00 UTC. Return skipped-origin reasons separately in callers.

- [ ] **1. Write failing tests.** Assert history has exactly 168 elapsed hours
  ending before the ingestion cutoff; target shape is `(24, 9)` when requested.
  Changing any post-cutoff sighting leaves input features unchanged. Test a
  two-hour ingestion delay, fixed UTC+1 camera clocks, both Berlin DST transitions,
  exactly 24 forecast bins, complete-zero context, unknown context rejection, and
  solar/moon values independent of sighting targets. Assert future weather cannot
  be accidentally replaced by realised target weather.
- [ ] **2. Run focused discovery and confirm these cases fail.**
- [ ] **3. Implement deterministic feature construction.** Define the history
  interval as `[origin - delay - 168h, origin - delay)`; require full covered
  context for all selected species. Use Astral/PyEphem with explicitly configured
  site coordinates, deriving local calendar features after UTC joins. Include
  recent 24h/168h count means, target-local-hour history mean, weather means/rain
  sum and pressure/temperature change. Record coordinates and time assumptions.
- [ ] **4. Rerun focused discovery; require all examples and leakage tests to pass.**
- [ ] **5. Review and commit:** `feat(forecast): construct causal forecast examples`.

## Task 4: Baseline And Poisson Count Forecasts

**Files:** Create `src/forecasting/predictor.py`; extend the forecast test file.

**Interfaces:** Consume `OriginBatch`. Produce
`baseline(batch: OriginBatch) -> np.ndarray`,
`fit_poisson(batches: list[OriginBatch], include_weather: bool,
include_moon: bool = False) -> dict[str, object]`,
`predict_poisson(bundle: dict[str, object], batch: OriginBatch) -> np.ndarray`,
and `normalise_counts(predicted: np.ndarray, reference: np.ndarray)
-> tuple[np.ndarray, np.ndarray]` (index and clipping mask).
Bundles retain site/species order, training support, cutoff, maxima and transforms.

- [ ] **1. Write failing tests.** Assert baseline matches known same-local-hour
  means and fallback context means. Require nonnegative finite Poisson outputs.
  Use the baseline for fewer than 30 distinct daily training origins, fewer than
  30 positive target hours for a species, all-zero targets, or unseen sites.
  Assert predicted count 5/reference 10 gives index 0.5; count 12 gives index 1
  with clipping true; reference 0 gives missing index. Alter held-out counts and
  verify the fitted maximum and scalers remain unchanged.
- [ ] **2. Run focused discovery and confirm failure.**
- [ ] **3. Implement baseline and scikit-learn pipelines.** Fit a regularised
  `PoissonRegressor(alpha=1.0, max_iter=1000)` per species with training-only
  numeric standardisation and one-hot location encoding. Do not standardise
  one-hot columns. Keep explicit model/fallback statuses. Compute maxima only
  from covered training target hours; never from prediction context or holdout.
- [ ] **4. Rerun focused discovery; require success without network requests.**
- [ ] **5. Review and commit:** `feat(forecast): add count baselines and regression`.

## Task 5: Small Shared Deep TCN

**Files:** Create `src/forecasting/neural.py`; extend the forecast test file.

**Interfaces:** Consume `OriginBatch`. Produce
`ActivityTCN(history_features: int, future_features: int, site_count: int,
species_count: int = 9)` with
`forward(history: torch.Tensor, future: torch.Tensor,
site_ids: torch.Tensor) -> torch.Tensor`,
`fit_tcn(train: list[OriginBatch], validation: list[OriginBatch],
include_moon: bool = False, max_epochs: int = 100, seed: int = 42)
-> dict[str, object]`, and
`predict_tcn(bundle: dict[str, object], batch: OriginBatch) -> np.ndarray`.
Keep neural imports lazy in the CLI and baseline modules.

- [ ] **1. Write failing tests.** For batch size 2, assert output shape
  `(2, 24, 9)`, finite positive means, and gradients reaching historical inputs
  168 hours back. Test causal blocks by perturbing later history positions and
  checking earlier states remain unchanged. Reject non-finite arrays and
  overlapping training/validation target windows. Verify same seed reproduces a
  short CPU training run, saved state reloads identical predictions, and reordered
  species/features or unknown sites cannot silently use the wrong embedding.
- [ ] **2. Run focused discovery; require the neural cases to fail.**
- [ ] **3. Implement the specified seven-block TCN using PyTorch primitives.**
  Use left padding only, kernel 3, dilations 1 through 64 by powers of two,
  16 channels, residual projections, ReLU and dropout 0.1. Decode the final state
  plus four-dimensional site embedding and each future-hour feature vector through
  `Linear -> ReLU -> Linear -> Softplus`, hidden size 32. Optimise raw target
  counts with `PoissonNLLLoss(log_input=False)`, AdamW learning rate 0.001,
  weight decay 0.0001, batch size 32, gradient clipping norm 1, and patience 10.
  Use training-only weather scaling and `log1p` count inputs; restore the best
  chronological validation state. Use CPU by default for deterministic proof.
  Apply Task 4 support fallbacks per species/site and record the actual method.
- [ ] **4. Run short CPU tests, not full training.** Require positive predictions,
  finite loss/gradients, and successful save/reload; do not assert TCN superiority
  on synthetic random data. A deterministic constant-target fixture must show
  lower loss after optimisation than before it.
- [ ] **5. Review and commit:** `feat(forecast): add shared temporal neural model`.

## Task 6: Chronological Backtests And Standalone CLI

**Files:** Extend `predictor.py` and the forecast test file.

**Interfaces:** Consume Tasks 1-5. Produce
`backtest(batches: list[OriginBatch], validation_start: pd.Timestamp,
test_start: pd.Timestamp, include_moon: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]`,
`save_bundle(bundle: dict[str, object], directory: Path) -> None`,
`load_bundle(directory: Path) -> dict[str, object]`, and
`main(argv: list[str] | None = None) -> int`.
Use JSON metadata, NumPy arrays, and PyTorch state dictionaries with safe tensor
loading; if scikit-learn persistence needs pickle, restrict it to explicitly trusted
local artefacts and document that restriction. Reject schema/config mismatches.

- [ ] **1. Write failing CLI and backtest tests using temporary synthetic files.**
  Exercise `audit`, `backtest`, and `forecast`; verify exactly `24 * 9 * sites`
  forecast rows and unchanged source files. Assert no training target overlaps
  validation/test, early stopping never reads final holdout targets, and all methods
  compare the same eligible origins. Require zero/count baseline and regression/TCN
  metrics, skipped-origin reasons, and status-labelled fallback on live weather
  failure. Reject missing coverage, unknown configuration, absent model files,
  invalid output paths, and corrupt/incompatible state before publishing outputs.
- [ ] **2. Run focused discovery; confirm CLI tests fail.**
- [ ] **3. Implement argparse commands and reporting.** Shared arguments:
  `--labels`, `--config`, `--coverage`, `--cache-dir`, `--output`.
  Audit allows omitted coverage. Backtest additionally requires `--validation-start`
  and `--test-start`; chronological training targets must end by validation start,
  validation targets by test start. Freeze model choices before final holdout,
  with no automatic winner selection. Save each fitted bundle under an explicit
  `--model-dir`. Forecast loads that bundle, accepts `--origin`, and never refits
  from live context. Default live origin is the next whole UTC hour after weather
  acquisition, not an hour that elapsed before fetching it. Use a labelled baseline
  when weather is unavailable and context/coverage are still valid.
- [ ] **4. Export rows and metrics atomically.** Include all fields specified by
  the design, MAE and Poisson deviance overall/by species/site/horizon, and support
  counts. Use epsilon only for deviance computation. Export raw per-origin
  predictions so paired comparisons are inspectable. Do not serialise secrets or
  assert coverage or accuracy that was not measured.
- [ ] **5. Rerun focused discovery and the offline synthetic CLI workflow.**
  Require exit code 0 for valid audit/backtest/forecast, actionable nonzero errors
  for invalid prerequisites, and no routine-test network calls.
- [ ] **6. Review and commit:** `feat(forecast): expose evaluated activity forecasts`.

## Task 7: Usage Documentation And Repository Gates

**Files:** Create `docs/activity-forecast.md`; update the forecast test file only
if the end-to-end exercise reveals a defect in the touched slice.

**Interfaces:** Document Task 6 CLI without introducing new commands or modes.

- [ ] **1. Write documentation with JSON and coverage CSV examples.** Explain how
  to obtain and record a representative Holzgerlingen coordinate, provide reviewed
  camera coverage, choose civil versus fixed-offset clock conventions, and declare
  ingestion delay. Label all example coverage as synthetic, not actual deployments.
  Explain image-count semantics, 168h/24h shapes, baseline/Poisson/TCN differences,
  training-only index, archive limitations, weather attribution, checkpoint trust,
  and why absent real metadata blocks a meaningful real forecast.
- [ ] **2. Run the documented workflow on temporary synthetic fixtures.** Confirm
  command flags match help output, generated CSV schemas match documentation, and
  the neural model loads its state without retraining. No live network is required.
- [ ] **3. Run required gates from the repository root:**
  `uv lock --check`;
  `uv run --locked python -m compileall -q src wildl_id.py tests`;
  `PYTHONPATH=. uv run --locked python -m unittest discover -s tests`.
  Require success or identify unrelated pre-existing failures without expanding
  scope. Do not bypass hooks or claim tests that were not run.
- [ ] **4. Review the final diff against the design and this plan.** Check label
  files and existing user edits remain untouched. Commit only verified owned changes:
  `docs(forecast): explain activity prediction workflow`.

## Handoff And Stop Condition

The plan is ready for user review. Recommended execution method: native execution
in this session, task by task, followed by one independent review, because the
tasks share strict time, shape, and provenance contracts. Subagent-driven execution
with fresh per-task reviews is an alternative at higher context cost.

Implementation stops after the working CLI, synthetic regression proof, documented
limitations, and repository gates. Real-data skill claims and live forecasts remain
blocked until reviewed coverage, site coordinates, camera clocks, and usable
weather archive periods are supplied. Do not fabricate these inputs to demonstrate
the feature.