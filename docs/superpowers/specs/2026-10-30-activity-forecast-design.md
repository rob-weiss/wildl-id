# Wildlife Activity Forecast Design

Status: User approved proceeding on 2026-10-30, including the neural comparison
and Holzgerlingen weather clarification. Implementation plan awaits review.

## Purpose

Predict the number of images containing each relevant species at each camera
location for the next 24 elapsed hours. Each image contributes at most once per
species, irrespective of duplicate detection rows. This is a camera-sighting
forecast, not a population estimate, independent-visit count, or probability.

The user selected per-location predictions and one image per species as the
sighting definition. Start with seven days of recent context, but train using
all suitable historical forecast origins. Support the nine species selected by
the existing visualisation script and honour manual class corrections.

## First-Version Scope

- A standalone Python module CLI for auditing data, backtesting, and forecasting.
- Explicit location, timestamp, and observation-coverage configuration.
- A count baseline, a regularised Poisson model shared across locations for
   each sufficiently supported species, and a small shared temporal convolutional
   network (TCN) as an experimental neural candidate.
- Historical and forecast weather from Open-Meteo, with local cached snapshots.
- Calendar, solar, and optional lunar features computed deterministically.
- CSV outputs containing expected counts and a training-reference activity index.
- Offline regression tests and concise usage documentation.

No web integration, automatic scheduling, independent-visit
grouping, model-selection framework, or calibrated prediction intervals in this
version. No edits to existing labelled data, camera EXIF, or visualisation output.

## Observation Contract

Location configuration supplies a stable location identifier, latitude, longitude,
local IANA timezone, the timezone convention for naive stored camera timestamps,
and the expected ingestion delay. Fixed UTC+1 camera clocks must remain distinct
from Europe/Berlin civil time. Timezone-aware input retains its explicit offset.
Reject ambiguous or nonexistent civil timestamps rather than guessing.

A separate coverage CSV supplies `location_id`, `start_utc`, and `end_utc` for
reviewed intervals in which the image collection is complete and camera settings
are consistent. Intervals are half-open. Require timezone-aware, hour-aligned,
non-overlapping intervals; a gap means unknown observation, not zero sightings.
Rows outside confirmed coverage may appear in an audit but are not model targets.
Deployment intervals alone are not evidence of complete ingestion.

Coverage must include the forecast horizon to express the user's planned camera
operation. Forecasts are conditional on that operation; they cannot predict outages.
Historical coverage is supplied by the operator, never inferred from detection gaps.
Auditing works without it, but training and forecasting fail with actionable errors.

Normalise location and image identifiers consistently with the repository. Apply
manual labels first, select relevant species, and deduplicate by location, image,
and effective species. Reject conflicting timestamps for the same image identity.
Construct chronological UTC bins `[hour, hour + 1 hour)`. Only confirmed covered
hours receive zero counts. Missing required columns and invalid timestamps produce
clear audit errors rather than silently dropping records.

## Forecast Origins And Features

An origin is an hour boundary; targets are the 24 complete hourly bins beginning
at that origin. Internal joins use UTC. Local calendar and astronomy calculations
use the configured site location and timezone. Daylight-saving changes do not
change the number of forecast rows.

At an origin, history ends at the origin minus the configured ingestion delay.
If actual per-image availability timestamps are unavailable, backtests explicitly
report that this delay is an assumption, not reconstructed historical availability.
Unknown historical coverage disqualifies an origin requiring that context.

For each target hour, construct features directly from information available at
the origin, avoiding recursive use of actual future sightings:

- Cyclic local hour and day-of-year, horizon, and encoded location.
- Seven-day mean count, last-day mean count, and recent same-local-hour mean.
- Recent weather summaries and target-hour weather forecasts.
- Solar elevation and daylight duration; optional moon illumination and altitude.

All summaries respect observation coverage and availability cutoffs. The baseline
uses the seven-day same-local-hour mean, falling back to the covered context mean
where no same-hour observation exists. It remains zero for genuinely all-zero
contexts; missing history is not treated as an all-zero context.

Fit each Poisson model using chronological historical origins, with regularisation,
feature scaling, and location encoding fitted on its training data only. Require
at least 30 distinct daily training origins with complete context and targets and
at least 30 positive target hours for the species. These are conservative initial
guardrails, not evidence of forecast skill. Sparse species and unseen locations
use the labelled baseline fallback. Store training support in output metadata.

## Weather And Astronomy

Use one shared weather location in Holzgerlingen, Germany, for all nearby camera
locations. Record its configured coordinates explicitly; do not silently reuse the
visualisation script's Renningen coordinates. Site coordinates remain separate
inputs for astronomy. If a representative coordinate is used for astronomy too,
record that approximation in configuration and output metadata.

Use the official `openmeteo-requests` client. Request temperature, relative
humidity, sea-level pressure, precipitation, wind speed, and cloud cover. A single
configured model serves both live forecasts and available archived runs; default
to ICON-D2 for the project's German sites, subject to API variable availability.
Unsupported variables or archive dates cause explicit errors, not model substitution.

Use historical model data for past conditions and archived individual model runs
for future covariates in backtesting. Only use runs published by the origin,
accounting for a configurable conservative publication delay after initialisation.
Describe that delay as assumed unless actual publication records are available.
The stitched Historical Forecast API is not an as-issued 24-hour forecast.
If the selected archive does not cover the requested period, report the supported
period and require the operator to choose a shorter evaluation interval.

Cache requests using model, coordinates, run, variables, and requested period.
Keep acquisition time and source metadata. Never overwrite an acquired live
snapshot with a later run under the same identity. Follow API attribution and
non-commercial/rate-limit terms. Mock network requests in automated tests.

Align precipitation's preceding-hour accumulation deliberately: the value stamped
at an hour boundary describes the interval ending there. Instantaneous variables
use the target bin's start; accumulated precipitation uses its end boundary.
Fetch sufficient endpoints to cover all 24 target bins.

Reuse Astral and PyEphem, already declared dependencies. Lunar features are an
optional ablation, not a built-in claim that solunar tables predict wildlife.
On missing or invalid live weather, return the labelled count baseline rather than
substituting invented weather. A failed baseline prerequisite remains an error.

## Neural Candidate

Train one small TCN across locations and species using existing PyTorch. Its input
is 168 hourly steps of species image counts and historical weather, plus coverage
masks. Complete context is required in this version; masks do not permit unknown
hours to become zeros. A site embedding distinguishes cameras.

Use seven causal residual convolution blocks with kernel size 3, 16 channels,
dilations 1, 2, 4, 8, 16, 32, 64, one convolution per block, and dropout 0.1. The
receptive field covers all 168 historical hours. Combine the final historical
state with a four-dimensional site embedding and each future hour's weather,
calendar, solar, horizon, and optional lunar features. A shared decoder with one
32-unit hidden layer and Softplus outputs predicts a 24-by-9 count-mean tensor.
No future sighting count enters the decoder.

Train using Poisson negative log likelihood, AdamW, chronological validation,
early stopping, and a fixed seed. Scale weather and log-transform count inputs
using training-only information; targets remain raw counts. Do not equate Poisson
loss with calibrated uncertainty or claim that a deep network must win.
Retain the same training-support and unseen-location fallback rules as Poisson
regression. Negative-binomial output is deferred unless later evidence warrants it.

## Backtesting And Normalisation

Use daily forecast origins and rolling chronological evaluation. Every training
target must end before the held-out origin. No random splits. Training origins
must have complete context and full 24-hour targets; validation origins likewise.
Compare zero prediction, the count baseline, a history/calendar/solar Poisson
model, that model plus weather, and the TCN with the same weather covariates.
Optionally compare lunar features for both model families. Compare methods on
the same eligible evaluation origins and report skipped-origin reasons.
Declare the feature configuration before evaluating a separate final holdout;
do not select a winning model using final-holdout scores.

Report MAE and Poisson deviance overall, by species/location, and by horizon,
alongside the number of evaluated origins and positive target hours. Use a small
positive floor only when calculating deviance, not to change reported zero counts.
Do not claim weather or lunar benefit without an observed improvement. A model
that does not outperform the baseline is an experimental candidate, not an upgrade.

Keep expected count as the primary output. The display index is
`min(1, expected_count / training_max)` for that species/location. Compute the
maximum from covered training target hours only and hold it fixed for each fitted
model. Report the reference and clipping flag. If the reference is zero or absent,
leave the index missing and explain the unsupported reference in status metadata.
Indices are not probabilities or absolute comparisons between species.

## Interfaces And Ownership

Proposed implementation files:

- `src/forecasting/activity.py`: configuration validation, canonical hourly counts,
  leakage-safe origin features, and deterministic astronomy.
- `src/forecasting/weather.py`: Open-Meteo acquisition, run selection, cache, and
  interval alignment.
- `src/forecasting/predictor.py`: audit/backtest/forecast CLI, baseline and Poisson
  training, metrics, and CSV serialisation.
- `src/forecasting/neural.py`: small TCN, deterministic training, and neural
   inference; keep PyTorch imports out of the audit-only path.
- `tests/test_activity_forecast.py`: focused synthetic regression coverage.
- `docs/activity-forecast.md`: configuration schemas, commands, limitations, and
  Open-Meteo attribution.

Reuse existing species/manual-label semantics without invoking the visualisation
script's `main`, which clears its output directory. Avoid unrelated refactoring.
Add only the weather-client dependency and its lockfile changes. Follow repository
unittest conventions and Python 3.10 compatibility.

Forecast rows contain location, species, origin, UTC and offset-aware local valid
time, horizon, expected count, activity index, reference maximum, clipping flag,
method/status, training cutoff/support, weather source/run, and availability
assumptions. Include every configured species/location/hour combination; unsupported
combinations have an explicit status rather than silently disappearing.

## Verification And Acceptance

1. Synthetic counts prove manual corrections and unique-image counting, including
   multiple species in one image and duplicate detection rows.
2. Tests distinguish covered zeros from unknown hours and reject invalid coverage.
3. Fixed-offset and civil camera times join correctly; daylight-saving transitions
   still produce exactly 24 elapsed hourly targets.
4. Changing post-origin sightings cannot change features or normalisation.
5. Publication-delay and ingestion-delay tests exclude unavailable data. Training
   targets never overlap held-out target periods.
6. Mocked weather tests cover caching, absent variables, stale or unavailable runs,
   failures, and precipitation alignment at the horizon endpoints.
7. Baseline, sparse/all-zero support, unseen locations, nonnegative model outputs,
   zero references, and index clipping have regression tests.
8. A synthetic offline CLI audit/backtest/forecast exercise produces readable CSV
   outputs without network access or changes to source data.
9. Run repository gates: lockfile check, compileall, and full unittest discovery.
   Report unrelated pre-existing failures without changing their owning code.

Real-data validation requires operator-reviewed coverage, site coordinates, and
timestamp conventions. Until supplied, deliver the working pipeline and synthetic
proof, not a fabricated live forecast or unsupported accuracy claim.

## Approval And Next Step

This document records the proposed contracts, not camera facts. The accompanying
test-first implementation plan includes the approved neural comparison and shared
weather location. Review that plan and agree its execution method before
implementing product code.