# UI Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the three requested UI corrections without changing page data selection or navigation behavior.

**Architecture:** Reuse the common chevron navigation toggle on Review, display the existing `Visit.image_count` in the landing-page table, and correct the camera-map column label. The misclassified queue retains its existing false-negative and unresolved-validation behavior.

**Tech Stack:** Python 3.14, NiceGUI, `unittest`.

**Spec:** [2026-11-04-exif-weather-metadata-and-ui-design.md](../specs/2026-11-04-exif-weather-metadata-and-ui-design.md)

## Global Constraints

- “The Review navigation toggle uses the same Material chevron icon and behavior as every other page.”
- “Each row in the landing-page Longest visits table shows the number of images in that visit.”
- “The camera-map column header reads `Sightings in the last 24h`.”
- “`misclassified` continues to mean a pending model error.”
- “A model `class` of `none` is an explicit no-accepted-detection result from the detector.”
- “When a human label identifies an animal for that image, this is a false negative and should remain in the review queue.”
- “Do not overwrite or discard the unrelated local edits in `labels/labels_deepfaune_finetuned_20261103-091800_018b8ca6/predictions.csv` or `tests/test_web_app.py`.”

## Review Focus

- Review toggle collapsed and expanded states: assert Material chevron icons and unchanged aria labels in `test_review_page_shows_two_modes_and_one_emoji_cleanup`.
- Visit rows with varying image counts: assert the table exposes `visit.image_count` in `test_home_shows_three_longest_positive_visits_and_refreshes`.
- Camera-map wording: assert the column header text in `test_map_seat_sightings_match_slider_and_refresh_without_fetching_wind`.
- A human animal label paired with model `none`: retain the false-negative behavior in the existing `test_misclassified_needs_a_prediction` and review-label misclassification coverage; do not make a UI-only filter change.
- A validation record marked incorrect without a predicted label: retain it as an unresolved queue item using the existing `needs_model_review` tests.

---

### Task 1: Align Review navigation and landing-page labels

**Files:**
- Modify: `web/app.py`
- Modify: `web/gallery.py`
- Modify: `web/camera_map.py`
- Test: `tests/test_web_app.py`

**Interfaces:**
- Consumes: `Visit.image_count` from `src.visits.Visit`.
- Produces: the same navigation-toggle `icon` property on Review as on other pages; a Longest visits row field named `image_count`; the exact column heading `Sightings in the last 24h`.

- [ ] **Step 1: Write failing UI assertions**

Update `test_review_page_shows_two_modes_and_one_emoji_cleanup` to expect `chevron_right` initially and `chevron_left` when expanded, instead of emoji button text. Update `test_home_shows_three_longest_positive_visits_and_refreshes` to assert visit image counts. Add an exact assertion for `Sightings in the last 24h` and absence of the old parenthesized label to the map-ranking UI test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --project web --locked --extra test --python 3.14 python -m unittest tests.test_web_app.WebAppTests.test_review_page_shows_two_modes_and_one_emoji_cleanup tests.test_web_app.WebAppTests.test_home_shows_three_longest_positive_visits_and_refreshes tests.test_web_app.WebAppTests.test_map_seat_sightings_match_slider_and_refresh_without_fetching_wind`

Expected: FAIL only on the newly asserted toggle, count, and heading behavior.

- [ ] **Step 3: Implement the UI changes**

In `web/app.py`, remove the Review-only emoji/text branch so the button always uses the existing Material chevron icon branch. In `web/gallery.py`, include `visit.image_count` in each visit record and table columns. In `web/camera_map.py`, replace the header with `Sightings in the last 24h`.

- [ ] **Step 4: Run focused and complete web tests**

Run the focused command from Step 2, then:
`uv run --project web --locked --extra test --python 3.14 python -m unittest discover -s tests -p 'test_web*.py'`

Expected: all selected tests and all web tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/app.py web/gallery.py web/camera_map.py tests/test_web_app.py
git commit -m "fix(web): align navigation and visit summaries"
```
