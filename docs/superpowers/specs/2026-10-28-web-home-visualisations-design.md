# Web Home And Visualisations Design

## Purpose

Add a main menu to the local wildl-id web app and integrate the existing
visualisation workflow. The menu exposes manual review and visualisations only;
other command-line operations remain out of scope for now.

This change extends the approved manual-labelling web feature. Keep the existing
Mac dataset mounts and localhost-only access. Move manual review from `/` to
`/review`; use `/` for the main menu.

## Scope

- A home page at `/` with links to `/review` and `/visualisations`.
- The current manual reviewer at `/review`, with its existing editing behavior.
- A visualisation page at `/visualisations` that generates charts from the
  default CSV, including its `manual_label` overrides.
- Run visualization generation without blocking the NiceGUI event loop.
- Show a spinner while generating, then a summary and a selectable chart view.
- Keep generated results in bounded container temporary storage, not in the
  host repository.

Email downloads, image organization, ML labeling/classification, the full
pipeline, a progress percentage, and public or LAN access remain out of scope.

## Architecture

The home page is a small NiceGUI navigation menu, not a marketing page. The two
commands are **Manual review** and **Visualisations**. Preserve the existing
reviewer UI at `/review`.

The visualisation page calls `src/visualisation/evaluate_labels.py` with the
mounted default CSV and an explicit temporary output directory. Extend
`main()` with optional `csv_path` and `output_dir` parameters while retaining
its existing defaults: CLI use without arguments continues reading the default
CSV and writing `docs/diagrams/`. The web app never uses that persistent output
directory.

Add only this visualiser module and its runtime dependencies to the strict
Docker build allowlist, along with `web/visualisations.py`. Mount web modules as
the `/app/web` namespace package and start Uvicorn with `web.app:app`. Add the
dependencies it imports directly: Astral, Ephem, NumPy, pandas, pytz, SciPy,
and seaborn. Select SciPy `>=1.14,<2.0` on Python before 3.14 and `>=1.16,<2.0`
on Python 3.14. Matplotlib is already in the isolated web project. Do not add ML
dependencies or copy visualisation source data into the image.

Run the synchronous visualiser through NiceGUI's `run.io_bound` so chart
calculation and file output do not block UI requests. Disable the Generate
button while a run is active and show a spinner/status message. Prevent
concurrent generation because the visualiser uses Matplotlib's process-global
state. Do not add percentage progress; the current visualiser exposes no
reliable progress events.

Write each run to a unique directory under `/tmp/wildl-id-visualisations/`.
Keep the current successful result available while a new run is in progress. On
success, replace the current result set and remove its old temporary directory;
on failure, remove only the failed run and retain the prior results. Raise the
Compose `/tmp` limit from 16 MiB to 64 MiB. This bounds temporary output without
adding another host mount. Results disappear when the container stops.

After success, show `summary_statistics.txt` and a list of generated SVGs. Load
one selected chart at a time to avoid transferring the full set of large SVGs
when the page opens. Show a clear error if input validation or generation fails;
keep detailed exceptions in container logs. Do not alter host `docs/diagrams/`.

## Behaviour And Failure Handling

The home page has no workflow execution side effects. Clicking Manual review
opens the existing reviewer at `/review`. Clicking Visualisations opens its
separate page; generation starts only when the user selects Generate.

The existing visualiser validates CSV existence, parseability, required
columns, and non-empty input before clearing its output directory. With the web
output directory isolated under `/tmp`, failures must not modify the host CSV,
labels, images, models, or `docs/diagrams/`. A false result or raised exception
shows an error state and retains the most recent successful result set.

The app remains bound to `127.0.0.1:8080`. There is no login; LAN and internet
access stay disabled. No operation added by this spec needs credentials or
writable host data beyond the already-approved CSV mount.

## Verification And Acceptance

1. Test that `/` contains links to Manual review and Visualisations; `/review`
   still builds the reviewer and hides Remove empty; `/visualisations` loads
   without running generation automatically.
2. Test the visualiser with temporary CSV and output paths. Prove successful
   output creation, manual-label override use, and that host `docs/diagrams/`
   remains untouched.
3. Test missing and invalid CSV results: show an error, preserve prior results,
   and create no host output files.
4. Test that generation uses a worker thread and the UI remains responsive while
   a job runs; verify a second concurrent run is rejected or disabled.
5. Test result selection loads only the selected SVG and summary text is shown.
6. Run existing visualisation, reviewer, web, and full repository tests.
7. Verify Docker includes only the visualiser code and declared dependencies,
   validates Compose with the 64 MiB `/tmp`, builds, and serves local routes.
8. Confirm no LAN/internet binding, data/model copy, or new host write mount was
   introduced.

Do not claim Pi hardware or architecture validation without testing on the Pi.

## Future Compatibility

Other CLI actions can be designed and added as separate workflows later. This
menu does not imply they are safe or supported in the web container.
