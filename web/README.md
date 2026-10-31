# Local Web App

The home page at <http://127.0.0.1:8080> links to manual review, the gallery,
visualisations, predictions, and the camera map, with a brief description of each. Every page includes a
navigation rail that starts collapsed to icons. Hover for page names, or expand
the rail to show the full logo and navigation labels.

The home page also shows a mini gallery of sightings captured in the last
24 hours, newest first. It includes roe deer, wild boar, badger, marten, fox,
hare, and raccoon, using manual labels when available to select species.
Compact species buttons allow multiple selections; **All** resets the filter
and **Unlabelled** selects pending photos. Button counts and the short summary
count each photo once per species, not each bounding box. Selections persist
through minute refreshes. There is no search field, guide or examples on home.
Thumbnails and enlarged photos retain their original aspect ratios.
Unclassified photos are included until their species is known, including
originals not yet present in the labels CSV. Capture times come from the CSV
or the original image's EXIF metadata; supported camera filenames are a
fallback. File modification times are not used. Available manual and automatic
labels are both shown, even without a bounding box. Boxes appear when present,
and photos with no annotations are marked **Unlabelled**. Click an image to
enlarge it. The viewer has previous/next controls and an image counter;
left/right arrow keys navigate and Escape closes it. Navigation covers all
matching images, including other thumbnail pages. An open viewer keeps its
image list stable during refresh; reopen it to browse the latest results.
The gallery reloads the CSV and image availability every minute
without reloading the page; older sightings drop out automatically. Results
are paginated when there are more than 48 images. Discovery is read-only and
never creates or changes labels.

The app binds only to localhost; it is not available from the
LAN or internet. There is no login, so other local processes may access it.

## Requirements

- macOS with OrbStack or Docker Desktop and Docker Compose.
- Raspberry Pi 5 with 64-bit Raspberry Pi OS and Docker Compose for later Pi use.

Build the image on each host's native architecture. Do not copy an image built
for one architecture to another host.

## Dataset Access

The app uses the default dataset:

- Human annotations: `labels/human_labels.csv` (location, filename, box, manual label)
- Image metadata: `labels/image_metadata.csv`
- Automatic predictions: `models/<latest-completed-run>/predictions.csv`
- Annotated images: `labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/images/`
- Original images: `images/`
- Validation predictions: `models/`

The container can write human annotations and their `.before-review.bak` backup.
Readers join human annotations with metadata and the latest completed model's
predictions; missing automatic predictions never fall back to an older model.
Annotated
images, original images, and model files are read-only. The app does not mount
`data/`, logs, credentials, or the Docker socket. It hides the destructive
“Remove empty” action.

## Run Locally

From the repository root, start the container with your Mac user IDs:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose up --build -d
```

Open <http://127.0.0.1:8080>. Check app health at
<http://127.0.0.1:8080/health> and container status with:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose ps
```

View logs with:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose logs -f web
```

Stop and remove the container with:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose down
```

Label edits remain in the host CSV after the container stops. To update the
app, pull the repository and rebuild it:

```bash
git pull --ff-only
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose up --build -d
```

## Visualisations

Choose **Visualisations** from the home page, then select **Generate
visualisations**. A spinner remains visible while the report runs. When it
finishes, read the summary and select one chart at a time.

Charts and summaries live only in the container's 64 MiB `/tmp` filesystem.
They disappear when the container stops or restarts. A failed rerun keeps the
previous successful results. The visualiser never writes to host
`docs/diagrams/`.

## Camera Map

Open `/map` for camera locations on Esri World Imagery satellite tiles.
Fill `latitude` and `longitude` (decimal degrees, WGS84) per camera in
`config/camera_locations.json`; cameras left `null` are listed as missing.
The read-only `config/` mount picks up edits on page reload, without a rebuild.
Direct Python launches may override the path with `WILDL_ID_CAMERA_LOCATIONS`.

The slider selects one of the next 24 hours. Each camera shows an arrow
pointing downwind, its length and colour scaled by the Open-Meteo 10 m wind
speed; the list below gives speed, source direction and gusts. Forecasts are
cached for 15 minutes. The map needs internet access for tiles and wind.

Optional `view_direction` (compass degrees) and `view_angle` (opening, degrees)
draw each camera's view sector. Seats are ranked for the three hours starting
at the slider time by how much of the wind source sector (forecast direction
±45°) lies in the view, so scent drifts away from the viewed area. Sectors are
green (≥67 %), yellow (≥33 %) or red.

## Predictions

Open `/predictions` for the next 24 elapsed hours of hourly activity for the same
seven species as the home gallery. The camera selector changes location; the
segmented control switches between expected images/hour and the training-reference
0-1 activity index. The page refreshes each minute without training a model or
requesting weather. It uses the best-performing count-only Poisson model family
from the exploratory comparison, not the neural/weather candidates.

Install the saved numeric checkpoint as
`models/activity_forecast/poisson/metadata.json`. The existing read-only `models/`
mount makes it available inside the container; model artefacts are not committed
or baked into the image. Direct Python launches may override the path with
`WILDL_ID_FORECAST_MODEL`. Missing or incompatible models show an unavailable state.

Forecasts are exploratory, not presence probabilities. The model's recorded
camera-clock, location and ingestion-delay settings are honoured, and its
training cutoff is checked. The display explicitly reports the last record and
model cutoff. Future image timestamps are excluded, manual labels override
automatic labels, and repeated boxes do not increase an image/species count.

Seven days of context are required. If the latest capture predates the final
completed context hour, the page leaves charts empty by default. **Assume continuous
observation** explicitly permits filling that trailing gap with zero sightings
under the exploratory assumption that the camera and ingestion remained active.
It is off by default and resets for each page visit. Do not enable it for known
camera downtime or incomplete uploads. Relative indices are missing rather than
invented when a positive training reference is unavailable; clipped indices and
baseline fallbacks are identified per species.

## Manual Review

Choose **Manual review** from the home page. The reviewer writes labels and a
backup to the default host CSV. Annotated and original images and models are
read-only inside the container.

## Gallery

Choose **Gallery** from the home page to browse the CSV's images, newest first.
Local originals under `images/` are shown when present; otherwise the app uses
`WILDL_ID_IMAGE_BASE_URL` (the published `images/` prefix). The **Smart filter**
turns a prompt such as “Show only wild boar of the last two weeks” into a CSV
filter inside the app; no prompt leaves the machine and no generated code is
executed. The line under the filter shows how the prompt was interpreted. It
understands labels and aliases (“boar”, “birds”, “deer”), locations, negation
(“except foxes”, “not at Douglasien”), relative and absolute dates (“last 3
months”, “last 24h”, “last 24 hours”, “this year”, “in March 2025”,
“since 2025-06-01”, “in winter”), night
or day, temperatures (“below 0 °C”, “freezing”), labelling status
(“unlabelled”, “misclassified”), and order or count (“oldest first”, “latest
20”).

Hour filters use an exact rolling window ending now, shared with the home
gallery. They exclude undated and future images and do not limit the image
count. “Latest 24” still selects the newest 24 images regardless of age.

On a Pi, `localhost` means the Pi; open the app in a browser on the Pi itself.
Pi hardware and cross-architecture execution have not been verified.
