# Local Web App

The home page at <http://127.0.0.1:8080> centres the logo and the summary
"Explore wildlife sightings, review detections, browse camera images, and view
activity across your cameras." above recent sightings. The side navigation on
the other pages links to Review, Gallery, Visualisations, Predictions, and
Camera map. It starts collapsed to icons; hover for a page name and summary, or
expand it to show the navigation labels.

The home page also shows a mini gallery of sightings captured in the last
24 hours, newest first. It shows only the relevant species listed in
`config/relevant_species.json` (roe deer, wild boar, badger, marten, fox, hare,
and raccoon), using manual labels when available to select species.
Compact species buttons allow multiple selections; **All** resets the filter.
Button counts and the short summary
count each photo once per species, not each bounding box. Selections persist
through minute refreshes. There is no search field, guide or examples on home.
Thumbnails and enlarged photos retain their original aspect ratios.
Toggling bounding boxes changes only the overlay, not the carousel frame. Boxes
are yellow before review, green when a human label agrees with the prediction,
and red when it disagrees. Gallery pagination uses bounded page links, ellipses,
and first/last controls for large datasets.
Photos that are unlabelled, empty, unknown, or of other species stay off the
home page; browse them in the Gallery. Capture times come only from
`timestamp_exif`, or from the original image's EXIF metadata when that column
has no valid time. OCR timestamps, camera filenames and file modification
times are not used. Thumbnails show the manual
label, or the automatic label when there is none. Boxes appear when present.
Click an image to
enlarge it. The viewer has previous/next controls and an image counter;
left/right arrow keys navigate and Escape closes it. Navigation covers all
matching images, including other thumbnail pages. An open viewer keeps its
image list stable during refresh; reopen it to browse the latest results.
The gallery reloads the CSV and image availability every minute
without reloading the page; older sightings drop out automatically. Results
are paginated when there are more than 48 images. Discovery is read-only and
never creates or changes labels.

Below the recent images, **Longest sightings** lists the longest positive
observed span for each animal species in the last 24 hours, with its site and
first/last detection times. It uses the same independent visits and EXIF-only
times as the gallery. Only detections inside the rolling window contribute to
the span, even when a visit began earlier. Single-photo and zero-span visits,
people and vehicles are omitted; an empty summary stays hidden. Ties select
the most recent visit. The table refreshes with the gallery and does not change
when thumbnail species filters change. Observed span is not proven dwell time.

Immediate species notifications show one emoji in the title and only the capture
time and date in the body. They omit ntfy emoji tags to avoid duplicate icons.
The landing page displays older species alerts in the same format without
rewriting notification history. Daily summaries retain their full text.
The notification feed shows every notification sent within the last 24 hours,
not only the latest five. Status shows the same 24-hour total separately from
the latest polling run, which can legitimately send zero notifications.
New image alerts retain a copy of the ntfy attachment under
`logs/notification_images/`; the landing card shows that image with the same
title and body. Text-only alerts and summaries have no image. Existing alerts
show an image only when site, species and capture minute identify one original
photo unambiguously. Missing or ambiguous historical images stay absent rather
than showing a guessed photo. The notification CSV keeps its original columns.

### Independent Visits

Both galleries frame images from the same visit in the same colour, without
moving them out of capture-time order. Frames use muted woodland colours;
hovering over a thumbnail brightens that card and raises its shadow. Every
displayed image in the same visit also gains a 3 px outline matching its frame
colour, making those frames appear 6 px thick without moving or resizing cards.
Leaving the thumbnail clears the visit outlines. Ungrouped photos retain only
the individual hover effect. Matching uses visit IDs, not shared palette colours.
Each visit belongs to one site and one
effective species: the manual label when present, otherwise the prediction.
A gap of **at least 30 minutes** since the previous detection starts another
visit. Visits can span midnight and last longer than 30 minutes when detections
continue. Other species do not end a visit. Colours repeat across the dataset;
hover over a frame for the visit's species, site, interval and photo count.
Frames remain consistent across filters, pagination and bounding-box toggles.
Photo counts still count photos, not visits.

Grouping uses the full dataset before display filters and does not change or
sort the source CSVs. Missing sites, missing or invalid EXIF times, and empty or
unknown species have no visit frame. Humans and vehicles form their own groups,
separate from wildlife. Repeated rows for the same site and image count once
within a visit; separate photos at the same instant remain separate members.
Rows that disagree on species or capture time for the same image have no visit
until the conflict is resolved. This is a one-label-per-image view, not support
for multiple species detections in one photo.

Naive EXIF times use `Europe/Berlin`; explicit offsets determine elapsed time.
Naive times within ambiguous or skipped daylight-saving hours remain ungrouped.
The observed span is the last detection minus the first, not proven dwell time.
A single-photo visit has an observed span of zero. New images and label changes
recompute visits; visit IDs identify the current derived dataset, not immutable
records. Predictions retain the existing acceptance rules, so incorrect labels
can still create false visits.

The app binds only to localhost; it is not available from the
LAN or internet. There is no login, so other local processes may access it.

## Requirements

- macOS with OrbStack or Docker Desktop and Docker Compose.
- Raspberry Pi 5 with 64-bit Raspberry Pi OS and Docker Compose for later Pi use.

Build the image on each host's native architecture. Do not copy an image built
for one architecture to another host.

## Dataset Access

The app uses the default dataset:

- Human annotations: `labels/human_labels.csv` (location, filename, box, manual label, reviewed model)
- Image metadata: `labels/image_metadata.csv`
- Automatic predictions: `labels/labels_<latest-completed-run>/predictions.csv`
- Annotated images: `labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/images/`
- Original images: `images/`
- Validation predictions: `models/<run>/training_data/labels.csv` (`predicted_label`)

The container can write human annotations and their `.before-review.bak` backup.
Readers join human annotations with metadata and the latest completed model's
predictions; missing automatic predictions never fall back to an older model.
Original and annotated images are writable so the Review page's **Remove empty**
action can run after explicit confirmation. It asks whether to remove only
images with no detection (human and model both `none`) or also false detections
(human `none`, model an animal or no prediction yet), backs up the annotation CSV, logs deleted paths under the writable `logs/`
mount, and records removals in `labels/deleted_images.csv` so old predictions
do not reappear when inference refreshes image metadata.
Model files remain read-only. The app does not mount `data/`, credentials, or the
Docker socket. Browser review offers only **Unseen**, **Latest model incorrect**, and
**Latest model correct**.

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

Choose **Visualisations** in the side navigation. Each visit generates a fresh
report automatically; a spinner remains visible while it runs. When it finishes,
read the summary and select one chart at a time. Filter-based plots remain
available for targeted comparisons.

Under **Filter-based plots**, choose **Individual species timelines (visits)**
for independent visits instead of the existing image-count timeline. Each visit
counts once at its start time. Date and other filters apply to that first photo,
so a visit beginning before the selected interval does not become a new visit
inside it. Capture times throughout these reports use EXIF only.

Visit charts offer **Visits CSV** and **Member images CSV** downloads. The first
contains site, species, UTC start/end, observed span in seconds, image count and
the 30-minute threshold. The second links visit IDs to all member image keys,
including members outside the chosen interval. Only selected visits are exported.
Neither visit counts nor observed spans estimate population size or camera uptime.

Charts and summaries live only in the container's 64 MiB `/tmp` filesystem.
They disappear when the container stops or restarts. A failed rerun keeps the
previous successful results. The visualiser never writes to host
`docs/diagrams/`.

## Camera Map

The home page places a compact map directly below **Best seats now**. Its species
emojis always cover the last 24 hours, and its wind arcs always cover the next
five forecast hours. This view has no day or wind-window controls. The map and
the top-three seat ranking use the same wind result. Sightings refresh every
minute and wind refreshes every 15 minutes.

Open `/map` for camera locations on Esri World Imagery satellite tiles.
Permanent site labels include one emoji per relevant species seen within the
selected window. The **Sightings** slider selects 0–90 rolling days, starting at
7 days. At 0, all species emojis are hidden; site names and camera markers remain.
Sightings use `timestamp_exif` only and prefer manual species labels;
empty, unknown and non-relevant species do not appear. Future times are excluded.
Changing days updates the icons without requesting wind again. Sightings refresh
every minute, independently of wind; missing label data or a wind outage does
not remove camera markers. Hover over each emoji for its species name.

The home best-seat list and the map's seat table repeat the same species icons.
Home uses the last 24 hours; the map table follows the days slider, including
zero to hide sightings. Both lists update with sighting refreshes without
requesting wind again.

Fill `latitude` and `longitude` (decimal degrees, WGS84) per camera in
`config/camera_locations.json`; cameras left `null` are listed as missing.
The read-only `config/` mount picks up edits on page reload, without a rebuild.
Direct Python launches may override the path with `WILDL_ID_CAMERA_LOCATIONS`.

The **Hours** dropdown (1–10, default 3) sets the window length, and the slider
selects its start within the next 24 hours. Each camera shows a dashed arc,
drawn like the view sector, covering every direction the wind blows towards
during the window (±10°) in grey; its radius reflects the maximum speed. The
list below gives each hour's speed, source direction and gusts.
Forecasts are cached for 15 minutes. The map needs internet access for tiles
and wind.

Optional `view_direction` (compass degrees) and `view_angle` (opening, degrees)
draw each camera's view sector. The sector is red when the wind arc overlaps it,
yellow when the smallest angular gap is up to 30°, and green when it is larger.
The list ranks cameras for the selected window by that gap, smallest first.

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
An empty smart filter shows all images. The first example, **Relevant species
only**, limits results to the species in `config/relevant_species.json`.
This phrase can be combined with dates, locations, and other filter conditions.
Bounding boxes are shown by default.
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
20”). **Misclassified** and the review page's **Latest model incorrect** use the
same pending-error rule: current inference mismatches and unresolved saved
validation errors for the latest model, excluding images marked reviewed for
that model. Confirming a label removes the image from both views after refresh;
undo restores it. A new model starts a fresh queue.

Hour filters use an exact rolling window ending now, shared with the home
gallery. They exclude undated and future images and do not limit the image
count. “Latest 24” still selects the newest 24 images regardless of age.

On a Pi, `localhost` means the Pi; open the app in a browser on the Pi itself.
Pi hardware and cross-architecture execution have not been verified.
