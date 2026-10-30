# Local Web App

The home page at <http://127.0.0.1:8080> links to manual review, the gallery,
and visualisations, with a brief description of each. Every page includes a
navigation rail that starts collapsed to icons. Hover for page names, or expand
the rail to show the full logo and navigation labels.

The home page also shows a mini gallery of sightings captured in the last
24 hours, newest first. It includes roe deer, wild boar, badger, marten, fox,
hare, and raccoon, using manual labels when available to select species.
Unclassified photos are included until their species is known, including
originals not yet present in the labels CSV. Capture times come from the CSV
or the original image's EXIF metadata; supported camera filenames are a
fallback. File modification times are not used. Available manual and automatic
labels are both shown, even without a bounding box. Boxes appear when present,
and photos with no annotations are marked **Unlabelled**. Click an image to
enlarge it. The gallery reloads the CSV and image availability every minute
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

- CSV: `labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/`
- Annotated images: that directory's `images/` child
- Original images: `images/`
- Validation predictions: `models/`

The container can write the CSV and its `.before-review.bak` backup. Annotated
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
