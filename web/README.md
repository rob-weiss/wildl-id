# Manual Labelling

The web app reuses the existing wildlife label reviewer. It writes accepted
labels and backups to the default CSV on the host. It does not run inference or
copy images and model files into the Docker image.

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

The app binds only to `127.0.0.1`; it is not available from the LAN or internet.
There is no login, so other local processes may access it. On a Pi, `localhost`
means the Pi; open the app in a browser on the Pi itself. Pi hardware and
cross-architecture execution have not been verified.
