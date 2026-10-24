# Wildlife Identification and Analysis

Python tools for identifying wildlife in camera-trap images and analysing activity patterns from labelled detections.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
the Python 3.10 environment from the lockfile:

```bash
uv sync --locked
```

Copy camera exports into `data/<location>/` before organising images.

The easiest way to use this system is through the main interface:

```bash
uv run wildl_id.py
```

The menu provides these actions:

1. Organise images from `data/` into `images/<year>/<location>/`.
2. Label and classify images with MegaDetector and DeepFaune.
3. Generate visualisations and analysis reports.
4. Run all supported steps in sequence.
5. Exit.

## Features

### Image Labelling and Classification

- **Animal Detection**: Uses PyTorch Wildlife MegaDetector V6 for detecting animals, humans, and vehicles
- **Species Classification**: DeepFaune classifier identifies European wildlife species
- **Metadata Extraction**: OCR extracts timestamp and temperature data from camera overlay
- **Lighting Analysis**: Classifies images as day/night using LAB colour space
- Supports incremental processing (only processes new images)
- Configurable to skip incomplete entries

### Visualisations and Analysis

- Activity patterns by hour of day
- Species distribution charts
- Calendar heatmaps
- Location-based statistics
- Day/night activity patterns
- Temperature correlations
- Moon phase analysis

Visualisations are saved to `docs/diagrams/`.

## Manual Usage

If you prefer to run individual components:

### Organise Images

Place camera exports in `data/<location>/`, then run:

```bash
uv run src/labelling/organize_images.py [--dry-run]
```

The script deletes videos and corrupt images, then moves images to
`images/<year>/<location>/image_XXXXXXX.jpg`, numbered by EXIF capture time.
Images without an EXIF timestamp stay in `data/` and are ignored by the other tools.
The labels CSV and annotated images are renamed to match. A mapping of old to new
paths is written to `logs/`.

### Label Images

```bash
uv run src/labelling/label_images.py
```

### Review Machine Labels

Run the browser-based reviewer locally:

```bash
uv run src/labelling/review_labels.py
```

The reviewer opens at `http://127.0.0.1:8080`. It uses original photos when available
and saved annotated images otherwise. Choose a mode, accept a prediction, or select a
label. Drag on the image to add or replace its bounding box; changes save to the CSV.
Use **Undo label** or **Undo box** to revert the latest change of that type.

Use the zoom icons, mouse wheel, or a pinch gesture to inspect the image. Choose
**Pan** to move the zoomed image or **Box** to edit its bounding box. **Focus subject**
toggles between the whole image and its saved bounding-box area, with padding.
This preference stays active across images; images without a valid box show in full.
Manual zoom resets for each new image. **Fit image** turns subject focus off.
Press **Z** to toggle subject focus.

Press **Space** to accept the current prediction. Press a displayed shortcut letter to
choose a label. Press **Enter** to focus the custom-label field, then press **Enter**
again to save it. Press **Backspace** to undo the last label.

Shortcuts: `r` roe deer, `w` wild boar, `p` pigeon, `b` badger, `m` marten,
`d` dog, `f` fox, `h` hare, `s` squirrel, `j` jay, `o` raccoon, `c` crow,
`a` human, `n` none, `u` unknown, `l` owl, `t` cat, `v` vehicle.

Each choice is saved immediately in a new `manual_label` column in the same CSV.
The original `class` column remains unchanged: an empty manual label means unchecked,
a matching label means confirmed, and a different label means corrected. By default,
the reviewer shows unchecked rows. Use the in-window mode dropdown to select **Latest
model incorrect**, **Unseen + errors**, **Seen**, or **All**. The incorrect-prediction
mode uses validation images from the newest timestamped fine-tuning run under `models/`.
A `.csv.before-review.bak` backup is created before the first save. Labelling runs
preserve the manual column and skip reviewed images, even if their metadata is
incomplete. Do not run the reviewer and labelling process simultaneously.

To review another dataset, use `--csv PATH` and optionally `--images DIRECTORY`.
By default, the reviewer uses the MegaDetector V6 yolov10-e and DeepFaune labels under
`labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/`.

### Remove Confirmed-Empty Images

Delete images confirmed as empty (`class` and `manual_label` both `none`):

```bash
uv run src/labelling/remove_empty_images.py --dry-run
uv run src/labelling/remove_empty_images.py
```

The original photos, annotated images, and CSV rows are removed. The CSV is backed up
first; deleted paths are written to the ignored `logs/` directory.

### Generate Visualisations

```bash
uv run src/visualisation/evaluate_labels.py
```

## Configuration

### Image Labelling Configuration

Edit [src/labelling/label_images.py](src/labelling/label_images.py) to configure:

- Model selection (MegaDetectorV5/V6 variants)
- Species classification on/off
- OCR enabled/disabled
- Reprocess incomplete entries
- Save annotated images

### Visualisation Configuration

Edit [src/visualisation/evaluate_labels.py](src/visualisation/evaluate_labels.py) to configure:

- Model to analyse
- Output directory
- Location coordinates for sunrise/sunset calculations

## Data and Licensing

The tracked CSV contains labels and camera metadata; original photographs are not
included. The CSV is licensed under [CC BY 4.0](DATA-LICENCE.md). The MIT licence
applies to the code only; neither licence covers third-party model software or
weights.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and data guidance.
