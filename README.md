# Wildlife identification and analysis

As a conservationist, hunter, and software developer I've been curious about what insights can be gained from the images taken by wildlife cameras in our local hunting area. This repo contains a collection of `python` tools for wildlife identification and statistical analysis. It is work in progress and by no means complete. Please feel free to contact the authors in case of questions or suggestions.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
the Python 3.10 environment from the lockfile:

```bash
uv sync --locked
```

The easiest way to use this system is through the main interface:

```bash
uv run wildl_id.py
```

This will present an interactive menu where you can:

1. **Download new images** from your camera gallery (ZEISS Secacam)
2. **Organize new images** from `data/` into `images/<year>/<location>/`
3. **Label/classify images** using MegaDetector and DeepFaune species classifier
4. **Generate visualizations** and analysis reports
5. **Run all steps** in sequence (complete pipeline)

### Usage

Simply run the main script and follow the prompts:

```bash
cd wildl-id
uv run wildl_id.py
```

The interface will guide you through each step with clear instructions and confirmations.

## Features

### 1. Image Download

- Downloads images from ZEISS Secacam gallery carousel
- Works with Safari browser automation
- Organizes images by location and timestamp

### 2. Image Labeling & Classification

- **Animal Detection**: Uses PyTorch Wildlife MegaDetector V6 for detecting animals, humans, and vehicles
- **Species Classification**: DeepFaune classifier identifies European wildlife species
- **Metadata Extraction**: OCR extracts timestamp and temperature data from camera overlay
- **Lighting Analysis**: Classifies images as day/night using LAB color space
- Supports incremental processing (only processes new images)
- Configurable to skip incomplete entries

### 3. Visualizations & Analysis

- Activity patterns by hour of day
- Species distribution charts
- Calendar heatmaps
- Location-based statistics
- Day/night activity patterns
- Temperature correlations
- Moon phase analysis

All visualizations are saved to `docs/diagrams/` for easy access.

## Manual Usage

If you prefer to run individual components:

### Download Images

```bash
uv run src/labelling/download_images.py
```

### Organize Images

Put new images into `data/<location>/`, then run:

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
and saved annotated images otherwise. Choose a mode, accept the prediction, or select
a label. Drag on the image to add or replace its bounding box; changes save to the CSV.
Use **Undo label** or **Undo box** to revert the last change of that kind.

Press **Space** to accept the current prediction. Press a displayed shortcut letter to
choose its label. Press **Enter** to focus the custom-label field; press **Enter** there
to save it. Press **Backspace** to undo the last label.

Shortcuts: `r` roe deer, `w` wild boar, `p` pigeon, `b` badger, `m` marten,
`d` dog, `f` fox, `h` hare, `s` squirrel, `j` jay, `o` racoon, `c` crow,
`a` human, `n` none, `u` unknown, `l` owl, `t` cat, `v` vehicle.
The exceptions avoid overlapping first letters.

Each choice is saved immediately in a new `manual_label` column in the same CSV.
The original `class` column remains unchanged: an empty manual label means unchecked,
a matching label means confirmed, and a different label means corrected. By default,
the reviewer shows unchecked rows. Use the in-window mode dropdown to select **Latest
model incorrect**, **Unseen + errors**, **Seen**, or **All**. The incorrect-prediction
mode uses validation images from the newest timestamped fine-tuning run under `models/`.
A `.csv.before-review.bak` backup is created before the first save. Machine-labelling
runs preserve the manual column and skip reviewed images, even if their metadata is
incomplete. Do not run both tools simultaneously.

To review another dataset, use `--csv PATH` and optionally `--images DIRECTORY`.
By default, the reviewer uses the MegaDetector V6 yolov10-e and DeepFaune labels under
`labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/`.

### Generate Visualizations

```bash
uv run src/visualisation/evaluate_labels.py
```

## Configuration

### Image Labeling Configuration

Edit [src/labelling/label_images.py](src/labelling/label_images.py) to configure:

- Model selection (MegaDetectorV5/V6 variants)
- Species classification on/off
- OCR enabled/disabled
- Reprocess incomplete entries
- Save annotated images

### Visualization Configuration

Edit [src/visualisation/evaluate_labels.py](src/visualisation/evaluate_labels.py) to configure:

- Model to analyze
- Output directory
- Location coordinates for sunrise/sunset calculations
