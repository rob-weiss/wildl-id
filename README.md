# Wildlife identification and analysis

As a conservationist, hunter, and software developer I've been curious about what insights can be gained from the images taken by wildlife cameras in our local hunting area. This repo contains a collection of `python` tools for wildlife identification and statistical analysis. It is work in progress and by no means complete. Please feel free to contact the authors in case of questions or suggestions.

## Quick Start

The easiest way to use this system is through the main interface:

```bash
python wildl_id.py
```

This will present an interactive menu where you can:

1. **Download new images** from your camera gallery (ZEISS Secacam)
2. **Label/classify images** using MegaDetector and DeepFaune species classifier
3. **Generate visualizations** and analysis reports
4. **Run all steps** in sequence (complete pipeline)

### Usage

Simply run the main script and follow the prompts:

```bash
cd wildl-id
python wildl_id.py
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
python src/labelling/download_images.py
```

### Label Images

```bash
python src/labelling/label_images.py
```

### Review Machine Labels

Run the lightweight reviewer using the existing Matplotlib and Pillow dependencies:

```bash
conda run --no-capture-output -n wildlife python src/labelling/review_labels.py
```

The reviewer displays images one at a time, with the machine label above the CSV
bounding box. It uses original photos when available so the box coordinates align,
and falls back to saved annotated images otherwise. Click **Correct** or press **Space**
to accept it. Click a species
button or press its displayed letter to correct it. Click **Other label** or press
**Enter**, type a custom label, then press **Enter** to save. **Backspace** or **Undo**
revisits the last choice in this session. **Escape** or **Close** exits.

Shortcuts: `r` roe deer, `w` wild boar, `p` pigeon, `b` badger, `m` marten,
`d` dog, `f` fox, `h` hare, `s` squirrel, `j` jay, `o` racoon, `c` crow,
`a` human, `n` none, `u` unknown, `l` owl, `t` cat, `v` vehicle.
The exceptions avoid overlapping first letters.

Each choice is saved immediately in a new `manual_label` column in the same CSV.
The original `class` column remains unchanged: an empty manual label means unchecked,
a matching label means confirmed, and a different label means corrected. Reopening
automatically skips checked rows. A `.csv.before-review.bak` backup is created before
the first save. Machine-labelling runs preserve the manual column and skip reviewed
images, even if their metadata is incomplete. Do not run both tools simultaneously.

To review another dataset, use `--csv PATH` and optionally `--images DIRECTORY`.
By default, the reviewer uses the MegaDetector V6 yolov10-e classified CSV and its
adjacent annotated images directory.

### Generate Visualizations

```bash
python src/visualisation/evaluate_labels.py
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
