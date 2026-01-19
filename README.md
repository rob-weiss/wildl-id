# Wildlife identification and analysis

As a conservationist, hunter, and software developer I've been curious about what insights can be gained from the images taken by wildlife cameras in our local hunting area. This repo contains a collection of `python` tools for wildlife identification and statistical analysis. It is work in progress and by no means complete. Please feel free to contact the authors in case of questions or suggestions.

## Quick Start

The easiest way to use this system is through the main interface:

```bash
python main.py
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
python main.py
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
