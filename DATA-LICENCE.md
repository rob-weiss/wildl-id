# Dataset Licence and Provenance

The tracked labelled CSV at
`labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/detection_results_MegaDetectorV6_MDV6-yolov10-e_classified.csv`
is licensed under the [Creative Commons Attribution 4.0 International licence
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Use this attribution when reusing the CSV:

> Rob Weiss, *Wildlife Identification and Analysis camera-trap detections*,
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
> [project repository](https://github.com/rob-weiss/wildl-id). Changes made.

Replace “Changes made” with a short description if you adapt the data. This
licence applies to the CSV and its annotations and metadata only. It does not
cover third-party software or model weights. Original camera photographs are
not included.

The CSV records camera-trap detections and related metadata, including location
IDs, timestamps, species labels, bounding boxes, model confidence, lighting,
temperature, and human review labels. Detections were produced with MegaDetector
V6 (`MDV6-yolov10-e`) and DeepFaune; OCR and human review provide additional
metadata and corrections.

Without the source photographs, users cannot independently inspect the images
behind individual detections. The repository does not provide further collection
protocols or site-level provenance.
