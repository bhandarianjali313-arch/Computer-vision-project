# Industrial Defect Detection — ML/CV Module

This directory contains the machine-learning and computer-vision pipeline
for the Real-Time Industrial Defect Detection System.

The ML component detects six types of steel-surface defects from NEU-DET
images and provides image, video, ONNX, deployment, and API-ready inference
support.

## Defect Classes

The project uses the following fixed class mapping:

| ID | Class |
|---:|---|
| 0 | crazing |
| 1 | inclusion |
| 2 | patches |
| 3 | pitted_surface |
| 4 | rolled-in_scale |
| 5 | scratches |

## Dataset

The project uses the NEU Metal Surface Defects dataset.

The inspected local dataset contained:

- 1,800 images
- 1,800 Pascal VOC XML annotation files
- 4,189 annotated objects
- six target defect classes
- images with a resolution of 200 × 200 pixels

The reproducible dataset split is:

| Split | Images |
|---|---:|
| Train | 1,260 |
| Validation | 360 |
| Test | 180 |

Dataset files are intentionally excluded from Git.

Expected local location:

```text
data/raw/NEU-DET/