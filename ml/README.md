# Industrial Defect Detection - ML Module

This module contains the machine learning and computer vision pipeline
for the Real-Time Industrial Defect Detection System.

## Objective

The goal of the ML pipeline is to detect and classify surface defects
on industrial metal components using object detection.

## Dataset

The project uses the NEU Metal Surface Defects dataset.

The target defect classes are:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## ML Pipeline

The planned pipeline includes:

Dataset
→ annotation preprocessing
→ dataset validation
→ augmentation
→ YOLOv8 training
→ model evaluation
→ error analysis
→ model optimization
→ real-time inference
→ edge deployment

## Technologies

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Albumentations
- NumPy
- Scikit-learn

## Current Status

ML development environment initialized and validated.