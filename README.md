# Car Damage Detection with YOLOv8‑seg

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Live%20Demo-Space-blue)]([https://huggingface.co/spaces/YOUR_USERNAME/car-damage-detector](https://huggingface.co/spaces/DeepLens/car-damage-detector?logs=build))
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Detect six types of car damage** – dent, scratch, crack, glass shatter, tire flat, lamp broken – using a YOLOv8 instance segmentation model. This repository contains the full pipeline: data preparation, training, evaluation, and a Gradio web app.

![Sample prediction](assets/sample_prediction.jpg)  
*Example output: model detecting multiple damages on a car.*

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Running the Demo Locally](#running-the-demo-locally)
- [Deployment on Hugging Face](#deployment-on-hugging-face)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This project implements an end‑to‑end pipeline for car damage detection and segmentation. The model is built on **YOLOv8‑seg** (instance segmentation) and fine‑tuned on the [CarDD](https://cardd-ustc.github.io/) dataset. It can localise damages with pixel‑level masks and estimate confidence scores.

**Key features:**
- Detects six damage types: dent, scratch, crack, glass shatter, tire flat, lamp broken.
- Provides bounding boxes and segmentation masks.
- Includes a **Gradio web app** for interactive testing (live on Hugging Face Spaces).
- Supports both **image** and **video** input.
- Training code with checkpoint resuming, class‑weighted focal loss, and automatic dataset conversion.

---

## Dataset

We use the **CarDD** (Car Damage Detection) dataset [1], which contains over 9,000 high‑resolution images with pixel‑level annotations for six damage categories. The dataset is split into train, validation, and test sets. Below are key statistics from our exploratory analysis.

### Class Distribution
![Class distribution](assets/class_distribution.png)  
*The dataset is imbalanced: “scratch” and “dent” are the most frequent; “tire flat” and “crack” are rarer.*

### Bounding Box Size Distribution
![BBox size distribution](assets/bbox_size_dist.png)  
*Most damages are small to medium (width/height < 300 px), but there is a long tail of larger damages.*

### Aspect Ratio Distribution
![Aspect ratio distribution](assets/aspect_ratio_dist.png)  
*Damages are roughly square, with a slight bias towards horizontal elongation (scratches, cracks).*

### Image Size Distribution
![Image size distribution](assets/image_size_dist.png)  
*Original images are around 700–950 px; we resize to 640 px during training.*

---

## Model Architecture

We employ **YOLOv8‑seg** (nano variant) – a state‑of‑the‑art single‑stage instance segmentation model. It combines object detection and segmentation in a unified architecture, making it both fast and accurate.

**Key characteristics:**
- **Backbone**: CSPDarknet with multi‑scale feature extraction.
- **Head**: Decoupled detection and segmentation heads.
- **Loss**: Focal loss for classification, CIoU for bounding boxes, and dice/BCE for masks.
- **Input size**: 640×640 pixels.
- **Pretrained weights**: Trained on COCO for 80 classes.

We fine‑tune the model on the CarDD dataset, adapting the final layer to 6 classes and using **class‑weighted focal loss** to give more importance to rare classes (especially “crack” and “tire flat”).

---

## Results

After training for 50 epochs (20 epochs with frozen backbone + 30 epochs full fine‑tuning) on a Tesla T4 GPU, we obtained the following metrics on the test set.

### Overall Performance
| Metric               | Value |
|----------------------|-------|
| Box mAP@0.5          | 0.72  |
| Box mAP@0.5:0.95     | 0.53  |
| Mask mAP@0.5         | 0.69  |
| Mask mAP@0.5:0.95    | 0.51  |

### Per‑Class mAP@0.5 (Box)
| Class          | mAP@0.5 |
|----------------|---------|
| Dent           | 0.68    |
| Scratch        | 0.62    |
| Crack          | 0.35    |
| Glass shatter  | 0.94    |
| Tire flat      | 0.76    |
| Lamp broken    | 0.88    |

### Training Curves
![Training losses](assets/training_losses.png)  
*Box, segmentation, and classification losses decrease steadily.*

### Validation Metrics
![Validation mAP](assets/validation_mAP.png)  
*Box and mask mAP improve throughout training.*

### Confusion Matrix
![Confusion matrix](assets/confusion_matrix.png)  
*The model confuses “dent” and “scratch” occasionally; “crack” has the highest misclassification rate.*

### Precision‑Recall Curves
![PR curves](assets/PR_curves.png)  
*Glass shatter and lamp broken achieve near‑perfect precision‑recall trade‑off.*

### Sample Predictions
| Input | Output |
|-------|--------|
| ![input dent](assets/input_dent.jpg) | ![output dent](assets/output_dent.jpg) |
| ![input scratch](assets/input_scratch.jpg) | ![output scratch](assets/output_scratch.jpg) |

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/car-damage-detection.git
   cd car-damage-detection
