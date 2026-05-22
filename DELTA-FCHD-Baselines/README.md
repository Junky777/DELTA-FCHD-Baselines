## 📂 Repository Structure & Workflow

This repository provides the complete, end-to-end pipeline used in the DELTA dataset paper, from raw data privacy protection to advanced deep learning baselines.

### 1. De-identification Pipeline (`1_deidentification_pipeline/`)
Implements the rigorous two-stage "human-in-the-loop" privacy protection mechanism:
- `pixel_wash.py` & `roi_memory.json`: Resolution-aware interactive ROI cropping to remove peripheral UI elements.
- `ocr_last.py`: The final pipeline executing spatial cropping and conservative OCR-based text redaction (using EasyOCR) synchronously.
- `extract.py` & `refix.py`: Scripts used for manual Quality Assurance (QA) and blackout refinement.

### 2. Dataset Preparation (`2_dataset_preparation/`)
- `data_split.py`: Strictly partitions the curated dataset into Train (70%), Val (10%), and Test (20%) at the **patient-level**, employing stratified sampling to maintain the real-world long-tail distribution of the 12 FCHD subtypes.

### 3. Data Quality Validation (`3_plane_classification/`)
Contains the PyTorch training scripts to validate the visual discriminability of the 5 ISUOG standard anatomical planes:
- `resnet50_train.py` (ResNet-50)
- `train_densenet.py` (DenseNet-121 - Best Performance)
- `train_vit.py` (ViT-B/16)

### 4. Image-based Binary Screening (`4_image_binary_screening/`)
Baselines for 2D multi-view spatial feature fusion (Normal vs. Abnormal). Includes Classic MVCNN (View-Pooling) and Attention-based MIL (AB-MIL).

### 5. Video-based Binary Screening (`5_video_binary_screening/`)
State-of-the-art 3D spatio-temporal architectures for continuous sweeping videos. Includes R(2+1)D, SlowFast Networks, and Video Swin Transformer (Swin3D).