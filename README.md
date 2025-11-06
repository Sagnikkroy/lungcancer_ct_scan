# Lung Cancer CT Scan Classification

## Project Overview
This project implements a deep learning classifier to detect and classify lung cancer types from CT scan images using PyTorch. The model classifies CT scans into five categories: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal Cases, and Benign Cases.

## Dataset Information
- **Dataset**: CT Scan Images of Lung Cancer
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/mdnafeesimtiaz/ct-scan-images-of-lung-cancer)
- **Classes**: 5
  - Adenocarcinoma
  - Large Cell Carcinoma
  - Squamous Cell Carcinoma
  - Normal Cases
  - Benign Cases
- **Total Images**: 1,535 CT scan images
- **Class Distribution**:
  - Adenocarcinoma: 337 images
  - Large Cell Carcinoma: 187 images
  - Squamous Cell Carcinoma: 260 images
  - Normal Cases: 631 images
  - Benign Cases: 120 images

### Dataset Structure
lung_cancer_dataset/
├── adenocarcinoma/
│ ├── image1.png
│ ├── image2.png
│ └── ...
├── large cell carcinoma/
│ ├── image1.png
│ └── ...
├── squamous cell carcinoma/
│ ├── image1.png
│ └── ...
├── normal cases/
│ ├── image1.png
│ └── ...
└── benign cases/
├── image1.png
└── ...

