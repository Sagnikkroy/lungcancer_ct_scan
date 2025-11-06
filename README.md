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


## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- NVIDIA GPU (optional, for faster training)

### Installation Steps

1. **Extract the submission files**
   ```bash
   # Extract the zip file to your desired location

   Install dependencies
   pip install -r requirements.txt

   Download and prepare the dataset

Download from: https://www.kaggle.com/datasets/mdnafeesimtiaz/ct-scan-images-of-lung-cancer

Extract the dataset and place it in your working directory

Ensure the folder structure matches the expected format

How to Run
Training the Model
bash
python lung_cancer_classifier.py
The script will automatically:

Load and preprocess all CT scan images

Split data into training (80%) and validation (20%) sets

Train a ResNet18-based model with transfer learning

Save the best model as best_lung_cancer_model.pth

Generate evaluation plots and metrics

Save the final model as final_lung_cancer_model.pth