# Lung Cancer CT Scan Classification

## Project Overview
This project implements a deep learning classifier to detect lung cancer types from CT scan images using PyTorch. The model classifies CT scans into three categories: Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma.

## Dataset Information
- **Dataset**: CT Scan Images of Lung Cancer
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/mdnafeesimtiaz/ct-scan-images-of-lung-cancer)
- **Classes**: 3
  - Adenocarcinoma
  - Large Cell Carcinoma 
  - Squamous Cell Carcinoma
- **Total Images**: ~1,000+ CT scan images

### Dataset Structure

ct-scan-images-of-lung-cancer/
├── adenocarcinoma/
│ ├── image1.png
│ ├── image2.png
│ └── ...
├── large.cell.carcinoma/
│ ├── image1.png
│ └── ...
└── squamous.cell.carcinoma/
├── image1.png
└── ...


## Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone or download the project files**

   ```bash
   # Extract the zip file to your desired location
Create a virtual environment (recommended)

bash
python -m venv lung_cancer_env
source lung_cancer_env/bin/activate  # On Windows: lung_cancer_env\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Download the dataset

Download from: https://www.kaggle.com/datasets/mdnafeesimtiaz/ct-scan-images-of-lung-cancer

Extract the dataset and place it in the project directory

Update the data_dir path in lung_cancer_classifier.py if needed

How to Run
Training the Model
bash
python lung_cancer_classifier.py
The script will:

Load and preprocess the CT scan images

Split data into training (80%) and validation (20%) sets

Train a ResNet18-based model with transfer learning

Save the best model as best_lung_cancer_model.pth

Generate evaluation plots and metrics

Expected Output
Training progress with loss and accuracy metrics

Training history plot (training_history.png)

Confusion matrix (confusion_matrix.png)

ROC curves (roc_curves.png)

Classification report in terminal

File Descriptions
lung_cancer_classifier.py - Main training and evaluation script

requirements.txt - Python dependencies

README.md - This documentation file

Analysis_Report.pdf - Comprehensive analysis of results

Model Architecture
Backbone: ResNet18 (pretrained on ImageNet)

Transfer Learning: Frozen early layers, fine-tuned later layers

Classifier: Custom fully connected layers with dropout

Input Size: 224x224 RGB images

Output: 3-class softmax classifier

Training Configuration
Batch Size: 32

Learning Rate: 0.001 with ReduceLROnPlateau scheduling

Epochs: 30

Optimizer: Adam with weight decay

Loss Function: Cross Entropy Loss

Performance Metrics
The model evaluation includes:

Accuracy and Loss curves

Confusion Matrix

ROC curves and AUC scores

Precision, Recall, F1-score per class

Overall classification report