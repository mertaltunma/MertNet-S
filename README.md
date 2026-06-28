# Deep Feature Extraction with MertNet-S and Classical Machine Learning for Image Classification

## Project Overview

In this project, the MertNet-S deep learning model was designed and trained from scratch using the Stable ImageNet-1K dataset. The trained model was exported in TorchScript format and used as a deep feature extractor on the Intel Image Classification dataset.

After end-to-end training on Stable ImageNet-1K, the MertNet-S model was used to extract 384-dimensional deep feature vectors from the target dataset images. Then, the most discriminative 128 features were selected using the ANOVA F-score based SelectKBest method, and image classification experiments were performed using classical machine learning algorithms.

This project investigates how effectively deep representations learned from a large-scale source dataset can be transferred to a different image classification problem.

---

## Workflow

```text
Stable ImageNet-1K
        │
        ▼
MertNet-S Training
        │
        ▼
TorchScript Model Export
        │
        ▼
Intel Image Classification Dataset
        │
        ▼
Deep Feature Extraction
        │
        ▼
Feature Selection
(384 → 128)
        │
        ▼
Classical Machine Learning
(Logistic Regression / SVM (RBF) / Random Forest)
        │
        ▼
Performance Evaluation
```

---

## Datasets

### Stable ImageNet-1K

The Stable ImageNet-1K dataset was used for training the source model. The model was trained end-to-end on this dataset for a 1000-class image classification problem.

Kaggle link:

```text
https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k
```

Expected directory structure:

```text
data/
└── stable_imagenet_1k/
    └── raw/
        └── imagenet1k/
            ├── 000_tench, Tinca tinca/
            ├── 001_goldfish, Carassius auratus/
            └── ...
```

### Intel Image Classification

The Intel Image Classification dataset was used to evaluate the trained MertNet-S model as a deep feature extractor. The dataset contains 6 classes: buildings, forest, glacier, mountain, sea, and street.

Kaggle link:

```text
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
```

Expected directory structure:

```text
data/
└── intel_image_classification/
    └── raw/
        ├── seg_train/
        │   └── seg_train/
        │       ├── buildings/
        │       ├── forest/
        │       ├── glacier/
        │       ├── mountain/
        │       ├── sea/
        │       └── street/
        │
        ├── seg_test/
        │   └── seg_test/
        │       ├── buildings/
        │       ├── forest/
        │       ├── glacier/
        │       ├── mountain/
        │       ├── sea/
        │       └── street/
        │
        └── seg_pred/
            └── seg_pred/
```

---

## About the Data Directory

The `data/` directory is not included in the project package because the datasets require a large amount of storage. Stable ImageNet-1K and Intel Image Classification together may occupy more than 10 GB of disk space, so keeping these files directly inside the project package is not practical.

Therefore, users who want to run the code from the beginning should download the datasets from the Kaggle links above and place them into the same directory structure.

Without the `data/` directory, the following files cannot be executed again:

```text
source_training/check_stable_imagenet_dataset.py
source_training/create_sample_grid.py
source_training/create_stable_imagenet_split.py
source_training/dataset_stable_imagenet.py
source_training/train_stable_imagenet.py

target_dataset_experiments/check_intel_dataset.py
target_dataset_experiments/create_intel_sample_grid.py
target_dataset_experiments/extract_features.py
```

However, the project already includes trained models, training logs, training curves, extracted features, selected features, classification results, and confusion matrix outputs. Therefore, the experimental results can still be inspected without the `data/` directory.

---

## Downloading Datasets with Kaggle CLI

If the Kaggle API is used, first download the `kaggle.json` API key from the Kaggle account settings. Then place this file into the following location:

macOS / Linux:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

On Windows, the `kaggle.json` file is usually placed into:

```text
C:\Users\<Username>\.kaggle\kaggle.json
```

To download the Stable ImageNet-1K dataset:

```bash
mkdir -p data/stable_imagenet_1k/raw

kaggle datasets download \
  -d vitaliykinakh/stable-imagenet1k \
  -p data/stable_imagenet_1k/raw \
  --unzip
```

To download the Intel Image Classification dataset:

```bash
mkdir -p data/intel_image_classification/raw

kaggle datasets download \
  -d puneet6060/intel-image-classification \
  -p data/intel_image_classification/raw \
  --unzip
```

After downloading, the directory structure should match the expected structures shown above.

---

## Python Environment Setup

Creating a Python virtual environment is recommended.

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

The required packages can be installed using `requirements.txt`.

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```text
torch
torchvision
torchaudio
numpy
pandas
scikit-learn
matplotlib
pillow
tqdm
pyyaml
opencv-python
kaggle
onnx
onnxruntime
```

---

## Project Structure

```text
MertNet-S/
│
├── data/                         # Download separately
├── data_splits/                  # Stable ImageNet-1K train/validation splits
├── exported_model/               # Trained and exported models
├── results/                      # Training, feature extraction, and classification outputs
├── source_training/              # Source model training and export scripts
├── target_dataset_experiments/   # Target dataset experiment scripts
├── README.md
└── requirements.txt
```

---

## Important Files

### Source Model Training

```text
source_training/model_architecture.py
source_training/dataset_stable_imagenet.py
source_training/train_stable_imagenet.py
source_training/export_model.py
source_training/inference_stable_imagenet.py
```

### Target Dataset Experiments

```text
target_dataset_experiments/check_intel_dataset.py
target_dataset_experiments/create_intel_sample_grid.py
target_dataset_experiments/extract_features.py
target_dataset_experiments/feature_selection.py
target_dataset_experiments/classify_features.py
```

### Model Files

```text
exported_model/mertnet_s_best.pth
exported_model/mertnet_s_last.pth
exported_model/mertnet_s_torchscript.pt
```

### Result Directories

```text
results/stable_imagenet/
results/intel_dataset/
results/intel_features/
results/intel_feature_selection/
results/intel_classification/
```

---

## Execution Order

After placing the datasets into the correct directory structure, the project can be executed in the following order.

### 1. Check the Stable ImageNet-1K dataset

```bash
python source_training/check_stable_imagenet_dataset.py
```

### 2. Create the Stable ImageNet-1K train/validation split

```bash
python source_training/create_stable_imagenet_split.py
```

### 3. Create a sample image grid

```bash
python source_training/create_sample_grid.py
```

### 4. Train the MertNet-S model

```bash
python source_training/train_stable_imagenet.py
```

For a quick test:

```bash
python source_training/train_stable_imagenet.py --smoke_test
```

### 5. Export the model

```bash
python source_training/export_model.py
```

### 6. Check the Intel Image Classification dataset

```bash
python target_dataset_experiments/check_intel_dataset.py
```

### 7. Create an Intel sample image grid

```bash
python target_dataset_experiments/create_intel_sample_grid.py
```

### 8. Extract deep features

```bash
python target_dataset_experiments/extract_features.py
```

### 9. Perform feature selection

```bash
python target_dataset_experiments/feature_selection.py
```

### 10. Perform classification with classical machine learning

```bash
python target_dataset_experiments/classify_features.py
```

---

## Machine Learning Algorithms

The 128-dimensional deep features extracted from the Intel Image Classification dataset were classified using the following classical machine learning algorithms:

```text
Logistic Regression
Support Vector Machine (RBF)
Random Forest
```

---

## Experimental Results

### MertNet-S Training on Stable ImageNet-1K

```text
Best validation Top-1 accuracy  : 80.82%
Final validation Top-1 accuracy : 80.60%
Final validation Top-5 accuracy : 94.62%
```

### Classification on Intel Image Classification

```text
Logistic Regression accuracy : 85.97%
Random Forest accuracy       : 86.23%
SVM (RBF) accuracy           : 87.57%
```

The best-performing classifier was SVM (RBF).

---

## Generated Outputs

Stable ImageNet-1K training outputs:

```text
results/stable_imagenet/stable_imagenet_training_log.csv
results/stable_imagenet/stable_imagenet_loss_curve.png
results/stable_imagenet/stable_imagenet_accuracy_curve.png
results/stable_imagenet/stable_imagenet_training_summary.txt
```

Intel feature extraction outputs:

```text
results/intel_features/intel_train_features.npy
results/intel_features/intel_test_features.npy
results/intel_features/intel_train_labels.npy
results/intel_features/intel_test_labels.npy
results/intel_features/intel_feature_extraction_summary.txt
```

Feature selection outputs:

```text
results/intel_feature_selection/intel_train_features_selected.npy
results/intel_feature_selection/intel_test_features_selected.npy
results/intel_feature_selection/selected_feature_indices.npy
results/intel_feature_selection/selected_feature_scores.csv
results/intel_feature_selection/feature_selection_summary.txt
```

Classification outputs:

```text
results/intel_classification/classification_results.csv
results/intel_classification/classification_results.json
results/intel_classification/classification_summary.txt
results/intel_classification/svm_rbf_confusion_matrix.png
```

---

## Author

Mert Altun

2026