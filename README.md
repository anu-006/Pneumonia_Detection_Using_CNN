# Pneumonia Detection Using CNN 

A deep learning project that classifies chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**. The model uses transfer learning with an ImageNet-pretrained VGG16 network.

## Project Overview

Pneumonia is a respiratory infection that can affect the lungs. This project explores how deep learning can be applied to chest X-ray image classification.

It takes a chest X-ray image as input and predicts one of two classes:

- `NORMAL`
- `PNEUMONIA`

The notebook covers dataset preparation, image preprocessing, transfer learning, model training, and evaluation on a separate test dataset.

## Results

Results from the current notebook run:

| Metric | Result |
|---|---:|
| Best validation accuracy | 97.68% |
| Test accuracy | 92.26% |
| Test loss | 0.1989 |
| NORMAL precision | 99% |
| NORMAL recall | 80% |
| PNEUMONIA precision | 89% |
| PNEUMONIA recall | 99% |

The test set contains 620 images: 234 NORMAL and 386 PNEUMONIA.

**How to interpret the results:** The model has high recall for the PNEUMONIA class in this test set, while recall for NORMAL images is lower. Accuracy alone does not describe the full performance, so the notebook also generates a classification report and confusion matrix.

These results are from one experiment and may vary with the environment, dataset version, and training run. They should not be interpreted as evidence of clinical reliability.

## Tech Stack

- **Python** – Main programming language
- **TensorFlow / Keras** – Deep learning and model training
- **VGG16** – ImageNet-pretrained convolutional neural network used for transfer learning
- **NumPy & Pandas** – Data handling
- **Scikit-learn** – Classification metrics and confusion matrix
- **Matplotlib & Seaborn** – Training plots and confusion matrix visualisation
- **Kaggle Notebooks** – Development and training environment

## Model Architecture

1. Load the VGG16 model with ImageNet weights and remove its original classification head.
2. Freeze the VGG16 convolutional base and use it as a feature extractor.
3. Resize input X-ray images to `224 × 224`.
4. Apply VGG16-specific image preprocessing.
5. Pass extracted features through Global Average Pooling and fully connected Dense layers.
6. Use a sigmoid output layer for binary classification.
7. Train with the Adam optimizer and binary cross-entropy loss.
8. Use Early Stopping to stop training when validation loss stops improving.

## Dataset

The notebook uses the **Chest X-ray Images (Pneumonia)** dataset available through Kaggle:

[Beginner Chest X-ray Image Classification – Kaggle dataset](https://www.kaggle.com/datasets/vivek468/beginner-chest-xray-image-classification)

The notebook uses the `train` directory for an 80:20 training-validation split and the separate `test` directory for final evaluation.

The dataset is not included in this repository. Download it from Kaggle and add it to your Kaggle Notebook, or update the dataset paths in the notebook if running locally.

## Run the Project

### Option 1: Run on Kaggle (recommended)

1. Open the notebook in Kaggle.
2. Add the chest X-ray dataset using **Add Input**.
3. Confirm that the dataset paths in the notebook match the attached dataset.
4. Run the notebook cells in order.
5. Review the training curves, test metrics, and confusion matrix.

### Option 2: Run locally

1. Clone this repository:

   ```bash
   git clone https://github.com/anu-006/Pneumonia_Detection_Using_CNN.git
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle, arrange it as shown above, and update `data_dir` and the test directory path in the notebook.

5. Open and run the notebook:

   ```bash
   jupyter notebook
   ```

   If Jupyter Notebook is not installed, install it with `pip install notebook`.

**Note:** The notebook loads VGG16 ImageNet weights. A working internet connection may be needed the first time those weights are downloaded. Training can be resource-intensive; a Kaggle Notebook or another environment with sufficient memory and compute is recommended.

## Evaluation

The notebook evaluates the trained model on the separate test dataset and reports:

- Accuracy and loss
- Precision, recall, and F1-score for each class
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves

For this medical-image classification task, recall for the PNEUMONIA class is an important metric because false-negative predictions require particular attention.

## Repository Structure

```text
Pneumonia_Detection_Using_CNN/
├── pneumonia-detection-using-cnn.ipynb
├── requirements.txt
└── README.md
└── chest_xray_classifier.keras
```


