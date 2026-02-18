# CIFAR-10 Image Classifier with CNN & AlexNet

A deep learning image classification project built using Convolutional Neural Networks (CNN) with an AlexNet-inspired architecture, trained and evaluated on the CIFAR-10 dataset.

---

## 📌 Overview

This project implements an image classifier capable of recognizing 10 different object categories from the CIFAR-10 dataset. The model leverages the AlexNet architecture adapted for the CIFAR-10 image size, trained using the AdamW optimizer and Cross-Entropy loss.

---

## 🗂️ Dataset

**CIFAR-10** consists of 60,000 color images (32×32 pixels) across 10 classes:

| Label | Class      |
|-------|------------|
| 0     | Airplane   |
| 1     | Automobile |
| 2     | Bird       |
| 3     | Cat        |
| 4     | Deer       |
| 5     | Dog        |
| 6     | Frog       |
| 7     | Horse      |
| 8     | Ship       |
| 9     | Truck      |

- **Training samples:** 50,000  
- **Test samples:** 10,000

---

## 🏗️ Model Architecture

The classifier is based on **AlexNet**, adapted for the 32×32 input resolution of CIFAR-10. The architecture includes:

- Multiple convolutional layers with ReLU activations
- Max pooling layers for spatial downsampling
- Dropout layers for regularization
- Fully connected layers for classification
- Softmax output over 10 classes

---

## ⚙️ Training Configuration

| Parameter     | Value              |
|---------------|--------------------|
| Optimizer     | AdamW              |
| Loss Function | Cross-Entropy Loss |
| Dataset       | CIFAR-10           |

---

## 📊 Results

Training and validation performance is visualized through:

- **Accuracy Plot** — Training vs. Validation accuracy over epochs
- **Loss Plot** — Training vs. Validation loss over epochs

> Plots are generated at the end of the notebook.

---

## 📁 Project Structure
```
├── cifar10_classifier.ipynb   # Main notebook: data loading, model, training & evaluation
├── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision matplotlib numpy
```

### Run the Notebook
```bash
https://colab.research.google.com/drive/1nLYF6SN16-NYQw7BwmAraVaso1ikqBGS?usp=sharing
```

The notebook covers:
1. Loading and preprocessing the CIFAR-10 dataset
2. Defining the AlexNet-based CNN model
3. Training with AdamW optimizer and Cross-Entropy loss
4. Evaluating model performance
5. Plotting accuracy and loss curves

---

## 📦 Dependencies

- Python 3.x
- PyTorch 
- torchvision
- matplotlib
- numpy
- Google Colab

---


