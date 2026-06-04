# 🧠 Multimodal Biomedical Data Analysis (MBDA) - Final Project

![Python](https://img.shields.io/badge/Python-3.14%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_13.x-EE4C2C?logo=pytorch)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt)

This repository contains an advanced, interactive Desktop Application built for classifying the **EMNIST Letters dataset (A-Z)**. It provides a comprehensive comparative study between **Artificial Neural Networks (ANN/Multilayer Perceptron)** and **Convolutional Neural Networks (CNN)**.

Built with a modular, industry-standard ML architecture, this application features real-time background training (via `QThread`), native CUDA hardware acceleration, and an interactive Matplotlib dashboard.

---

## 📂 Project Structure

```text
MBDA_FP_GUI_ANN_CNN_root/
├── data/                       # Datasets automatically downloaded here (EMNIST)
├── outputs/                    # Exported model weights and artifacts
│   ├── ann/
│   │   └── checkpoints/
│   │   └── logs/
│   │   └── plots/
│   └── cnn/
│       └── checkpoints/
│       └── logs/
│       └── plots/
├── src/                        # Core source code
│   ├── configs/
│   │   └── default_config.py   # Global constants and hyperparameters
│   ├── datasets/
│   │   └── emnist_loader.py    # DataLoader, data augmentation, orientation fixes
│   ├── engine/
│   │   ├── trainer.py          # PyTorch training loop (Adam, StepLR, ReduceLROnPlateau)
│   │   └── evaluator.py        # Model inference and Matplotlib evaluation plots
│   ├── gui/
│   │   ├── interactive_viewers.py # PyQt6 Pop-ups (Dataset and Prediction Viewers)
│   │   ├── main_window.py      # Main Dashboard layout and styling
│   │   ├── plot_widget.py      # PyQt6 wrapper for Matplotlib Canvas
│   │   ├── tab_ann.py          # ANN specific UI and background thread logic
│   │   └── tab_cnn.py          # CNN specific UI and background thread logic
│   ├── models/
│   │   ├── ann.py              # DynamicMLP architecture
│   │   └── cnn.py              # EMNISTConvNet architecture
│   └── utils/
│       ├── metrics.py          # Precision, Recall, and Confusion Matrix logic
│       └── visualization.py    # Native Matplotlib plotting functions
├── main.py                     # 🚀 Application Entry Point
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🧮 Mathematical Principles and Architectures

This project implements two deep learning paradigms. Below are the governing mathematical principles for the operations performed in the backend.

### 1. Artificial Neural Network (MLP)
The ANN processes flattened 1D arrays of the 28x28 images ($X \in \mathbb{R}^{784}$).

* **Linear Transformation:** Each fully connected layer computes the dot product of weights $\mathbf{W}$ and inputs $\mathbf{x}$, plus a bias $\mathbf{b}$:
  $$ \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} $$
* **ReLU Activation:** Introduces non-linearity to allow the network to learn complex patterns:
  $$ f(z) = \max(0, z) $$
* **Dropout Regularization:** Randomly zeroes out a fraction $p$ of neurons during training to prevent overfitting.
* **Cross-Entropy Loss:** Evaluates the difference between the true label probability distribution $\mathbf{y}$ and the predicted probabilities $\mathbf{\hat{y}}$ (calculated internally via Softmax):
  $$ \mathcal{L} = -\sum_{i=1}^{C} y_i \log\left(\frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}\right) $$

### 2. Convolutional Neural Network (CNN)
The CNN preserves the 2D spatial topology of the images ($X \in \mathbb{R}^{1 \times 28 \times 28}$).

* **2D Convolution:** A sliding kernel $\mathbf{K}$ extracts spatial features (edges, curves) from the input image $\mathbf{I}$:
  $$ \mathbf{S}(i, j) = (\mathbf{I} * \mathbf{K})(i, j) = \sum_{m} \sum_{n} \mathbf{I}(i+m, j+n) \mathbf{K}(m, n) $$
* **Max Pooling:** Downsamples the spatial dimensions to reduce computational load and provide translation invariance:
  $$ f(S) = \max_{a,b \in S} X_{a,b} $$
* **Batch Normalization:** Stabilizes the learning process by normalizing the output of the previous activation layer across the mini-batch:
  $$ \hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}} \gamma + \beta $$

---

## 🚀 Getting Started

Follow these steps to set up the environment and run the application on your local machine. This guide assumes you have **Python 3.10+** installed.

### Step 1: Create a Virtual Environment
Navigate to the root directory of the project in your terminal (PowerShell/CMD) and run:
```powershell
python -m venv .venv
```

### Step 2: Activate the Virtual Environment
```powershell
# On Windows PowerShell
.\.venv\Scripts\activate

# On Mac/Linux
source .venv/bin/activate
```
*(You should see `(.venv)` appear in your terminal prompt).*

### Step 3: Upgrade pip and Install Dependencies
This project utilizes CUDA-enabled PyTorch for GPU acceleration. The `requirements.txt` is pre-configured to fetch the massive CUDA 13.x binaries.
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Launch the Application
```powershell
python main.py
```

---

## 🖥️ Usage Guide

1. **Information Tab:** Read the overview of the MBDA project.
2. **Configuration (Tab 1):** 
   * Click **📥 Load EMNIST Dataset**. (The app will automatically download it on the first run and load locally on subsequent runs).
   * Adjust hyperparameters (Epochs, Learning Rate, Dropout).
   * Click **🏗️ Build Model** to instantiate the architecture onto your GPU.
3. **Training (Tab 2):** 
   * Click **▶️ Start Training**.
   * Monitor the real-time Matrix-style terminal and progress bar. The UI will not freeze thanks to asynchronous `QThread` processing.
4. **Evaluation (Tab 3):** 
   * Click **🔍 Generate Interactive Dashboard** to view the Loss Curves, Per-Class Accuracy, and Normalized Confusion Matrix.
   * Click **👀 Inspect Predictions** to open a pop-up window where you can filter and investigate exactly which letters the model classified correctly or incorrectly.

---
*Designed for the Multimodal Biomedical Data Analysis (MBDA) Course.*