# Landslide Size Classification using Stacked Neural Network Ensemble

This repository contains the implementation code for the research paper:

> **"Prediction of Severity in terms of Landslide Sizes using Stacked 
> Neural Network Ensemble Techniques"**  
> Debanjana Basu, Asif Iqbal Middya, Sarbani Roy  
> Department of Computer Science and Engineering, Jadavpur University, 
> Kolkata, India

---

## Overview

Landslides are among the most destructive natural disasters, causing 
significant loss of life and property, particularly in mountainous 
regions. This study addresses the problem of **landslide size 
classification** — predicting whether a landslide event is large, 
medium, or small in terms of severity — across three geographically 
distinct regions of India:

- **NILD** — North India Landslide Dataset
- **NEILD** — North East India Landslide Dataset  
- **SWILD** — South West India Landslide Dataset

The proposed approach employs a **stacked neural network ensemble**, 
combining three feed-forward neural networks as base learners with 
five machine learning classifiers as meta-learners.

---

## Model Architecture

Layer 1 (Base Learners):

├── ANN1: Input → Dense(20, relu) → Dense(25, relu) → Dense(3, softmax)

├── ANN2: Input → Dense(15, relu) → Dense(20, relu) → Dense(25, sigmoid) → Dense(3, softmax)

└── ANN3: Input → Dense(20, relu) → Dense(50, sigmoid) → Dense(50, relu) → Dense(3, softmax)

Layer 2 (Meta-Learners — evaluated separately):

├── SM-GB   : Gradient Boosting

├── SM-KNN  : K-Nearest Neighbours

├── SM-KSVM : Kernel Support Vector Machine

├── SM-NB   : Naive Bayes

└── SM-SVM  : Support Vector Machine


## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

---

## How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/[username]/landslide-size-prediction.git
cd landslide-size-prediction
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Open the notebook:**

```bash
jupyter notebook code/Stacking_DL_LSM.ipynb
```

4. **Set the dataset path** in the notebook:

```python
# Replace with your dataset file path
data = pd.read_csv('dataset.csv')
```

5. **Choose the meta-learner** by setting:

```python
# Options: 'KNN', 'SVM', 'KSVM', 'NB', 'GB'
META_LEARNER_NAME = 'GB'
```

6. **Run all cells.** Results will be saved automatically as CSV files 
in the working directory.

---


## Key Implementation Notes

- **Within-fold SMOTE:** SMOTE oversampling is applied strictly inside 
each training fold to prevent data leakage into the test fold.
- **StandardScaler:** Feature scaling is fitted on the training fold 
only and applied to the test fold, ensuring no leakage.
- **Random seed:** A fixed seed of `42` is used across NumPy, Python 
random, and TensorFlow for full reproducibility.
- **10-Fold Cross-Validation:** Implemented using `sklearn.KFold` with 
`shuffle=True` and `random_state=42`.

---

