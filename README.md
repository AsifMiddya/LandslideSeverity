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
