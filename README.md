# BloodMNIST Classification: Deep Learning Lab 10

This project focuses on the classification of peripheral blood cell images using the **BloodMNIST** dataset. It compares a **Custom Convolutional Neural Network (CNN)** architecture against a **Transfer Learning** approach using **ResNet50** to achieve high-accuracy medical image classification.

---

## Problem Statement

* **Clinical Significance**: Accurate identification of different types of blood cells (e.g., neutrophils, lymphocytes, and basophils) is a critical step in diagnosing various hematological diseases, such as leukemia and anemia.
* **Current Challenges**: Traditional manual classification by pathologists is time-consuming and prone to human error.
* **Project Objective**: To automate this process by building a deep learning pipeline that:
    * Processes microscopic **28 x 28** blood cell images.
    * Addresses the challenge of **multi-class classification** involving 8 distinct categories.
    * Optimizes training time and computational resources using **TPU** hardware.
    * Maintains a target test accuracy of **85% or higher**.

---

## Dataset Information

The **BloodMNIST** dataset is part of the **MedMNIST v2** collection, a large-scale lightweight benchmark of 2D biomedical datasets.

* **Classes**: 8 (Basophil, Eosinophil, Erythroblast, Immature Granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet).
* **Total Samples**: 17,092.
* **Data Splits**: Train (11,959), Validation (1,712), and Test (3,421).
* **Source**: MedMNIST v2.
* **Link**: [https://medmnist.com/](https://medmnist.com/)

---

## Explanation of the Workflow

The notebook follows a structured machine learning pipeline:

### 1. Data Preprocessing & Augmentation
The original BloodMNIST images are 28 x 28 pixels. To optimize for deep learning models:
* **Resizing**: Images are upscaled to **64 x 64** to provide enough detail for the ResNet50 architecture while remaining efficient enough to prevent Out of Memory (OOM) crashes.
* **Normalization**: Pixel values are scaled to the range **[0, 1]**.
* **Augmentation**: To improve generalization, random horizontal/vertical flips and brightness/contrast adjustments are applied to the training set.

### 2. Model Architectures
Two distinct approaches were evaluated:

* **Custom CNN**: A lightweight model featuring three convolutional blocks. Each block utilizes **Batch Normalization** for training stability and **Dropout** (up to 50% in the dense head) to mitigate overfitting.
    * **Result**: Achieved a test accuracy of **87.43%**.
* **ResNet50 (Transfer Learning)**: Utilizing a model pre-trained on ImageNet.
    * **Phase 1**: Freezing the base and training only the classification head.
    * **Phase 2**: Fine-tuning the final 30 layers with a low learning rate (1e-5) to adapt the features to medical imagery.
    * **Result**: Achieved a test accuracy of **74.66%**.

### 3. Optimization Techniques
* **TPU Strategy**: The models are wrapped in a `distribute.TPUStrategy` scope to leverage Tensor Processing Units for rapid training.
* **Callbacks**: `EarlyStopping` was implemented to monitor validation loss and prevent unnecessary training once the model converged.

---

## Performance Summary

| Model | Test Accuracy | Status |
| :--- | :--- | :--- |
| **Custom CNN** | **87.43%** | **Target Reached** |
| **ResNet50** | **74.66%** | **Underperforming** |

---
