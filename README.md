# Cats vs Dogs Image Classification using SVM

## Overview

This project was completed as part of the Machine Learning Internship at Prodigy InfoTech.

The objective of this project is to develop a Support Vector Machine (SVM) model capable of classifying images of cats and dogs. The model uses image preprocessing techniques and machine learning algorithms to distinguish between the two categories.

---

## Dataset

Dataset Used:
Microsoft Cats and Dogs Dataset (PetImages)

Dataset Structure:

* Cat Images
* Dog Images

Total Images Used:

* Approximately 1000 Cat Images
* Approximately 1000 Dog Images

---

## Technologies Used

* Python
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib

---

## Project Workflow

### 1. Data Collection

Loaded cat and dog images from the dataset.

### 2. Image Preprocessing

* Converted images to grayscale
* Resized images to 64 × 64 pixels
* Normalized pixel values
* Flattened images into feature vectors

### 3. Feature Scaling

Applied StandardScaler to standardize image features.

### 4. Train-Test Split

* Training Data: 80%
* Testing Data: 20%

### 5. Model Training

Used Support Vector Machine (SVM) with RBF Kernel.

### 6. Model Evaluation

Evaluated model performance using:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report

---

## Model Performance

### Accuracy

60.75%

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| Cat   | 0.60      | 0.62   | 0.61     |
| Dog   | 0.61      | 0.59   | 0.60     |

Overall Accuracy: 60.75%

---

## Key Concepts Applied

* Image Processing
* Computer Vision
* Feature Extraction
* Data Normalization
* Support Vector Machines (SVM)
* Classification Algorithms

---

## Project Structure

PRODIGY_ML_03_Cats_vs_Dogs_SVM

├── data

│   └── PetImages

│       ├── Cat

│       └── Dog

├── src

│   ├── preprocess_images.py

│   └── cats_dogs_svm.py

├── README.md

├── requirements.txt

└── .gitignore

---

## Results

The SVM classifier successfully learned to distinguish between cat and dog images after preprocessing and feature scaling. The project demonstrates the application of machine learning techniques to image classification problems and provides a foundation for more advanced computer vision models.

---

## Internship

Machine Learning Internship

Prodigy InfoTech

Task 03 – Cats vs Dogs Image Classification using Support Vector Machine (SVM)
