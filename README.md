# Disease-Diagnosis-Binary-Classification-by-Random-Forest-
Binary classification ML models that are based on random forest technique. The model are for predicting the presence of heart attack, diabetes and stroke diseases.
These models are built for a coursework of the AI module in the University of Bradford.
The datasets that used for training the models have been provided by the university.

The first dataset has 22 columns and more than 253000 rows. It includes features about the characteristics of patients such as their physical/mental health, thier diet, BMI, age and whether they are a smoker.
By using this dataset, three separate machine learning models have been built using random forest technique. For data normalization and preparation, SMOTE and MinMaxScalar functions have been applied. Additionally, since the utilized dataset was imbalanced, required process has been applied to the dataset to create samples of the minority class.

Also, a separate model has been generated to predict the presence of multiple disease simultaneously. This model uses a multi output classifier which is based on Decision Tree.

The second dataset is the "pneumonia MNIST" that consist x-ray images of the chest of patients. CNN binary classification technique is used to make predictions on the presence of pneumonia from x-ray images.

For each model, a user interaction part has been added to the end of the model to make predictions based on the used inputs.

Finally, all the trained models has been saved and download as .pkl for a later usage and a program has been created that used these trained models and lets the used to make disease diagnosis by inputing data or uploading chest a x-ray image.

# Machine Learning in Healthcare: Disease Diagnosis

This project explores the integration of machine learning (ML) techniques in healthcare, focusing on disease diagnosis and prediction. The report provides an overview of ML models, their implementation, and performance analysis. Additionally, challenges in applying AI to healthcare and future directions are discussed.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [ML Techniques in Healthcare](#ml-techniques-in-healthcare)
    - [Random Forest](#random-forest)
    - [Decision Tree](#decision-tree)
    - [Convolutional Neural Networks (CNN)](#cnn)
    - [Variational Autoencoders (VAE)](#vae)
4. [Challenges in Integrating AI in Healthcare](#challenges-in-integrating-ai-in-healthcare)
5. [Methodology and Data](#methodology-and-data)
    - [Tabular Dataset](#tabular-dataset)
        - [Binary Classification](#binary-classification)
        - [Multi-Output Classification](#multi-output-classification)
    - [X-ray Images Dataset](#x-ray-images-dataset)
        - [CNN Classification](#cnn-classification)
        - [VAE](#vae)
    - [AI Program with User Interaction](#ai-program-with-user-interaction)
6. [Analysis and Discussion](#analysis-and-discussion)
    - [RF Binary Classification](#rf-binary-classification)
    - [Decision Tree Multi-Classification](#decision-tree-multi-classification)
    - [CNN Classification](#cnn-classification-performance)
    - [VAE](#vae-performance)
7. [Conclusions and Suggestions for Future Work](#conclusions-and-suggestions-for-future-work)
8. [References](#references)

---

## Introduction

The integration of machine learning into the medical sector has revolutionized disease diagnosis. ML enables systems to learn from data, identify patterns, and make predictions (Basu Dev Shivahare et al., 2024). This project discusses the ML models developed to predict disease presence, evaluates their performance, and highlights the building blocks of each model.

---

## Background

The healthcare industry is increasingly leveraging artificial intelligence (AI), particularly deep learning (DL), to enhance diagnosis and treatment outcomes (Chaddad et al., 2023). This project focuses on four ML techniques: Random Forest, Decision Tree, Convolutional Neural Networks (CNN), and Variational Autoencoders (VAE).

---

## ML Techniques in Healthcare

### Random Forest
Random Forest (RF) is a robust supervised learning algorithm that builds multiple decision trees and aggregates their votes to make predictions. It is particularly effective in regression and classification tasks, such as heart disease diagnosis (Dhanka and Maini, 2021).

### Decision Tree
Decision Trees are widely used for supervised classification. They follow a tree-like structure with internal nodes representing feature tests, branches as outcomes, and leaf nodes as target classes (Jadhav and Channe, 2016).

### CNN
Convolutional Neural Networks (CNN) are popular in image classification, particularly in detecting diseases from X-ray images. CNNs extract features through convolutional layers and use these features for binary classification (Behera et al., 2024).

### VAE
Variational Autoencoders (VAEs) are deep learning models used for data reconstruction. They consist of an encoder-decoder architecture, where the encoder encodes input into latent space and the decoder reconstructs it. VAEs are useful in semi-supervised learning and image generation (Pu et al., 2016).

---

## Challenges in Integrating AI in Healthcare

Integrating ML in healthcare faces challenges such as:
- **Data Scarcity**: Limited availability of high-quality labeled data for training.
- **Data Imperfection**: Presence of unclean, biased, or incomplete data.
These issues can lead to biased or inaccurate models (Vidhya et al., 2023).

---

## Methodology and Data

### Tabular Dataset
The tabular dataset includes health characteristics and labels for three diseases: heart disease, diabetes, and stroke. Features are numerical, categorical, or binary.

#### Binary Classification
- **Data Preparation**: Normalization and handling of categorical data.
- **Resampling**: Synthetic Minority Oversampling Technique (SMOTE) to balance classes.
- **Model Development**: Random Forest classifiers for each disease.
- **User Interaction**: A prediction function for user-provided inputs.

#### Multi-Output Classification
- Uses a Decision Tree classifier to predict the presence of all three diseases simultaneously.
- Includes normalization and separate resampling for each target.

### X-ray Images Dataset

#### CNN Classification
- **Preprocessing**: Images resized, normalized, and augmented.
- **Architecture**: Convolutional layers for feature extraction, max-pooling for dimensionality reduction, and dense layers for classification.
- **Output**: Binary classification for pneumonia detection.

#### VAE
- **Preprocessing**: Images resized to 28x28 pixels and normalized.
- **Architecture**: Encoder and decoder networks for image reconstruction.

### AI Program with User Interaction
A program integrates pre-trained models to:
1. Diagnose heart attack, diabetes, or stroke using user-provided data.
   ![image](https://github.com/user-attachments/assets/b96a9db7-d243-43b9-a421-e47f98122537)
3. Detect pneumonia from user-uploaded X-ray images.
   ![image](https://github.com/user-attachments/assets/0a847796-8a5a-4430-ab52-4a94250be6cd)


---

## Analysis and Discussion

### RF Binary Classification
Classification reports and confusion matrices demonstrate model performance:
- Heart Disease: High recall and precision.
- Diabetes: Moderate performance, some false positives.
- Stroke: Strong overall performance.

### Decision Tree Multi-Classification
Multi-output classification results:
- Stroke diagnosis outperformed other diseases in precision and recall.
- Diabetes diagnosis overpredicted positive cases.

### CNN Classification Performance
The CNN achieved high accuracy for pneumonia detection. However, fluctuations in validation loss indicate potential overfitting.

### VAE Performance
The VAE demonstrated high accuracy in reconstructing images, with a relative root mean square error of 18.96%.

---

## Conclusions and Suggestions for Future Work

The trained models are effective for disease diagnosis, particularly binary classification models for tabular datasets. Future work could explore:
- Implementing balanced random forest classifiers for multi-output classification.
- Addressing data challenges through advanced augmentation and synthetic data generation.

---

## References

1. Basu Dev Shivahare et al. (2024). _Delving into Machine Learning’s Influence on Disease Diagnosis and Prediction_. The Open Public Health Journal, 17(1).
2. Behera, N. et al. (2024). _CNN-Based Medical Image Classification Models for Pneumonia and Malaria Identification_. IEEE.
3. Chaddad, A. et al. (2023). _Survey of Explainable AI Techniques in Healthcare_. Sensors, 23(2), 634.
4. Dhanka, S., & Maini, S. (2021). _Random Forest for Heart Disease Detection_. IEEE.
5. Jadhav, S.D., & Channe, H.P. (2016). _Comparative Study of Classification Techniques_. IJSR, 5(1), 1842–1845.
6. Pu, Y. et al. (2016). _Variational Autoencoder for Images, Labels, and Captions_. arXiv.
7. Vidhya, G. et al. (2023). _Quality Challenges in Deep Learning Data Collection_. Journal of Information Technology and Computing, 4(1), 46–58.

