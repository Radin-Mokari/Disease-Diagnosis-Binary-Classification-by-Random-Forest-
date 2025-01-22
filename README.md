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

![image](https://github.com/user-attachments/assets/2ab06056-fada-4ec7-9bbe-b3784055da26)
