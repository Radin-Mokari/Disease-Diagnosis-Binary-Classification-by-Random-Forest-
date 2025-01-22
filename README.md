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

Table of Contents

Introduction	
Background	
ML Techniques in Healthcare	
Random Forest	
Decision Tree	
CNN	
VAEs	
Challenges in Integrating AI in Healthcare	
Methodology And Data	
Tabular Dataset	
Binary Classification	
Multi-Output Classification	
X-ray Images dataset	
CNN Classification	
VAE	
AI Program with User Interaction	
Analysis And Discussion
RF Binary Classification	
Decision Tree Multi-Classification	
CNN Classification
VAE	
Conclusions And Suggestions for Future Work
References

Introduction
The integration of machine learning into medical sector promoted a revolution in disease diagnosis. Machine learning empowers computer systems to learn from data, identify intricate patterns and make predictions (Basu Dev Shivahare et al. 2024). In this report, the ML models that have been built for predictions on disease presence will be discussed and analysed. Thought this report, firstly an overview of the ML techniques used for the models will be discussed. Then the building blocks of each model will be covered. And finally the performance of each model will be analysed.
Background
The healthcare industry makes substantial use of artificial intelligence (AI), particularly AI based on deep learning (DL) models (Chaddad et al. 2023). In this study, the machine learning techniques that used for the built models are Random Forest, Decision Tree, Convolutional Neural Network, and Variational Autoencoder.
ML Techniques in Healthcare
Random Forest
The random forest (RF) algorithms is a very suggested ML model technique that summarizes medical data. Rf learns from past information patterns, and it performs by building multiple decision trees, that each tree votes depending on data characteristics and make a prediction by the majority vote. It is effective for regression and classification. One key application of RF is heart disease diagnosis that it will be covered in this report (Dhanka and Maini 2021).
Decision Tree
Another widely used supervised ML techniques is decision tree. It utilizes a tree-based construction and plots investigations regard to item to conclusions about the target value of the item. The tree structure has internal nodes, leaf nodes and a root node. Each internal node refers to a test on a feature, all leaf nodes is assigned to a class and each branch corresponds to the test consequence (Jadhav and Channe 2016).
CNN
Convolutional neural network (CNN) is one of the most popular deep learning models applied for image classification, especially in medical industry. These networks perform a binary classification of detecting if a condition exists or not, therefore they are used primarily for disease detection from X-ray images. These models typically contain a sequence of convolutional layers that try to learn features from a digitized X-ray image. These layers extract useful visualizations and patterns, and then they are applied to categorization (Behera et al. 2024).
VAEs
Variational autoencoder is e deep learning technique that reconstruct a data. Once trained on some data, a VAE would take a sample from the latent space and generate a new sample that closely resembles the input data. The two main components of VAEs are:
Encoders that encode input data into the latent space and decoders that decode the latent representation back into the input space. Essentially, the task of the encoder network is to learn to take a data input and convert it into some low-dimensional latent space. This description of input into the model is translated into the latent space, then reconstructed by a decoder network. VAEs are used to different types of tasks such as semi-supervised learning, text/image generation (Pu et al. 2016).
Challenges in Integrating AI in Healthcare
In the medical field preprocessing and data quality hold great importance for machine learning (ML), especially for deep learning (DL). Building precise and trustworthy ML models that help with tasks like disease diagnosis needs very high-quality data. Based on this truth data challenges to tackle include data scarcity, meaning the real-world data needed for DL training is small, and data imperfection, which means unclean, biased or corrupted information. These issues lead to biased or inaccurate models (Vidhya et al. 2023).
Methodology And Data
Tabular Dataset
Binary Classification
The tabular dataset contains a range of features about health characteristics and 3 diseases (heart disease or attack, diabetes, stroke). The data is either numerical, categorical or binary. For each disease, feature importance has been discovered, and a separate random forest model has been built.
Data Preparation and Normalization and model development
In the csv file, the column “Diabetes” has 3 value (0, 1 and 2). However, since both 1 and 2 represent “true”, all 2 is replaced with 1 with the following code:
![image](https://github.com/user-attachments/assets/4db169b2-c34f-4615-87de-d7dea7d7e095)
On the other hand, since there are numerical and categorical features in the dataset, a normalization is required to be apply before splitting the data into training and testing. The following code block is used to normalize it to a range between 0 and 1.
![image](https://github.com/user-attachments/assets/48a82a97-dc4b-450e-b655-8d36db745708)
After that, the SMOTE function is applied to balance the dataset by adding synthetic data points to the minority class.
Now the dataset is normalized and prepared, it’s time to split it into training and testing. Finally, the hyperparameters are modified and adjusted to prevent overfitting and increase the robustness of the model.
For all disease, the same steps have been applied with a difference in changing the disease column that defines the target.
Additionally, at the end of each model, a prediction function has been built to make a prediction based on the user inputs.
Multi-Output Classification
For building a model that predicts the presence of each disease simultaneously, the same normalization has been applied to the dataset, but, since we have 3 separate outputs, the resampling is applied for each target to handle multi-output prediction. In this model, a Decision Tree classifier is used as the base model for multi output classification. Also, similar to the RF models, a user interaction has been added to get data from user.
X-ray Images dataset
For this dataset, a CNN classification for the presence of pneumonia, and a variational autoencoder have been created. 
CNN Classification
The data preparation step of this model is done by collecting and preprocessing, including resizing images to a uniform dimension, pixel values normalization, and augmentation with transformations such as rotation and flipping to improve generalization. It is importance to mention that the model has been built in the same notebook file as the dataset for an easier implementation and debugging.
The architecture of this model is built by convolutional layers for extracting the features, followed by max-pooling layers for dimensionality reduction and dense layers for classification. Also, non-linearity has been specified by ReLU activation. And finally, the Sigmoid activation function is utilized in the last layer for binary classification.
Although the model has a part for make a prediction by using an uploaded image, a separate program has been built that uses the pre-trained model.
VAE 
As it discussed before, this model is for reconstructing images. Same as the CNN model, the Pneumonia MNIST dataset was used with data preprocessing that includes resizing images to 28 × 28 pixels, normalization and batching. The model architecture has two main parts: encoder and decoder. The encoder has convolutional layers for feature extraction, computing latent space parameters and sampling. On the other hand, the decoder has transposed convolutional layers for reconstructing images from latent variables.
AI Program with User Interaction
By saving and downloading the pre-trained models that we covered, a program has been built to make disease diagnosis based on the user data. The program has 2 components: a diagnosis for heart attack, diabetes or stroke, and a pneumonia diagnosis for the uploaded image from user. An example for working with each one is demonstrated below:
![image](https://github.com/user-attachments/assets/9390e4fc-d83f-4e67-9ba5-125cf44953a7)
![image](https://github.com/user-attachments/assets/23d87fa9-bc1b-48d8-94be-7123f47440e0)
Analysis And Discussion
RF Binary Classification
Now it is apparent to dive into model analysis and discuss their performance. The performance of the classification models can be demonstrated by classification report and confusion matrix. For RF models that were built by the tabular data, the performance for all disease diagnosis is approximately similar.:
RF Heart Disease Diagnosis
![image](https://github.com/user-attachments/assets/ca4512a7-8023-4217-8796-83499a386562)
RF Diabetes Diagnosis
![image](https://github.com/user-attachments/assets/c09ca8d3-b834-4789-a588-54230d61c945)
RF Stroke Diagnosis
![image](https://github.com/user-attachments/assets/57d0c572-d9b5-4ba3-a9ea-5ab4d1482dfe)
Decision Tree Multi-Classification
According to the above images, heart disease and stroke diagnosis models have better accuracy and F1-scores then the diabetes model. Also, heart disease has the highest recall score for class 1.
Turning to the performance of the multi output classification model, as represented below, the heart disease diagnosis model has a coherent precision, recall and F1-scores. For the diabetes diagnosis, the model overpredicts positive cases, and this leads to false positives. On the other hand, the stroke detection has the strongest performance, which can be seen from excellent precision and recall for both classes.
Multi-classification report and confusion matrix
![image](https://github.com/user-attachments/assets/323f910a-bce8-4152-a6cd-17bcd479c680)
CNN Classification
As it can be seen in the following pictures, the model obtains high accuracy on the test set, which displays its ability to detect pneumonia cases correctly. However, the fluctuations in validation loss refers to some noise sensitivity or overfitting.
![image](https://github.com/user-attachments/assets/e8c6bf9c-cb56-4ba4-b136-d1e3c7a21f30)
![image](https://github.com/user-attachments/assets/8c4a78f3-d7f5-41fb-9b60-76131c12c935)  ![image](https://github.com/user-attachments/assets/ac2c2f64-cb2f-4c03-92c2-dbd8490fab34)
![image](https://github.com/user-attachments/assets/b481bb2d-3160-4e57-a7a8-c7eb5f169f16)
VAE
Regard to the performance of the VAE model, the absolute root mean square error shows that the reconstructed images significantly close to the original images. And the relative root mean square error is 18.96%, which indicates that the model captures key data distributions.
![image](https://github.com/user-attachments/assets/8e645387-26d8-48e0-8108-1a12a75c4b6d)

Conclusions And Suggestions for Future Work
In conclusion, the trained models are usable for disease diagnosis, particularly the binary classification models that have been built for the tabular dataset. However, several refinements and improvements can be applied to advance the performance of the models. For instance, a balanced random forest classifier would be a better chois for multi-output classifier.

References
Basu Dev Shivahare, Singh, J., Ravi, V., Radha Raman Chandan, Tahani Jaser Alahmadi, Singh, P. and Manoj Diwakar (2024) Delving into Machine Learning’s Influence on Disease Diagnosis and Prediction. The Open public health journal 17 (1), Bentham Science Publishers.
Behera, N., Das, S., Singh, A.P., Swain, A.K. and Rout, M. (2024) CNN-Based Medical Image Classification Models for the Identification of Pneumonia and Malaria . 2024 International Conference on Advancements in Smart, Secure and Intelligent Computing (ASSIC) IEEE.
Chaddad, A., Peng, J., Xu, J. and Bouridane, A. (2023) Survey of Explainable AI Techniques in Healthcare. Sensors 23 (2), 634.
Dhanka, S. and Maini, S. (2021) Random Forest for Heart Disease Detection: A Classification Approach. 1–3https://ieeexplore.ieee.org/abstract/document/9699506 Accessed.
Jadhav, S.D. and Channe, H.P. (2016) Comparative Study of K-NN, Naive Bayes and Decision Tree Classification Techniques. International Journal of Science and Research (IJSR) 5 (1), 1842–1845.
Pu, Y., Gan, Z., Henao, R., Yuan, X., Li, C., Stevens, A. and Carin, L. (2016) Variational Autoencoder for Deep Learning of Images, Labels and Captions. arXiv (Cornell University) Cornell University.
Vidhya, G., Nirmala, D. and Manju, T. (2023) Quality challenges in Deep Learning Data Collection in perspective of Artificial Intelligence. Journal of Information Technology and Computing 4 (1), 46–58.
