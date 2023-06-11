﻿# ML2Project
 
 Project Goal/Motivation:
The goal of this project is to train a machine learning model to classify images using the COCO (Common Objects in Context) dataset. The motivation behind the project is to develop a model that can accurately identify objects in images and improve its performance through training and validation.

Data Collection:
The COCO dataset is loaded using the TensorFlow Datasets (tfds) library. The dataset is specifically the 'coco/2017' split from the training set. The dataset is shuffled, and a subset of 10,000 data points is selected. Each data point consists of an image and object labels. The images are resized to a target size of (128, 128) pixels.

Modeling:
The model used is a sequential model built using the Keras API. It consists of several layers, including Convolutional and Dense layers. The model architecture includes convolutional layers for feature extraction, max-pooling layers for downsampling, and fully connected layers for classification. The model is compiled with an Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

Interpretation:
The model is trained for 3 epochs on the training data and validated using the validation data. During training, the loss and accuracy metrics are monitored. After training, the model is evaluated on the test dataset, and the test accuracy is reported.

The model's performance is as follows: After the first epoch, the training accuracy is 16.66% and the validation accuracy is 19.47%. After the second epoch, the training accuracy increases slightly to 18.11% and the validation accuracy remains the same. After the third epoch, the training accuracy remains at 18.10% and the validation accuracy is still 19.47%.

The model is then evaluated on the test dataset, and it achieves an accuracy of 25%.

It's important to note that the interpretation is based on the provided code snippet, and additional context or code outside the snippet may be required for a comprehensive understanding of the project.
