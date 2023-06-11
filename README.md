# ML2Project
 
 Introduction
 First, I tried to train a neural network for object recognition with X-Ray data from baggages(Model.ipynb). The goal was to detect illicit objects like knives, razor blades or handguns. Since I had problems preparing the images and labels for modeling, I decided to change the dataset and work with the COCO dataset. In the ML2Project.ipynb you can find the code for the object detection with the COCO dataset.
 
 Project Goal/Motivation:
The goal of this project is to train a machine learning model to classify images using the COCO (Common Objects in Context) dataset. The motivation behind the project is to develop a model that can accurately identify objects in images and improve its performance through training and validation.

Data Collection:
The COCO dataset is loaded using the TensorFlow Datasets (tfds) library. The dataset is specifically the 'coco/2017' split from the training set. The dataset is shuffled, and a subset of 10,000 data points is selected. Each data point consists of an image and object labels. The images are resized to a target size of (128, 128) pixels.

Modeling:
The model used is a sequential model built using the Keras API. It consists of several layers, including Convolutional and Dense layers. The model architecture includes convolutional layers for feature extraction, max-pooling layers for downsampling, and fully connected layers for classification. The model is compiled with an Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric. I tried different amount of layers and neurons, to find out which configuration performs best. Also I tried to choose the best configuration of the different Hyperparameters like learningrate, beta1, beta2 and L2-Regularization or dropout.

Interpretation:
During the training process, the model goes through three epochs. Let's analyze the results of each epoch and the final test accuracy in more detail:

Epoch 1:
Training: The model is trained on the training dataset, consisting of 249 steps (or batches). The average loss during this epoch is 4.7204, and the accuracy achieved is 0.1666 (16.66%).
Validation: The validation dataset is evaluated at the end of each epoch. The validation loss is 4.0687, and the validation accuracy is 0.1947 (19.47%).

Epoch 2:
Training: The model continues training for the second epoch. The average loss decreases to 4.0404, but the training accuracy only slightly improves to 0.1811 (18.11%).
Validation: The validation loss decreases slightly to 3.8749, but the validation accuracy remains the same at 0.1947 (19.47%).

Epoch 3:
Training: In the third epoch, the loss further decreases to 3.9238, but the training accuracy remains similar to the previous epoch at 0.1810 (18.10%).
Validation: The validation loss continues to decrease to 3.7917, but the validation accuracy remains the same at 0.1947 (19.47%).
After the training process, the model is evaluated on the test dataset, which is not shown in the code snippet. The test accuracy achieved is 0.25 (25%).

Overall, the model's performance during training does not show significant improvement. The training accuracy remains relatively low, and the validation accuracy stays constant throughout the epochs. This suggests that the model may not be effectively learning from the data or that the dataset may be challenging for the model to generalize to unseen examples.

The low test accuracy of 0.25 indicates that the model's performance on unseen data is also limited. It may be necessary to further analyze the dataset, model architecture, and training process to identify potential issues and improve the model's performance.

Problems:
I tried to use more data in training, validation and testing of the model, set the input shape of the images to (256,256) instead of (128, 128) and also tried to train with different batch sizes. But the program always collapsed and i had to start with over again. 
