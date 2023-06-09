# Deep Neural Network for Software Defect Prediction
This repository contains code and resources for training a Deep Neural Network (DNN) on the CM1 software defect dataset. The DNN is built using Keras with the TensorFlow backend, and feature selection is performed using a Genetic Algorithm to improve accuracy.

## Dataset
The CM1 dataset is used for training the DNN. It is a well-known dataset in the field of software defect prediction and contains a collection of software metrics that can be used to predict the presence of defects in software modules.

## Requirements
Make sure you have the following dependencies installed:

Python 3.x
Keras
TensorFlow
Genetic Algorithm library (add specific library name if used)
You can install the dependencies by running the following command:
```
pip install -r requirements.txt
```

## Usage
Clone the repository:
```
git clone [https://github.com/m-aliabbas/GAwithNN.git]
```

## Results
The trained Deep Neural Network achieves an accuracy of 89% on the training dataset and 87% on the test dataset. The results of the Genetic Algorithm applied to the Neural Network and the training history of the DNN are provided in the repository.

## Conclusion
In this project, we have demonstrated the use of a Deep Neural Network for software defect prediction using the CM1 dataset. The combination of Genetic Algorithm for feature selection and the DNN model has resulted in high accuracy on both the training and test datasets.
