# CS669 Deep Learning Assignment 3: Dog vs Cat vs Bird Classifier

## Image Classification Project: Cat vs Dog vs Bird

This repository contains code for an image classification project that classifies images of cats, dogs, and birds. The code utilizes different deep learning models, optimization techniques, and enhancements to achieve the best classification results. Below, you will find a comprehensive overview of the project, the methods used, and the models implemented.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Experiments and Models](#experiments-and-models)
- [Results](#results)

## Project Overview

The goal of this project is to train deep learning models for the classification of images into three classes: *Cat, **Dog, and **Bird*. The dataset used consists of images with dimensions of 32x32 pixels.

### Key Features:
- Multiple models and optimization strategies were experimented with, including both pretrained and non-pretrained architectures.
- Performance evaluation is conducted using a validation set, and confusion matrices are plotted for better understanding of model performance.
- Models are trained using the *PyTorch* library.

## Setup Instructions

### Prerequisites:
To get started with this project, you need to have the following dependencies installed:

- Python 3.x
- PyTorch
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas
- NumPy

You can install these dependencies using pip:

bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas numpy


The images are 32x32 pixels, and the training and validation sets are organized into class-specific folders. The test set consists of unlabelled images.

## Experiments and Models

### 1. *Baseline Classifier*
A simple baseline model using a fully connected neural network (FCNN) was trained to classify the images.

### 2. *SGD to ADAM Optimizer Switch*
The model performance was evaluated using the Stochastic Gradient Descent (SGD) optimizer, and then optimized further by switching to the Adam optimizer.

### 3. *Learning Rate Scheduler (Adam + Cosine Annealing with Warmup)*
A learning rate scheduler was used in combination with the Adam optimizer to adjust the learning rate during training, helping to improve convergence.

### 4. *Learning Rate Finder*
This technique was used to identify the ideal learning rate for training. Although it can be useful, it was not trusted in this experiment due to inconsistencies in the results.

### 5. *Adam + Dropout*
The Adam optimizer was used alongside dropout to prevent overfitting by randomly disabling neurons during training.

### 6. *Adam + Dropout + Weight Decay*
Building on the previous experiment, weight decay was introduced to regularize the model further.

### 7. *Adam + Dropout + Weight Decay + Batch Normalization*
Batch normalization was introduced to stabilize training by normalizing activations across the mini-batch. This helps speed up convergence.

### 8. *ResNet-18 Structure (No Transfer Learning)*
The ResNet-18 architecture was used as the base model without leveraging pretrained weights. This model was trained from scratch.

### 9. *ResNet-18 Pretrained (Transfer Learning)*
The ResNet-18 model was fine-tuned using pretrained weights from ImageNet. This significantly improved the model's performance.

### 10. *ResNet-50 Pretrained (Transfer Learning)*
A deeper ResNet-50 model with pretrained weights was used for transfer learning, yielding even better results compared to ResNet-18.

## Results

Here are the performance metrics and results from training various models:

- *ResNet-18 (Pretrained)* achieved an accuracy of approximately 83-84%.
- *ResNet-50 (Pretrained)* showed improved performance with a higher accuracy compared to ResNet-18 for 87.92%.

The models were evaluated based on the validation loss and accuracy, and confusion matrices were plotted to evaluate the predictions.
