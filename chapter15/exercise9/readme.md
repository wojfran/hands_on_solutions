# SketchRNN Classification Model

## Overview

This project implements a classification model for the SketchRNN dataset using TensorFlow and Keras. The model is designed to classify sketches drawn by users into predefined categories. The dataset consists of sketches stored in TensorFlow's TFRecord format, which allows for efficient data loading and preprocessing.

## Contents

- Introduction
- Prerequisites
- Dataset Preparation
- Model Architecture
- Training
- Evaluation
- Results
- Exercise Instruction
- Conclusion

## Introduction

The SketchRNN dataset contains millions of sketches drawn by users, categorized into various classes. This project focuses on building a neural network model that can learn to classify these sketches based on their content. By leveraging TensorFlow and Keras, we can create a pipeline that efficiently processes the data, trains a model, and evaluates its performance.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Other necessary libraries as specified in `requirements.txt`

## Dataset Preparation

The dataset is downloaded from TensorFlow Datasets and consists of training and evaluation files in TFRecord format. The following steps are performed to prepare the dataset:

1. Download the dataset from the TensorFlow repository.
2. Parse the TFRecord files to extract sketches, their lengths, and corresponding class labels.
3. Create TensorFlow data pipelines to handle batching and shuffling.

## Model Architecture

The model is built using a sequential architecture that combines convolutional layers and LSTM layers to effectively capture the spatial and temporal features of the sketches. The architecture includes:

- Conv1D layers for feature extraction.
- Batch normalization for improved training stability.
- LSTM layers for sequential data processing.
- A Dense layer with softmax activation for classification.

## Training

The model is compiled with the following settings:

- Loss function: Sparse categorical crossentropy.
- Optimizer: SGD with momentum and weight decay.
- Metrics: Accuracy and top-k categorical accuracy.

The model is trained on the cropped training dataset with early stopping and learning rate reduction callbacks to optimize performance.

## Evaluation

After training, the model's performance is evaluated on the test set. The mean top-5 accuracy is computed to assess the model's ability to predict the correct class among the top five predictions.

## Results

The model's predictions are displayed alongside the corresponding sketches. Each sketch is accompanied by the top-5 predicted classes and their probabilities.

## Exercise Instruction

- Train a classification model for the SketchRNN dataset available in TensorFlow Datasets.
- Implement the model using the provided code as a guide.
- Experiment with different architectures, hyperparameters, and data augmentation techniques to improve the model's performance.

## Conclusion

This project demonstrates how to train a classification model for sketch recognition using the SketchRNN dataset. The use of TensorFlow and Keras allows for efficient data handling and model training, resulting in a functional sketch classification model.
