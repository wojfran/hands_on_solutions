# Deep Learning Style Transfer

## Overview

This project implements a style transfer technique using TensorFlow and Keras. The method leverages the VGG19 pretrained model to blend the content of one image with the style of another, generating visually appealing results that resemble artistic styles.

## Contents

- Introduction to Style Transfer
- Prerequisites
- Dataset Preparation
- Content and Style Representations
- Model Building
- Style and Content Extraction
- Gradient Descent Optimization
- Total Variation Loss
- Results

## Introduction to Style Transfer

Style transfer is a fascinating technique that allows you to merge the content of one image with the artistic style of another. This is achieved by manipulating the intermediate layers of a convolutional neural network (CNN), allowing the model to extract both content and style representations.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Other necessary libraries as specified in `requirements.txt`

## Dataset Preparation

The project uses two images:
1. A content image: The image whose content will be preserved.
2. A style image: The image whose artistic style will be applied to the content image.

Make sure to place these images in a directory accessible by the script.

## Content and Style Representations

The VGG19 network architecture is utilized to define content and style representations. The intermediate layers of the model provide essential features for both content and style. 

- **Content layers**: Capture the high-level features of the image.
- **Style layers**: Capture the textures and patterns.

## Model Building

Using the Keras functional API, a model is created to extract the outputs from the desired intermediate layers of the VGG19 network. This allows for the separation of style and content information from the input images.

## Style and Content Extraction

A custom model, `StyleContentModel`, is defined to extract style and content tensors. The extracted style is represented by Gram matrices, while the content is represented by the values of the intermediate feature maps.

## Gradient Descent Optimization

The main optimization process is achieved using a weighted combination of style and content losses. The Adam optimizer is employed to update the image variable iteratively, refining it to match the desired style and content.

### Style and Content Loss Calculation

The losses are computed based on the differences between the extracted outputs from the generated image and the target images. 

## Total Variation Loss

To reduce high-frequency artifacts in the generated image, a total variation loss is implemented. This regularization technique promotes spatial coherence in the output image.

## Results

After running the optimization process, the final stylized image is generated and saved. The results can be visually inspected to see how effectively the content and style have been blended.

## Exercise Instruction

Go through TensorFlow’s Style Transfer tutorial. It is a fun way to generate art using Deep Learning.

## Conclusion

This project provides a hands-on experience with neural style transfer using deep learning techniques. By utilizing a pretrained CNN model, you can create stunning visual art by combining different images' content and style.
