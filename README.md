# Pneumonia-classification-using-deep-learninng


## Overview

This project focuses on classifying pneumonia from chest X-ray images using deep learning techniques. Two models were implemented - a custom Convolutional Neural Network (CNN) and a transfer learning model based on MobileNetV2. The dataset was sourced from Kaggle.

## Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- PIL (Python Imaging Library)
- Scikit-Learn

## Installation

1. **Install the required Python packages:**
   ```bash
   pip install tensorflow numpy opencv-python matplotlib pillow scikit-learn
## Install Kaggle API for dataset retrieval:
pip install kaggle
Configure your Kaggle API credentials by placing your kaggle.json file in the ~/.kaggle/ directory.
## Download the dataset from Kaggle:
kaggle datasets download -d andrewmvd/pediatric-pneumonia-chest-xray
## Extract the dataset:
unzip pediatric-pneumonia-chest-xray.zip
## Usage

    Run the notebook in a Jupyter or Colab environment.
    The notebook performs the following steps:
        Imports necessary libraries.
        Downloads and extracts the dataset.
        Preprocesses and augments the images.
        Builds and trains a custom CNN model.
        Implements transfer learning using MobileNetV2.
        Evaluates the models and displays performance metrics.

## Model Files

    The custom CNN model is saved as custom_model.h5.
    The MobileNetV2 model is saved as mobilenet_model.h5.

## Evaluation

To evaluate the models on the test dataset, utilize the evaluate_model function with the path to the test folder and the trained model.
evaluate_model('/content/Pediatric Chest X-ray Pneumonia/test/', mobilenet_model)

## Author
Abhishek kumar
