# Multi Input Deep Learning Model Repository

This repository contains code for a multi-input deep learning model that combines a Convolutional Neural Network (CNN) and a Multi-Layer Perceptron (MLP). The model architecture involves concatenating the output of these two components and passing it through another MLP.
This code automaticaly builds a multi-input type model based on your model configuration. Read more about multi modal deep learning models in the review paper:

ieeexplore.ieee.org/abstract/document/8103116

## Files Description

1. **cfg.py**:
   - This file reads the configuration of the model. It is imported in `model.py` to configure the model setup.

2. **model.py**:
   - Imports `cfg.py` to automatically build a multi-input model by combining a CNN and an MLP. It concatenates their outputs and passes them through another MLP for further processing.

3. **run_models.py**:
   - Reads text configurations and runs `model.py` for each of the model configurations stored in the `cfg` folder. It automates the process of running multiple model configurations.

## Usage

1. Configure your model settings in the `cfg.py` file.
2. Run `run_models.py` to execute the defined model configurations and training processes.

## Instructions

1. **Setting up configurations**:
   - The `cfg.py` file is used to read and customize the model architecture, hyperparameters, and training settings. You should text file of model configuration in `cfg` folder. Sample text files are there and you can use them.

2. **Running the models**:
   - Execute `run_models.py` to automatically run the models based on the configurations provided in the `cfg` folder.

3. **Training and Evaluation**:
   - Utilize the defined model architecture to train and evaluate multi-input deep learning models for your specific use case.

## Requirements

- Python
- TensorFlow/Keras
- numpy
- sklearn
Feel free to explore and enhance this repository for your multi-input deep learning model experiments!
