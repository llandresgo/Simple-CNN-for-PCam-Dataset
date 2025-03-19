# Simple CNN for PCam Dataset

This repository contains the implementation of a simple Convolutional Neural Network (CNN) for training and validating on the PCam dataset (PatchCamelyon dataset). The goal is to classify microscopic histopathology images.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- pandas
- Pillow

You can install the required packages using pip:

```sh
pip install torch torchvision pandas pillow
```

## Dataset

The PCam dataset consists of microscopic histopathology images. The dataset should be organized in csv files where each row contains the image path and the corresponding label.

- `data/train_labels.csv`: Training data
- `data/validation_labels.csv`: Validation data
- `data/test_labels.csv`: Test data

## Model

The model is a simple CNN with three convolutional layers followed by two fully connected layers. The architecture is as follows:

- Conv2d(3, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
- Conv2d(32, 64, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
- Conv2d(64, 128, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
- Flatten
- Linear(128 * 12 * 12, 256) -> ReLU
- Linear(256, 1) -> Sigmoid

## Training

To train the model, use the script provided:

```sh
python CNN_histo_train.py
```

The script will train the model for a specified number of epochs (5 in this case) and print the training and validation loss for each epoch.
The model will then be save as cnn_model_histo.pth in the directory in which the script is run

## Training

To predict with the model, use the script provided:

```sh
python CNN_histo_predict.py

## Author

Andres Gonzalez
