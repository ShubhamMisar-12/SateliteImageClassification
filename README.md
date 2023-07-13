# Satellite Image Prediction Project

[Link to Kaggle Noebook]("https://www.kaggle.com/code/shubhammisar/satellite-image-classification-pytorch-cnn?scriptVersionId=133741513")

This project aims to develop a model for predicting categories in satellite images. The code provided uses a Convolutional Neural Network (CNN) architecture to train and evaluate the model.

![Image](https://storage.googleapis.com/kaggle-datasets-images/1544742/2546969/587dbc48162374bca68d9f8c10299a90/dataset-cover.jpg?t=2021-08-21-19-05-27)

## Folder Structure

.

├── Data

│ └── archive.zip

├── LICENSE

├── README.md

├── requirements.txt

└── satellite-image-classification.ipynb

## Project Description

The `satellite-image-classification.ipynb` notebook contains the code for the satellite image prediction project. The notebook uses PyTorch to implement a CNN model for image classification. The provided code performs the following tasks:

1. Imports necessary modules and libraries for data processing and model training.
2. Prepares the data by applying transformations to the satellite image dataset.
3. Creates training and testing datasets using a random split.
4. Defines the CNN model architecture using the `CNNModel` class.
5. Trains the model using the training dataset and evaluates its performance on the testing dataset.
6. Prints the training and testing accuracies for each epoch.
7. Displays a plot showing the training and testing accuracies over epochs.
8. Generates a confusion matrix to visualize the model's predictions.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Open the `satellite-image-classification.ipynb` notebook in Jupyter or any compatible notebook environment.
3. Ensure that you have the required dependencies installed (refer to the notebook for details).
4. Use the provided local data or replace it with your own dataset.
5. Run the code cells in the notebook to train and evaluate the model.
6. Modify the code and experiment with different parameters as needed.

## Model Architecture

The model architecture is defined using a CNN model in PyTorch. Here's a summary of the model:

```bash
CNNModel(
(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu1): ReLU()
(maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu2): ReLU()
(maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu3): ReLU()
(flatten): Flatten(start_dim=1, end_dim=-1)
(fc1): Linear(in_features=16384, out_features=64, bias=True)
(relu4): ReLU()
(fc2): Linear(in_features=64, out_features=4, bias=True)
(softmax): Softmax(dim=1)
)

```

## License

This project is licensed under the MIT license.
