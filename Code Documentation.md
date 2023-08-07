# Code Documentation

This code is a script for training and evaluating a deep learning model on a dataset of faulty solar panel images. The script uses PyTorch to train a ResNet50 model with various image augmentations and L2 regularization. It also provides functionality for multiple retraining iterations, early stopping, and saving the best model based on the highest training accuracy achieved.

## Prerequisites

Before running the script, ensure that you have the following prerequisites:

1. Python environment with PyTorch and other required libraries installed.
2. A dataset of faulty solar panel images in two folders: one for training data and another for validation data.
3. The dataset should be organized in class folders, where each folder contains images of a specific class.

## Usage

1. Import the necessary libraries: The script imports several libraries for data handling, model creation, and visualization.

2. User Input: The script prompts the user to enter the number of times they want to retrain the model. This allows you to run multiple training iterations.

3. Hyperparameters and Data Transformations: The script sets up hyperparameters and data transformations based on the value of `num_retrain`. For a single retraining iteration, it uses 3 epochs, a learning rate of 0.0001, and a batch size of 32. For multiple iterations, it uses 20 epochs, a learning rate of 0.0001, and a batch size of 32.

4. Load Data: The script loads the training and validation datasets using PyTorch's `ImageFolder` and creates data loaders to efficiently process batches of data.

5. Model Setup: The script loads the base ResNet50 model with pre-trained weights from ImageNet and replaces the fully connected layer with a new one suited to the number of classes in the dataset.

6. Training Loop: The script performs the training loop for the specified number of epochs. It also implements early stopping based on validation loss to prevent overfitting.

7. Save Best Model and Plots: If `num_retrain > 1`, the script saves the best model based on the highest training accuracy and generates various plots like training accuracy, running loss, validation loss vs. epoch, accuracy vs. epoch, and confusion matrix.

8. Evaluate Model: The script evaluates the model on the training data and calculates the confusion matrix.

9. Print Best Model and Accuracy: The script prints the best model filename and the best accuracy achieved across all retraining iterations.

10. Put Computer to Sleep (Optional): If `num_retrain > 1`, the script puts the computer to sleep after 5 seconds. This seems to be an optional feature and may have been added to allow for automated tasks after multiple retraining iterations.

## Note

- The file paths for training and validation data (`data_path` and `val_data_path`) are currently hardcoded to local paths. You need to update them to the correct paths on your system.

- The code uses a specific transformation pipeline for data augmentation and normalization. If you want to customize the data transformations, you can modify the `transform` variable.

- The class names are hardcoded under `class_names`. If your dataset has different class names, you should update this list accordingly.

- The code uses the Adam optimizer and cross-entropy loss. If you want to experiment with different optimizers or loss functions, you can modify the relevant sections of the code.

- It's worth noting that this documentation does not cover implementation details for the individual image transformations used in the data augmentation pipeline. These transformations are provided by PyTorch's `transforms` module.

- If you plan to use this script on a remote server or in a different environment, ensure you have the necessary libraries installed and update the file paths accordingly.

Overall, this script provides a straightforward way to train a deep learning model on a dataset of faulty solar panel images, visualize the training progress, and save the best model for further use. With multiple retraining iterations, it aims to find the best-performing model based on training accuracy.
