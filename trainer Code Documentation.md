# Faulty Solar Panel Classifier Training Script

## Author: metalmerge

## Estimated Training Time: 136.62 minutes

This script trains a deep learning model to classify images of faulty solar panels. It uses the ResNet50 architecture with pre-trained weights from ImageNet and applies various hyperparameters to find the best model.

### Importing Required Libraries

The script begins by importing necessary libraries for data manipulation, deep learning, and visualization.

### User Input

The script prompts the user for the number of times to retrain the model and whether it's a test run. The user's input determines the number of epochs, learning rates, and batch sizes to be tested.

### Hyperparameters Setup

Hyperparameters are defined based on user input. The script includes options for a test run with fewer iterations and specific values for faster testing.

### Model Training Loop

The script iterates through different hyperparameters combinations for multiple retraining iterations. Within each iteration, the model trains for a specified number of epochs with varying learning rates and batch sizes. The training loop includes the following steps:

1. Device Selection: Selects GPU if available, otherwise uses CPU.
2. Data Transformations: Defines data transformations including resizing, AutoAugment, and normalization.
3. Data Loading: Loads training and validation datasets using `torchvision.datasets.ImageFolder`.
4. Model Initialization: Loads a ResNet50 model with pre-trained weights and modifies the output layer for the number of classes.
5. Model Transfer: Moves the model to the selected device.
6. Loss and Optimizer: Defines the loss function (CrossEntropyLoss) and optimizer (Adam).
7. Learning Rate Scheduler: Sets up a scheduler to reduce learning rate during training.
8. Training Loop: Iterates through training batches, computes losses, performs backpropagation, and updates model parameters.
9. Validation Loss: Calculates validation loss at the end of each epoch.
10. Early Stopping: Monitors validation loss for early stopping if improvement stalls.
11. Learning Rate Scheduling: Updates the learning rate using the scheduler.
12. Training Metrics: Tracks training accuracy, loss, and validation loss.
13. Model Evaluation: Evaluates the model on validation data and saves best model and accuracy.

### Result Visualization

After training iterations are complete, the script visualizes the results:

1. Accuracy, Running Loss, and Validation Loss Plot: Plots accuracy and losses over epochs.
2. Accuracy vs. Epoch Plot: Displays accuracy trend over epochs.
3. Image Examples: Plots a grid of image examples from the validation set.
4. Confusion Matrix: Generates and displays the confusion matrix based on model predictions.
5. Final Best Model and Accuracy: Prints the best model filename and highest achieved accuracy along with the corresponding hyperparameters.

### Script Completion

If it's a test run, the script displays a notification. Otherwise, it puts the computer to sleep after a 5-second delay.

## Conclusion

This script automates the process of training a deep learning model for classifying faulty solar panels. By systematically varying hyperparameters, the script identifies the best model configuration for the given task.

**Note:** Ensure that the necessary libraries (`torch`, `numpy`, `seaborn`, `PIL`, `auto_augment`, etc.) are installed before running the script.
