# Code Documentation

## Author: metalmerge

This script performs training and evaluation of a deep learning model using PyTorch's ResNet50 architecture for image classification. The script includes data loading, data augmentation, model training, early stopping, evaluation, and visualization of results.

### Libraries Used

- `os`: The operating system interface, used for file operations.
- `ssl`: Used to disable SSL verification for downloading data.
- `time`: Used for timing the execution.
- `torch`: The PyTorch deep learning framework.
- `numpy`: Used for numerical operations.
- `seaborn`: Used for creating visualizations.
- `torch.nn`: Contains various neural network layers, loss functions, etc.
- `torch.optim`: Contains optimization algorithms.
- `matplotlib.pyplot`: Used for creating plots.
- `torch.optim.lr_scheduler`: Contains learning rate schedulers.
- `PIL`: The Python Imaging Library, used for opening and manipulating images.
- `auto_augment`: Custom module for image augmentation.
- `torch.utils.data.DataLoader`: Used for loading data efficiently.
- `sklearn.metrics`: Contains confusion matrix for evaluation.
- `torchvision.datasets`: Contains standard datasets for vision tasks.
- `torchvision.transforms`: Contains data transformations.

### Variables

- `validation_dataset_path`: Path to the validation dataset.
- `training_dataset_path`: Path to the training dataset.
- `num_retrain`: Number of times to retrain the model.
- `testing`: Indicates whether this is a test run (`"y"` or `"n"`).
- Various hyperparameter values and settings for model training.

### Functions and Execution

1. The script prompts the user for the number of times to retrain the model (`num_retrain`) and whether it's a test run (`testing`).

2. Based on the test run status, the script defines hyperparameter values for number of epochs, learning rate, and batch size.

3. The script enters a loop for retraining iterations. Within each iteration:
   - Hyperparameters for training are set based on iteration values.
   - Data transformations are defined using `transforms.Compose`.
   - Training and validation datasets are loaded using `datasets.ImageFolder`.
   - The base model (ResNet50) is loaded with pre-trained weights from ImageNet.
   - Model training loop is executed, including loss computation, optimization, and early stopping based on validation loss.
   - Various statistics and visualizations are generated for analysis.

4. The best model and corresponding statistics are printed after all retraining iterations.

5. If `testing` is `"y"`, a system notification is displayed. Otherwise, the script puts the computer to sleep after 5 seconds.

### Usage

1. Run the script in a Python environment.
2. Follow the prompts to provide retraining iterations and test run status.
3. The script will perform model training, evaluation, and visualization.

## Note

- This script assumes that you have the required datasets in the specified paths.
- Make sure to have the necessary libraries (`os`, `ssl`, `time`, `torch`, `numpy`, `seaborn`, `matplotlib`, `PIL`, `auto_augment`, `sklearn`, `torchvision`) installed in your environment.
- Adjust the script as needed for your specific dataset and hyperparameters.
