# Author: metalmerge
# estimated training time: 136.62 minutes
import os
import ssl
import time
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from auto_augment import AutoAugment
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models

# User input: Number of times to retrain the model
num_retrain = int(input("How many times do you want to retrain the model? "))
testing = input("Is this a test run? (y/n) ").lower()
# Initialize variables to store best accuracy and model filename
best_accuracy = 0.0
best_hyperparameters = {}
best_model_filename = ""

# Disable SSL verification for downloading data (can be optional)
ssl._create_default_https_context = (
    ssl._create_default_https_context
) = ssl._create_unverified_context

if testing == "y":
    num_epochs_values = [1]
    learning_rate_values = [0.001]
    batch_size_values = [55]
else:
    num_epochs_values = [20]
    learning_rate_values = [0.0001]
    batch_size_values = [32, 48, 55, 64]

# Loop for multiple retraining iterations
for retrain_index in range(num_retrain):
    # Training loop for the specified number of epochs
    for num_epochs in num_epochs_values:
        for learning_rate in learning_rate_values:
            for batch_size in batch_size_values:
                weight_decay = (
                    0.001  # L2 regularization strength, adjust based on needs
                )
                step_size = 5  # Step size for learning rate scheduling
                gamma = 0.1  # Factor to reduce learning rate
                patience = 5  # Number of epochs to wait for improvement before stopping
                best_val_loss = float("inf")
                counter = 0

                # Set the device to use GPU if available, otherwise use CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Define data transformations for image augmentation
                transform = transforms.Compose(
                    [
                        transforms.Resize(
                            (224, 224)
                        ),  # Resize images to (224, 224) before augmentations
                        AutoAugment(),
                        # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally with a probability of 0.5
                        # transforms.RandomVerticalFlip(),  # Randomly flip the image vertically with a probability of 0.5
                        # transforms.RandomRotation(
                        #     10
                        # ),  # Randomly rotate the image by a maximum of 10 degrees
                        # transforms.RandomPerspective(),  # Random perspective transformation
                        # transforms.RandomAdjustSharpness(
                        #     0.3
                        # ),  # Randomly adjust sharpness with a factor of 0.3
                        # transforms.RandomApply(
                        #     [transforms.RandomPerspective(distortion_scale=0.3, p=0.5)],
                        #     p=0.1,
                        # ),  # Stronger perspective transformation with a probability of 0.1
                        # transforms.RandomAffine(
                        #     degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)
                        # ),
                        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),  # Normalize the image with mean and standard deviation
                    ]
                )

                # Load the training and validation datasets
                data_path = "/Users/dimaermakov/Downloads/Faulty_solar_panel_Train"
                train_dataset = datasets.ImageFolder(data_path, transform=transform)

                val_data_path = (
                    "/Users/dimaermakov/Downloads/Faulty_solar_panel_Validation"
                )
                val_dataset = datasets.ImageFolder(val_data_path, transform=transform)

                # Create data loaders to efficiently process batches of data during training and evaluation
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                # Load the base model (ResNet50) with the pre-trained weights from ImageNet
                base_model = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V1
                )
                num_classes = len(train_dataset.classes)
                in_features = base_model.fc.in_features
                base_model.fc = nn.Linear(in_features, num_classes)

                # Move the model to the selected device
                model = base_model.to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )

                # Set up the learning rate scheduler to reduce the learning rate over time
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma
                )

                # Function to calculate validation loss during model evaluation
                def calculate_validation_loss(model, criterion, val_loader, device):
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                    return val_loss / len(val_loader)

                # Initialize lists to store training statistics

                # Record the start time for training duration calculation
                start_time = time.time()
                print("Training started...")

                true_labels = train_dataset.targets
                accuracies = []
                running_losses = []
                val_losses = []
                predicted_labels = []
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    # Loop through batches of the training data
                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(device), labels.to(device)

                        # Reset gradients to zero before computing backward pass
                        optimizer.zero_grad()

                        # Forward pass through the model and compute loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()

                        # Calculate L2 regularization term and add it to the loss
                        l2_regularization = 0.0
                        for param in model.parameters():
                            l2_regularization += torch.norm(param, 2)
                        loss += weight_decay * l2_regularization

                        # Update model parameters using the optimizer
                        optimizer.step()
                        running_loss += loss.item()

                        # Calculate training accuracy
                        _, predicted_classes = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted_classes == labels).sum().item()

                        # Print batch-level training progress
                        print(
                            f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {(correct / total) * 100:.2f}%"
                        )

                    # Calculate validation loss at the end of each epoch
                    val_loss = calculate_validation_loss(
                        model, criterion, val_loader, device
                    )
                    epoch_accuracy = correct / total
                    accuracies.append(epoch_accuracy)
                    running_losses.append(running_loss / len(train_loader))
                    val_losses.append(val_loss)

                    # Print epoch-level training progress
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {(correct / total) * 100:.2f}%, Learning Rate: {learning_rate}, Batch Size: {batch_size}"
                    )

                    # Early stopping: Check if validation loss has improved, if not, increment counter
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print(
                                f"Early stopping: No improvement for {patience} epochs."
                            )
                            num_epochs = epoch + 1
                            break

                    running_loss = 0.0
                    scheduler.step()

                    # Function to predict the class of an input image

                    # Record the end time for training duration calculation
                    end_time = time.time()
                    total_time = end_time - start_time / 60
                    print(
                        f"Training completed in {(end_time - start_time) / 60:.2f} minutes."
                    )

                    # Example usage

                    # Get the final training accuracy

                    final_train_accuracy = accuracies[-1]

                    # Save the best model and update best_accuracy if applicable
                    if final_train_accuracy > best_accuracy:
                        best_accuracy = final_train_accuracy
                        best_hyperparameters = {
                            "num_epochs": num_epochs,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

                        if testing != "y":
                            best_model_filename = f"/Users/dimaermakov/models_folder/model_{best_accuracy:.2f}_{learning_rate}_{batch_size}.pth"
                            torch.save(model.state_dict(), best_model_filename)

                            # Combined plot for accuracy, running loss, and validation loss vs. epoch
                            epochs = range(1, num_epochs + 1)
                            plt.figure()
                            plt.plot(
                                epochs,
                                accuracies,
                                "g",
                                label="Accuracy of Training data",
                            )
                            plt.plot(
                                epochs,
                                running_losses,
                                "r",
                                label="Loss of Training data",
                            )
                            plt.plot(
                                epochs, val_losses, "b", label="Loss of Validation Data"
                            )
                            plt.title(
                                "Training data accuracy, running loss, and validation loss"
                            )
                            plt.xlabel("Epoch")
                            plt.ylabel("Value")
                            plt.legend(loc=0)
                            plt.tight_layout()

                            plt.savefig(
                                f"/Users/dimaermakov/Downloads/night_images/training_combined_plot_{best_accuracy:.2f}_{learning_rate}_{batch_size}.png"
                            )

                            # Save plot for accuracy vs. epoch
                            plt.figure()
                            plt.plot(range(1, num_epochs + 1), accuracies, marker="o")
                            plt.xlabel("Epoch")
                            plt.ylabel("Accuracy")
                            plt.title("Accuracy vs. Epoch")
                            plt.grid()
                            plt.tight_layout()

                            plt.savefig(
                                f"/Users/dimaermakov/Downloads/night_images/accuracy_plot_{best_accuracy:.2f}_{learning_rate}_{batch_size}.png"
                            )

                            # Plot a grid of individual image examples
                            class_names = [
                                "Bird-drop",
                                "Clean",
                                "Dusty",
                                "Electrical-damage",
                                "Physical-Damage",
                                "Snow-Covered",
                            ]
                            num_examples = 32
                            rows = int(np.ceil(num_examples / 4))
                            plt.figure(figsize=(15, 15))
                            val_filenames = [
                                val_dataset.samples[i][0]
                                for i in range(len(val_dataset))
                            ]

                            # Loop through batches of validation data to show image examples
                            for batch_idx, (images, labels) in enumerate(val_loader):
                                for i in range(
                                    min(
                                        num_examples - batch_idx * batch_size,
                                        batch_size,
                                    )
                                ):
                                    ax = plt.subplot(
                                        rows, 4, i + 1 + batch_idx * batch_size
                                    )
                                    image = images[i].cpu().numpy().transpose(1, 2, 0)
                                    image = image * [0.229, 0.224, 0.225] + [
                                        0.485,
                                        0.456,
                                        0.406,
                                    ]
                                    image = np.clip(image, 0, 1)
                                    plt.imshow(image)
                                    plt.title("Actual: " + class_names[labels[i]])
                                    plt.gca().axes.yaxis.set_ticklabels([])
                                    plt.gca().axes.xaxis.set_ticklabels([])
                                    start_index = batch_idx * batch_size
                                    filenames = val_filenames[
                                        start_index + i : start_index + i + 1
                                    ]
                                    for filename in filenames:
                                        plt.xlabel(os.path.basename(filename))
                                    with torch.no_grad():
                                        model.eval()
                                        inputs = images[i].unsqueeze(0).to(device)
                                        outputs = model(inputs)
                                        _, predicted_class = torch.max(outputs, 1)
                                        class_index = predicted_class.item()
                                        predicted_class_name = class_names[class_index]

                                    color = (
                                        "green"
                                        if class_names[labels[i]]
                                        == predicted_class_name
                                        else "red"
                                    )
                                    plt.ylabel(
                                        "Predicted: " + predicted_class_name,
                                        fontdict={"color": color},
                                    )

                                    if batch_idx * batch_size + i >= num_examples - 1:
                                        break

                            plt.tight_layout()

                            plt.savefig(
                                f"/Users/dimaermakov/Downloads/night_images/image_examples_{best_accuracy:.2f}_{learning_rate}_{batch_size}.png"
                            )

                            # Evaluate the model on the training data and calculate the confusion matrix
                            model.eval()
                            predicted_labels = []
                            with torch.no_grad():
                                for inputs, labels in train_loader:
                                    inputs = inputs.to(device)
                                    outputs = model(inputs)
                                    _, predicted_classes = torch.max(outputs, 1)
                                    predicted_labels.extend(
                                        predicted_classes.cpu().numpy().tolist()
                                    )

                            true_labels = np.array(true_labels)
                            predicted_labels = np.array(predicted_labels)

                            conf_matrix = confusion_matrix(
                                true_labels, predicted_labels
                            )
                            plt.figure(figsize=(10, 8))
                            sns.heatmap(
                                conf_matrix,
                                annot=True,
                                fmt="d",
                                cmap="Blues",
                                xticklabels=train_dataset.classes,
                                yticklabels=train_dataset.classes,
                            )
                            plt.xlabel("Predicted")
                            plt.ylabel("True")
                            plt.title("Confusion Matrix")
                            plt.tight_layout()

                            plt.savefig(
                                f"/Users/dimaermakov/Downloads/night_images/confusion_matrix_{best_accuracy:.2f}_{learning_rate}_{batch_size}.png"
                            )

                # Print best model and accuracy after all retraining iterations (if applicable)
                print(f"Best model saved successfully as {best_model_filename}.")
                print(f"Best accuracy achieved: {best_accuracy:.2f}%.")
                print("Best Hyperparameters:", best_hyperparameters)

# Put the computer to sleep after 5 seconds (only for last retraining iteration)
if testing == "y":
    os.system(
        'osascript -e \'display notification "Your program is done running" with title "Program Finished"\''
    )
else:
    os.system("sleep 5 && pmset sleepnow")
