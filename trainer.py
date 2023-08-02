# Author: metalmerge
# estimated training time: 136.62 minutes
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import ssl
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import ssl

num_retrain = int(input("How many times do you want to retrained the model? "))
best_accuracy = 0.0
best_model_filename = ""
ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

for retrain_index in range(num_retrain):
    if num_retrain > 1:
        num_epochs = 20  # 10 or 20, bigger is better
        learning_rate = .0001  # 0.001 or 0.0001, lower is better
        batch_size = 32  # 8 or 16, bigger is better
    else:
        num_epochs = 1  # 10 or 20, bigger is better
        learning_rate = .01  # 0.001 or 0.0001, lower is better
        batch_size = 2  # 8 or 16, bigger is better
    weight_decay = 0.001  # Adjust this value based on your needs
    step_size = 5  # Reduce the learning rate every 5 epochs
    gamma = 0.1  # Reduce the learning rate by a factor of 0.1
    patience = 5  # Number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')
    counter = 0

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224) before augmentations
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally with a probability of 0.5
    transforms.RandomVerticalFlip(),    # Randomly flip the image vertically with a probability of 0.5
    transforms.RandomRotation(10),      # Randomly rotate the image by a maximum of 10 degrees
    transforms.RandomPerspective(),     # Random perspective transformation
    transforms.RandomAdjustSharpness(0.3),  # Randomly adjust sharpness with a factor of 0.3
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.3, p=0.5)], p=0.1),  # Stronger perspective transformation with a probability of 0.1
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Random affine transformation (rotation, translation, scaling)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
])
    # Load the dataset
    data_path = "/Users/dimaermakov/Downloads/Faulty_solar_panel_Train"
    train_dataset = datasets.ImageFolder(data_path, transform=transform)
    val_data_path = "/Users/dimaermakov/Downloads/Faulty_solar_panel_Validation"
    val_dataset = datasets.ImageFolder(val_data_path, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the base model (ResNet50) with the pre-trained weights using the 'weights' parameter
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_classes = len(train_dataset.classes)
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the device (GPU or CPU)
    model = base_model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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

    # Create a confusion matrix
    true_labels = train_dataset.targets
    predicted_labels = []

    # Training loop
    accuracies = []
    running_losses = []
    val_losses = []


    start_time = time.time()
    print("Training started...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = running_loss / len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate L2 regularization term (optional, add to loss)
            l2_regularization = 0.0
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2)
            loss += weight_decay * l2_regularization

            # Calculate accuracy
            _, predicted_classes = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted_classes == labels).sum().item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {(correct / total) * 100:.2f}%")

        val_loss = calculate_validation_loss(model, criterion, val_loader, device)
        epoch_accuracy = 100 * correct / total
        accuracies.append(epoch_accuracy)
        running_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)  # Append the validation loss for this epoch

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping: No improvement for {patience} epochs.")
                num_epochs = epoch + 1
                break

        running_loss = 0.0
        scheduler.step()

    # Function to predict the class of an input image
    end_time = time.time()
    total_time = end_time - start_time / 60
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")

    # Example usage

    final_train_accuracy = accuracies[-1]
    if final_train_accuracy > best_accuracy:
        best_accuracy = final_train_accuracy
        if num_retrain > 1:
            best_model_filename = f"/Users/dimaermakov/models_folder/model_{best_accuracy:.2f}.pth"
            torch.save(model.state_dict(), best_model_filename)

        # model_filename = f"/Users/dimaermakov/night_model_{final_train_accuracy:.2f}.pth"
        # torch.save(model.state_dict(), model_filename)
        print(f"Model saved successfully as {best_model_filename}.") # Plot training accuracy and loss
        epochs = range(1, num_epochs + 1)

        plt.figure()
        plt.plot(epochs, accuracies, 'g', label='Accuracy of Training data')
        plt.plot(epochs, running_losses, 'r', label='Loss of Training data')
        plt.title('Training data accuracy and loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend(loc=0)
        plt.tight_layout()
        if num_retrain > 1:
            plt.savefig(f"/Users/dimaermakov/Downloads/night_images/training_accuracy_loss_{best_accuracy:.2f}.png")

        # Plot training and validation loss
        plt.figure()
        plt.plot(epochs, running_losses, 'g', label='Loss of Training Data')
        plt.plot(epochs, val_losses, 'r', label='Loss of Validation Data')  # Use val_losses instead of val_loss
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(f"/Users/dimaermakov/Downloads/night_images/training_validation_loss_{best_accuracy:.2f}.png")


        plt.figure()
        plt.plot(range(1, num_epochs + 1), accuracies, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.grid()
        plt.tight_layout()
        if num_retrain > 1:
            plt.savefig(f"/Users/dimaermakov/Downloads/night_images/accuracy_plot_{best_accuracy:.2f}.png")


        # Plot a grid of individual image examples
        class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

        num_examples = 32  # Number of image examples to show
        rows = int(np.ceil(num_examples / 4))
        plt.figure(figsize=(15, 15))

        # Extract the filenames for the entire validation dataset
        val_filenames = [val_dataset.samples[i][0] for i in range(len(val_dataset))]

        for batch_idx, (images, labels) in enumerate(val_loader):
            for i in range(min(num_examples - batch_idx * batch_size, batch_size)):
                ax = plt.subplot(rows, 4, i + 1 + batch_idx * batch_size)
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize the image
                image = np.clip(image, 0, 1)  # Clip to [0, 1] range in case of numerical errors
                plt.imshow(image)
                plt.title("Actual: " + class_names[labels[i]])
                plt.gca().axes.yaxis.set_ticklabels([])
                plt.gca().axes.xaxis.set_ticklabels([])

                # Get the filenames for the current batch
                start_index = batch_idx * batch_size
                filenames = val_filenames[start_index + i:start_index + i + 1]  # Extract the filename for the current image
                for filename in filenames:
                    plt.xlabel(os.path.basename(filename))  # Show only the filename without the full path

                with torch.no_grad():
                    model.eval()  # Set the model to evaluation mode
                    inputs = images[i].unsqueeze(0).to(device)
                    outputs = model(inputs)
                    _, predicted_class = torch.max(outputs, 1)
                    class_index = predicted_class.item()
                    predicted_class_name = class_names[class_index]

                color = "green" if class_names[labels[i]] == predicted_class_name else "red"
                plt.ylabel("Predicted: " + predicted_class_name, fontdict={'color': color})

                if batch_idx * batch_size + i >= num_examples - 1:
                    break

        plt.tight_layout()
        if num_retrain > 1:
            plt.savefig(f"/Users/dimaermakov/Downloads/night_images/image_examples_{best_accuracy:.2f}.png")

        model.eval()
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted_classes = torch.max(outputs, 1)
                predicted_labels.extend(predicted_classes.cpu().numpy().tolist())

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if num_retrain > 1:
            plt.savefig(f"/Users/dimaermakov/Downloads/night_images/confusion_matrix_{best_accuracy:.2f}.png")

print(f"Best model saved successfully as {best_model_filename}.")
print(f"Best accuracy achieved: {best_accuracy:.2f}%.")
if num_retrain > 1:
    os.system("sleep 5 && pmset sleepnow") # Put the computer to sleep after 5 seconds