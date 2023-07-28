# Author: metalmerge
# Estimated training time: 1-2 hours depending early stoping

import os
import ssl
import torch
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from PIL import Image

num_epochs = 20  # 10 or 20, bigger is better
learning_rate = .0001  # 0.001 or 0.0001, lower is better
batch_size = 16  # 8 or 16, bigger is better
weight_decay = 0.001  # Adjust this value based on your needs

# Ignore SSL certificate errors for data download
ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),   # Randomly crop and resize images
    transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
    transforms.RandomVerticalFlip(),     # Randomly flip images vertically
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust color
    transforms.RandomRotation(10),       # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),               # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])

# Load the dataset
data_path = "/Users/dimaermakov/Downloads/Faulty_solar_panel_Train"
train_dataset = datasets.ImageFolder(data_path, transform=train_transform)
val_data_path = "/Users/dimaermakov/Downloads/Faulty_solar_panel_Validation"
val_dataset = datasets.ImageFolder(val_data_path, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)

# Download the ResNet-50 weights
resnet50_weights = models.resnet50(pretrained=True)

# Save the weights to a file
local_weights_path = "/Users/dimaermakov/SPECTRA/resnet50.pth"
torch.save(resnet50_weights.state_dict(), local_weights_path)

# Later, to load the saved weights
base_model = models.resnet50()
base_model.load_state_dict(torch.load(local_weights_path))

# Fine-tune the model
for param in base_model.parameters():
    param.requires_grad = False

in_features = base_model.fc.in_features
base_model.fc = nn.Linear(in_features, num_classes)
model = base_model.to(device)

# Move the model to the device (GPU or CPU)
model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

step_size = 5  # Reduce the learning rate every 5 epochs
gamma = 0.1  # Reduce the learning rate by a factor of 0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
patience = 5  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
counter = 0

# Create a confusion matrix
true_labels = train_dataset.targets
predicted_labels = []
accuracies = []
running_losses = []
val_losses = []


# Training loop
print(f"Training started for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)
    epoch_loss = running_loss / len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Calculate L2 regularization term (optional, add to loss)
        l2_regularization = 0.0
        for param in model.parameters():
            l2_regularization += torch.norm(param, 2)
        loss += weight_decay * l2_regularization
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted_classes = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted_classes == labels).sum().item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches}, "
              f"Loss: {loss.item():.4f}, Accuracy: {(correct / total) * 100:.2f}%")

    val_loss = calculate_validation_loss(model, criterion, val_loader, device)
    epoch_accuracy = 100 * correct / total
    accuracies.append(epoch_accuracy)
    running_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss)  # Append the validation loss for this epoch


    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches}, "
              f"Loss: {loss.item():.4f}, Accuracy: {(correct / total) * 100:.2f}%")

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
def predict_image(image_path):
    model.eval()  # Set the model to evaluation mode
    image = Image.open(image_path)
    image = val_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        class_index = predicted_class.item()
        class_name = train_dataset.classes[class_index]
    return class_name

# Example usage
# save_model = input("Do you want to save the trained model? (y/n): ").lower()
# if save_model == "y":
final_train_accuracy = accuracies[-1]
# model_filename = f"/Users/dimaermakov/model_{final_train_accuracy:.2f}.pth"
model_filename = f"/Users/dimaermakov/night_model_{final_train_accuracy:.2f}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved successfully as {model_filename}.")
# Plot training accuracy and loss
epochs = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs, accuracies, 'g', label='Accuracy of Training data')
plt.plot(epochs, running_losses, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc=0)

# Save the plot if the user wants to
# save_model = input("Do you want to save the training_accuracy_loss? (y/n): ").lower()
# if save_model == "y":
plt.savefig("/Users/dimaermakov/SPECTRA/server/training_images/training_accuracy_loss.png")

plt.show()

# Plot training and validation loss
plt.figure()
plt.plot(epochs, running_losses, 'g', label='Loss of Training Data')
plt.plot(epochs, val_losses, 'r', label='Loss of Validation Data')  # Use val_losses instead of val_loss
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc=0)

# Save the plot if the user wants to
# save_model = input("Do you want to save the training_validation_loss? (y/n): ").lower()
# if save_model == "y":
plt.savefig("/Users/dimaermakov/SPECTRA/server/training_images/training_validation_loss.png")

plt.show()

plt.figure()
plt.plot(range(1, num_epochs + 1), accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.grid()
# save_model = input("Do you want to save the accuracy_plot? (y/n): ").lower()
# if save_model == "y":
plt.savefig("/Users/dimaermakov/SPECTRA/server/training_images/accuracy_plot.png")
plt.show()

# Plot a grid of individual image examples
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
num_examples = 16  # Number of image examples to show
rows = int(np.ceil(num_examples / 4))
plt.figure(figsize=(15, 15))

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
        filenames = [val_dataset.samples[batch_idx * batch_size + j][0] for j in range(batch_size)]
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

# save_model = input("Do you want to save the image_examples? (y/n): ").lower()
# if save_model == "y":
plt.savefig("/Users/dimaermakov/SPECTRA/server/training_images/image_examples.png")
plt.show()

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

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# save_model = input("Do you want to save the confusion_matrix? (y/n): ").lower()
# if save_model == "y":
plt.savefig("/Users/dimaermakov/SPECTRA/server/training_images/confusion_matrix.png")
plt.show()

os.system("sleep 5 && pmset sleepnow") # Put the computer to sleep after 5 seconds