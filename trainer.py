#estimated completion time: 2 hours 30 minutes
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

ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define data transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),   # Randomly crop and resize images
    transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
    transforms.RandomVerticalFlip(),     # Randomly flip images vertically
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust color
    transforms.RandomRotation(10),       # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),               # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])

# Load the dataset
# data_path = "/Users/dimaermakov/solar-Panel-Dataset"
data_path = "/Users/dimaermakov/Downloads/Faulty_solar_panel"
train_dataset = datasets.ImageFolder(data_path, transform=transform)

# Create data loaders
batch_size = 16 #8 or 16 bigger better
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the base model (ResNet50) with the pre-trained weights using the 'weights' parameter
base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_classes = len(train_dataset.classes)
in_features = base_model.fc.in_features
base_model.fc = nn.Linear(in_features, num_classes)

# Move the model to the device (GPU or CPU)
model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = .0001 #0.001 or 0.0001 lower better
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

step_size = 5  # Reduce the learning rate every 5 epochs
gamma = 0.1    # Reduce the learning rate by a factor of 0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Create a confusion matrix
true_labels = train_dataset.targets
predicted_labels = []

# Training loop
false_positive_filenames = []
false_positives = []

num_epochs = 20 # 10 or 20 bigger better
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            model.eval()
            predicted_classes = torch.argmax(outputs, dim=1)
            false_positive_mask = (predicted_classes != labels)
            false_positive_indices = (batch_idx * batch_size) + torch.nonzero(false_positive_mask).flatten()
            false_positives.extend(false_positive_indices.cpu().numpy().tolist())
            model.train()

            # Print filenames of false positive images
            for idx in false_positive_indices.cpu().numpy().tolist():
                false_positive_filenames.append(train_dataset.imgs[idx][0])

        # Print progress every 5 batches
        # if batch_idx % 5 == 4:
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss / 5:.4f}")
        running_loss = 0.0
    scheduler.step()

# Function to predict the class of an input image
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        class_index = predicted_class.item()
        class_name = train_dataset.classes[class_index]
    return class_name

# Example usage
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
plt.savefig("/Users/dimaermakov/SPECTRA/server/static/confusion_matrix.png")
plt.show()

# Display the images of false positives
num_images_to_display = 5
plt.figure(figsize=(12, 6))
for i, idx in enumerate(false_positives[:num_images_to_display]):
    image_path = train_dataset.imgs[idx][0]
    image = Image.open(image_path)
    plt.subplot(1, num_images_to_display, i + 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"False Positive {i+1}")
plt.savefig("/Users/dimaermakov/SPECTRA/server/static/false_positive_images.png")
plt.show()


image_path = "/Users/dimaermakov/solar-Panel-Dataset/Clean/example1.jpeg"
predicted_class = predict_image(image_path)
print("Predicted class:", predicted_class)

# Print the list of false positive filenames
# print("Filenames of False Positive Images:")
# for filename in false_positive_filenames:
#     print(filename)

# Save the trained model
save_model = input("Do you want to save the trained model? (y/n): ").lower()
if save_model == "y":
    torch.save(model.state_dict(), "/Users/dimaermakov/model.pth")
    print("Model saved successfully.")
else:
    print("Model not saved.")