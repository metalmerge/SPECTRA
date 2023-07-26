import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])

# Load the dataset
data_path = "/Users/dimaermakov/solar-Panel-Dataset"
train_dataset = datasets.ImageFolder(data_path, transform=transform)

# Create data loaders
batch_size = 16
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
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
false_positives = []
num_epochs = 10   # Increase the number of epochs for better accuracy
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

        # Check for false positives during training
        with torch.no_grad():
            model.eval()
            predicted_classes = torch.argmax(outputs, dim=1)
            false_positive_mask = (predicted_classes != labels)
            false_positive_indices = (batch_idx * batch_size) + torch.nonzero(false_positive_mask).squeeze()
            false_positives.extend(false_positive_indices.cpu().numpy().tolist())
            model.train()

            # Print filenames of false positive images
            if(num_epochs > 8):
                for idx in false_positive_indices.cpu().numpy().tolist():
                    print("False positive image filename:", train_dataset.imgs[idx][0])

        # Print progress every 5 batches
        if batch_idx % 5 == 4:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss / 5:.4f}")
            running_loss = 0.0

# Save the trained model
save_model = input("Do you want to save the trained model? (y/n): ").lower()
if save_model == "y":
    torch.save(model.state_dict(), "/Users/dimaermakov/model.pth")
    print("Model saved successfully.")
else:
    print("Model not saved.")

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
def get_false_positive_images(dataset, false_positive_indices):
    false_positive_images = [dataset[idx][0] for idx in false_positive_indices]
    return false_positive_images


# Example usage
false_positive_images = get_false_positive_images(train_dataset, false_positives)

# Example usage
print("Number of false positive images during training:", len(false_positive_images))
image_path = "/Users/dimaermakov/Downloads/solar-Panel-Dataset/Clean/spotless-and-chemical-free-solar-panel-cleaning.jpg"
predicted_class = predict_image(image_path)
print("Predicted class:", predicted_class)