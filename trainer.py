import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])
print(transform)

# Load the dataset
data_path = "/Users/dimaermakov/Downloads/solar-Panel-Dataset"
train_dataset = datasets.ImageFolder(data_path, transform=transform)

# Create data loaders
batch_size = 16   # Decrease batch size for slower computers
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define the path to the locally downloaded pre-trained weights
local_weights_path = "/Users/dimaermakov/SPECTRA/resnet18-f37072fd.pth"

# Load the base model (ResNet18) with the pre-trained weights
base_model = models.resnet18(pretrained=False)
base_model.load_state_dict(torch.load(local_weights_path))


# Replace the fully connected layer (classifier) with a new one for our task
num_classes = len(train_dataset.classes)
in_features = base_model.fc.in_features
base_model.fc = nn.Linear(in_features, num_classes)

# Move the model to the device (GPU or CPU)
model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001   # Use a relatively larger learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 10   # Increase the number of epochs for better accuracy
print(num_epochs)
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

        # Print progress every 50 batches
        if batch_idx % 5 == 49:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss / 50:.4f}")
            running_loss = 0.0

# Save the trained model
torch.save(model.state_dict(), "/Users/dimaermakov/SPECTRA/model.pth")

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
image_path = "/Users/dimaermakov/Downloads/example1.jpeg"  # Replace with the path to your input image
predicted_class = predict_image(image_path)
print("Predicted class:", predicted_class)