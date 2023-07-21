import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
data_path = "/Users/dimaermakov/SPECTRA/Faulty_solar_panel_no_desktop"
train_dataset = datasets.ImageFolder(data_path, transform=transform)

# Create data loaders
batch_size = 16   # Decrease batch size for slower computers
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Initialize the model
num_classes = len(train_dataset.classes)
print(num_classes)
model = SimpleCNN(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001   # Reduce learning rate for slower training
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 5   # Reduce the number of epochs for slower training
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
        if batch_idx % 50 == 49:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss / 50:.4f}")
            running_loss = 0.0

# Save the trained model
torch.save(model.state_dict(), "/Users/dimaermakov/SPECTRA/")