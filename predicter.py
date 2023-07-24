from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the same size as used during training
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])

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


# Load the saved model
data_path = "/Users/dimaermakov/Faulty_solar_panel_no_desktop"
train_dataset = datasets.ImageFolder(data_path, transform=transform)
num_classes = len(train_dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load("/Users/dimaermakov/Downloads/model.pth"))
model.eval()  # Set the model to evaluation mode

# Function to predict the class of an input image
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        class_index = predicted_class.item()
        class_name = train_dataset.classes[class_index]
    return class_name

# Example usage
image_path = "/Users/dimaermakov/Downloads/image.jpeg"  # Replace with the path to your input image
predicted_class = predict_image(image_path)
print("Predicted class:", predicted_class)
