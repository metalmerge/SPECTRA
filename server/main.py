import torch.nn as nn
from flask import Flask, request
from flask_cors import CORS
from flask import render_template
from torchvision import datasets, transforms, models
from pathlib import Path
import torch
from PIL import Image

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image values
    ]
)

# Load the trained model
model = models.resnet50(pretrained=False)
num_classes = 6  # Replace this with the number of classes in your dataset
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model_filename = "/Users/dimaermakov/models_folder/model_97.11.pth"
# model_filename = "/Users/dimaermakov/models_folder/model_92.35.pth"
model.load_state_dict(torch.load(model_filename, map_location=device))
model.to(device)
model.eval()

app = Flask(__name__, static_url_path="/static")
cors = CORS(app)


@app.route("/")
def upload():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    img_files = request.files.getlist("file")  # Get a list of all the uploaded images
    results = []

    for img_file in img_files:
        img = Image.open(img_file)
        img = (
            transform(img).unsqueeze(0).to(device)
        )  # Add batch dimension and move to the device
        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output, 1)
            class_index = predicted_class.item()
            # Replace 'dataset.classes' with the list of class names in your dataset
            class_name = [
                "Bird-drop",
                "Clean",
                "Dusty",
                "Electrical-damage",
                "Physical-Damage",
                "Snow-Covered",
            ][class_index]
            results.append(class_name)

    return results


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
