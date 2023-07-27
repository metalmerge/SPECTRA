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
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
])

# Load the trained model
model = models.resnet50(pretrained=False)
num_classes = 6  # Replace this with the number of classes in your dataset
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("/Users/dimaermakov/model.pth", map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)
cors = CORS(app) #Request will get blocked otherwise on Localhost

@app.route("/")
def upload():
    return render_template("upload.html")
 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = Image.open(request.files['file'])
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    with torch.no_grad():
        output = model(img)
        _, predicted_class = torch.max(output, 1)
        class_index = predicted_class.item()
        # Replace 'dataset.classes' with the list of class names in your dataset
        class_name = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered'][class_index]
    return class_name

if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080)

    # def predict():
    # img = PILImage.create(request.files['file'])
    # label,_,probs = learn.predict(img)
    # return f'{label} ({torch.max(probs).item()*100:.0f}%)'