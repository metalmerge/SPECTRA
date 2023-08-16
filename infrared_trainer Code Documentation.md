# YOLOv8 Solar Panel Detection Script

## Author: metalmerge

This script utilizes the YOLOv8 object detection model to detect solar panels in images and videos. It provides functionality for training the model, validating on images, and performing inference on videos.

## Dependencies

Ensure you have the following dependencies installed before running the script:

- `ultralytics`: Library for YOLO model training and inference
- `PIL`: Python Imaging Library for image processing
- `cv2`: OpenCV library for image and video processing
- `os`: Python standard library for interacting with the operating system

## Training Model

The `train_model` function loads a pretrained YOLOv8 model, trains it on solar panel detection data, and saves the trained model. It takes the number of epochs as input and utilizes the following parameters:

- `data`: Path to the data configuration file
- `epochs`: Number of training epochs
- `patience`: Early stopping patience
- `imgsz`: Input image size
- `device`: Device to run the training on ("cpu" or "cuda")
- `verbose`: Print training progress
- `project`: Project name
- `name`: Model name
- `weight_decay`: L2 regularization strength

## Validating and Visualizing

The `validate_and_visualize` function takes a trained model and an array of image paths, runs inference on the images, and visualizes the detection results using PIL. Detected bounding boxes are plotted on the images.

## Inferring and Saving Video

The `infer_and_save_video` function performs inference on a video file, annotates the frames with bounding boxes, and saves the annotated video. It takes the paths to the input video and the output video as inputs.

## Main Function

The `main` function is the entry point of the script. It offers two modes of operation based on user input:

1. Training Mode: Prompts the user for the number of epochs to train the model. After training, it exports the model in ONNX format and puts the computer to sleep.
2. Inference Mode: Loads a trained model, runs inference on sample images, and performs video inference.

## Usage

1. Ensure the required dependencies are installed.
2. Run the script. If you want to train the model, enter the number of epochs when prompted. If you want to perform inference on images and videos, select a sample mode.

## Conclusion

This script demonstrates the use of the YOLOv8 model for solar panel detection. It provides options for training the model and performing inference on images and videos. The script's flexibility allows users to adapt it to their specific object detection tasks.

**Note:** Make sure to replace the file paths with your actual paths and adjust other settings as needed for your environment and dataset.
