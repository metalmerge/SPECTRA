# Code Documentation

## Author: metalmerge

This script performs various tasks related to object detection using the YOLOv8 model from the `ultralytics` library. It includes functions to train the model, validate and visualize its predictions on images, and perform inference on a video stream.

### Libraries Used

- `ultralytics`: A deep learning library that provides implementations for various computer vision tasks, including object detection.
- `PIL`: The Python Imaging Library, used for opening and manipulating images.
- `cv2`: OpenCV, a computer vision library, used for video capture, processing, and display.
- `os`: The operating system interface, used for file operations.

### Variables

- `YAML_PATH`: Path to the YAML configuration file for model training.
- `best_pt_model_path`: Path to the pre-trained model checkpoint.
- `test_image_folder`: Path to the folder containing test images.
- `test_video_path`: Path to the input video file for inference.

### Functions

1. `train_model(epoch_num)`: Trains the YOLOv8 model for a specified number of epochs.

   - `epoch_num`: Number of epochs for training.
   - Returns: Trained YOLOv8 model.

2. `validate_and_visualize(model, image_folder)`: Validates the model's predictions on images and visualizes the results.

   - `model`: Trained YOLOv8 model.
   - `image_folder`: Folder containing images for validation.

3. `infer_and_save_video(model, video_path, output_path)`: Performs inference on a video stream and saves the annotated video.

   - `model`: Trained YOLOv8 model.
   - `video_path`: Path to the input video file.
   - `output_path`: Path to save the annotated output video.

4. `main()`: Main function that drives the script's execution.

### Execution

1. The script starts by prompting the user for the number of epochs to train the model. If `train > 0`, the script trains the model, exports it in ONNX format, and puts the system to sleep after a delay.

2. If `train == 0`, the script loads a custom model checkpoint and performs the following steps:
   - Validates and visualizes the model's predictions on test images.
   - Performs inference on the specified video file, annotates each frame, and saves the annotated video.

3. The script uses the `__name__` check to ensure that it's being run as the main module before executing the `main()` function.

### Usage

1. Run the script in a Python environment.
2. Follow the prompts to specify the number of epochs for training or choose to perform inference on test images and a video.
3. The script will display validation images with annotations and save an annotated video if applicable.

## Note

- This script assumes that you have the required image and video files in the specified paths.
- Make sure to have the necessary libraries (`ultralytics`, `PIL`, `cv2`) installed in your environment.
