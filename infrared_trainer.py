from ultralytics import YOLO
from PIL import Image
import cv2
import torch
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# best_path = '/Users/dimaermakov/SPECTRA/runs/detect/train3/weights/best.pt'
# model = YOLO(best_path)  # load a custom model
#TODO:
#figure out how to usee model onnx

# Train the model
test = model.train(data='/Users/dimaermakov/Downloads/SolarPanelAI.v1i.yolov8/data.yaml', epochs=30, patience=10, imgsz=640, verbose=True, device='cpu', weight_decay=.0005)


# results = model(['/Users/dimaermakov/Downloads/SolarPanelAI.v1i.yolov8/test/images/82_jpeg.rf.777821a1868148596a8cafa2cc4d7193.jpg', '/Users/dimaermakov/Downloads/Solar Panel.v1-solardataset.yolov8/test/images/Thermal_Solar_508_jpg.rf.c982e0e42970e383ccd77c8d4fa36f86.jpg'])  # results list

# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category

# Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image
    # print(r.boxes,r.boxes.xyxy, r.probs)

video_path = '/Users/dimaermakov/Downloads/videoplayback (1).webm'

# Run inference on the source
# cap = cv2.VideoCapture(video_path)

# Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()

model.export(format='onnx')
torch.save(model, "infrared_model.pth")
# torch.save(model, "savetesting2.pt")
