from ultralytics import YOLO
from PIL import Image
import cv2


def train_model(epoch_num):
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # Load pretrained model
    model.train(
        data="/Users/dimaermakov/Downloads/SolarPanels.2738.yolov8/data.yaml",
        epochs=epoch_num,
        patience=max(1, round(epoch_num / 6)),
        imgsz=640,
        device="cpu",
        verbose=True,
        project="SPECTRA_YOLOv8",
        name=f"model_{epoch_num}",
        weight_decay=0.0005,
    )

    return model


def validate_and_visualize(model, image_array):
    results = model(image_array)  # Run validation
    for r in results:
        im_array = r.plot()  # Plot predictions
        im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image
        im.show()  # Show image


def infer_on_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)  # Run inference on the frame
            annotated_frame = results[0].plot()  # Visualize results
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    best_path = "/Users/dimaermakov/SPECTRA/runs/detect/493_run/weights/best.pt"
    train = int(input("Number of epochs: "))

    if train > 0:
        model = train_model(train)
        model.export(format="onnx")
        # torch.save(model, "infrared_model.pth")

    elif train == 0:
        model = YOLO(best_path)  # Load custom model

        image_array = [
            "/Users/dimaermakov/Downloads/SolarPanelAI.493.yolov8/test/images/468_jpeg.rf.7907654da74862b08c65c83f2f58e689.jpg",
"/Users/dimaermakov/Downloads/SolarPanelAI.493.yolov8/test/images/50_jpeg.rf.a9237d9b3e87abdb25df228965a27c21.jpg",
"/Users/dimaermakov/Downloads/SolarPanelAI.493.yolov8/test/images/56_jpeg.rf.78d9bdd3d8c01bf317b29cea45e36812.jpg",
"/Users/dimaermakov/Downloads/SolarPanelAI.493.yolov8/test/images/82_jpeg.rf.777821a1868148596a8cafa2cc4d7193.jpg",
"/Users/dimaermakov/Downloads/SolarPanelAI.493.yolov8/test/images/96_jpeg.rf.c2a6093ac882077421dc534bca4330c6.jpg",        ]
        validate_and_visualize(model, image_array)

        video_path = "/Users/dimaermakov/Downloads/Thermography Solar Panel Video.mp4"
        # infer_on_video(model, video_path)


if __name__ == "__main__":
    main()
