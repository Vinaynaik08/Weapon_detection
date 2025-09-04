from ultralytics import YOLO

def main():
    # 1) Load model
    model = YOLO("yolov8s.pt")   # change to yolov8m.pt or yolov8l.pt for higher accuracy

    # 2) Train
    model.train(
        data="dataset/data.yaml",  # your dataset yaml
        epochs=50,
        imgsz=640,
        batch=16,
        workers=0   # 👈 IMPORTANT on Windows (avoid multiprocessing issues)
    )

    # 3) Validate
    model.val()

    # 4) Predict on test images
    model.predict(
        source="dataset/test/images",
        save=True,
        conf=0.25
    )

if __name__ == "__main__":   # 👈 Required for Windows
    main()
