from ultralytics import YOLO

def main():
    # Load your trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Validate the model
    metrics = model.val(
        data="dataset/data.yaml",
        save=True,
        workers=0   # 👈 important on Windows
    )

    print(metrics)  # precision, recall, mAP, etc.

if __name__ == "__main__":   # 👈 required for Windows
    main()
