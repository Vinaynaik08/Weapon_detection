from ultralytics import YOLO

def main():
    # Load your previously trained model (continue training)
    model = YOLO("runs/detect/train/weights/best.pt")

    # Continue training
    model.train(
        data="dataset/data.yaml",   # path to your dataset yaml
        epochs=50,                 # maximum epochs (total, including the first 50 you ran)
        imgsz=640,                  # image size
        batch=16,                   # adjust based on GPU memory
        workers=0,                  # set to 0 on Windows to avoid multiprocessing issues
        patience=20,                # early stopping if no improvement for 20 epochs
        save=True,                  # save checkpoints
        save_period=10,             # save weights every 10 epochs
        verbose=True                # show training logs
    )

    # Validate the trained model
    model.val()

    # (Optional) Run prediction on test set to visualize results
    model.predict(
        source="dataset/test/images",   # folder with test images
        save=True,
        conf=0.25
    )

if __name__ == "__main__":
    main()