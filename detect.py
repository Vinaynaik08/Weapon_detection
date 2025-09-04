import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Path to the image you want to test
image_path = "dataset/test/images/picture-3355-_png.rf.47f1f947ce2a17f0b2f6dab203d7edc3.jpg"   # <-- replace with your image filename

# Run inference
results = model(image_path)

# Show the results
for r in results:
    annotated_frame = r.plot()  # Draw bounding boxes on image
    cv2.imshow("Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Optionally, save the output
cv2.imwrite("output.jpg", annotated_frame)
