import cv2
from ultralytics import YOLO
import time

# Load your trained model (best weights from training)
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame, stream=True)

    # Draw results on the frame
    for r in results:
        annotated_frame = r.plot()

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
