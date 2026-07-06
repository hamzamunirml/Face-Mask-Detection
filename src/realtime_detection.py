"""
Phase 8 - Real-Time Face Mask Detection
Uses OpenCV's Haar Cascade for face detection and the trained model for
mask classification. Draws live bounding boxes with confidence scores and FPS.

Usage:
    python realtime_detection.py --model ../models/mobilenetv2_best.h5
"""

import cv2
import argparse
import time
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = (128, 128)
LABELS = {0: "Mask", 1: "No Mask"}
COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}


def run_detection(model_path, camera_index=0):
    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check camera index / permissions.")

    prev_time = time.time()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, IMG_SIZE)
            face_array = face_resized.astype("float32") / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            pred_prob = model.predict(face_array, verbose=0)[0][0]
            label_idx = int(pred_prob >= 0.5)
            confidence = pred_prob if label_idx == 1 else 1 - pred_prob

            label_text = f"{LABELS[label_idx]}: {confidence*100:.1f}%"
            color = COLORS[label_idx]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Mask Detection - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../models/mobilenetv2_best.h5",
                        help="Path to trained .h5 model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    run_detection(args.model, args.camera)
