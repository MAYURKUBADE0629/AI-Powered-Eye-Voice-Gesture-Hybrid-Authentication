import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 🔹 Load trained model
model = load_model("eye_model.h5")

# 🔹 Dynamically load labels from dataset folders
dataset_dir = "dataset"
labels = sorted(os.listdir(dataset_dir))   # folder names = person names
print("✅ Loaded Labels:", labels)

# Haar Cascade for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye_resized = cv2.resize(eye, (64, 64))
        eye_resized = eye_resized.astype("float32") / 255.0
        eye_resized = np.expand_dims(eye_resized, axis=-1)  # (64,64,1)
        eye_resized = np.expand_dims(eye_resized, axis=0)   # (1,64,64,1)

        pred = model.predict(eye_resized, verbose=0)
        label = labels[np.argmax(pred)]
        confidence = np.max(pred) * 100

        # Draw rectangle & name on main frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show each detected eye separately
        eye_display = cv2.resize(eye, (200, 200))  # bigger for clarity
        cv2.imshow(f"{label}'s Eye", eye_display)

    cv2.imshow("Eye Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
