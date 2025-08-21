import cv2
import os
person_name = input("Enter your name: ").strip()
save_dir = f"dataset/{person_name}"
os.makedirs(save_dir, exist_ok=True)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
count = 0
print(f"📷 Collecting eye images for {person_name}. Press 'q' to stop.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        eye = frame[y:y+h, x:x+w]
        eye = cv2.resize(eye, (64, 64))
        cv2.imwrite(f"{save_dir}/eye_{count}.jpg", eye)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} #{count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0,255,0), 2)
    cv2.imshow("Eye Dataset Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 200:
        break
cap.release()
cv2.destroyAllWindows()
print(f"✅ Saved {count} eye images in {save_dir}")
