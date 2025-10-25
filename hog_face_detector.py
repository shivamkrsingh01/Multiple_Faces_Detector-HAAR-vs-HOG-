# hog_face_detector.py

import cv2
import dlib
import time

# initialize HOG-based face detector
hog_detector = dlib.get_frontal_face_detector()

# open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("HOG Face Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HOG Face Detector", 1280, 720)

fps_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_detector(gray)

    for i, face in enumerate(faces):
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    fps = 1.0 / (time.time() - fps_time + 1e-6)
    fps_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("HOG Face Detector", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
