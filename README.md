# Multiple Face Detection and Comparison 👀  

This project compares **two popular face detection models** —  
**Haar Cascade** and **HOG (Histogram of Oriented Gradients)** —  
to analyze their performance in detecting **multiple faces** in real time.

---

## 🧠 Features
- Detects **multiple faces** using webcam or video input.  
- Compares **speed, accuracy, and detection quality** between Haar and HOG.  
- Displays bounding boxes around detected faces in real time.  
- Easy to extend with **MobileNet SSD** using the provided `prototxt` file.

---

## 📁 Project Structure
```
Multiple_faces/
│
├── haar_multi_face_tracker.py     # Face detection using Haar Cascade
├── hog_face_detector.py           # Face detection using HOG
├── MobileNetSSD_deploy.prototxt   # Model configuration for SSD (optional)
```

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install opencv-python dlib numpy imutils
```

If you face issues installing `dlib`, run:
```bash
pip install cmake
pip install dlib
```

---

## 🚀 How to Run

### 1. Run Haar Cascade Model
```bash
python haar_multi_face_tracker.py
```

### 2. Run HOG Model
```bash
python hog_face_detector.py
```

Make sure your webcam is connected or change the video source inside the code.

---

## 📊 Comparison Summary

| Model            | Technique                               | Speed (FPS) | Accuracy | Works Best For            |
|------------------|-----------------------------------------|-------------|----------|---------------------------|
| **Haar Cascade** | Haar-like features + Cascade Classifier | Fast        | Moderate | Well-lit, frontal faces   |
| **HOG Detector** | Histogram of Oriented Gradients         | Slower      |   High   | Varying angles & lighting |

---

## 🧩 Notes
- You can modify both scripts to test on **images or video files**.  
- For higher accuracy or object detection, integrate SSD with the provided `prototxt`.

---

## 🏁 Output
- Displays real-time bounding boxes around detected faces.  
- Compares model performance visually.

---


