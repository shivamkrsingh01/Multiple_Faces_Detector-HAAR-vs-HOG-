# faces_haar.py


import cv2, sys, time, os
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

# Simple centroid tracker for stable IDs (optional)
class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=80):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (sX, sY, eX, eY)) in enumerate(rects):
            inputCentroids[i] = (int((sX + eX) / 2.0), int((sY + eY) / 2.0))

        if len(self.objects) == 0:
            for i in range(inputCentroids.shape[0]):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (r, c) in zip(rows, cols):
                if r in usedRows or c in usedCols:
                    continue
                if D[r, c] > self.maxDistance:
                    continue
                oid = objectIDs[r]
                self.objects[oid] = inputCentroids[c]
                self.disappeared[oid] = 0
                usedRows.add(r); usedCols.add(c)

            for r in set(range(0, D.shape[0])) - usedRows:
                oid = objectIDs[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

            for c in set(range(0, D.shape[1])) - usedCols:
                self.register(inputCentroids[c])

        return self.objects

def main(source=0):
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade)
    if face_cascade.empty():
        print("Failed to load Haar cascade:", cascade); return

    single_image = False
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            print("Cannot read image:", source); return
        single_image = True
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Cannot open camera", source); return

    tracker = CentroidTracker(maxDisappeared=30, maxDistance=80)
    win = "Haar-Face-Only"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    fps_t = time.time()
    while True:
        if single_image:
            frame = img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame"); break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # optional: equalize to improve contrast in tough lighting
        # gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(40,40),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        rects = [(x, y, x+w, y+h) for (x,y,w,h) in faces]
        objects = tracker.update(rects)

        # Draw rectangles and IDs
        for (sX, sY, eX, eY) in rects:
            cv2.rectangle(frame, (sX, sY), (eX, eY), (255, 0, 0), 2)

        for oid, centroid in objects.items():
            cX, cY = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cX, cY), 3, (0,255,0), -1)
            cv2.putText(frame, f"ID:{oid}", (cX-10, cY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        fps = 1.0 / (time.time() - fps_t + 1e-6)
        fps_t = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow(win, frame)

        if single_image:
            out = "haar_faces_output.jpg"
            cv2.imwrite(out, frame); print("Saved", out)
            cv2.waitKey(0); break

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    if not single_image:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(0)
