import cv2
from ultralytics import YOLO
import time
import math

def compute_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def compute_speed(dist, time1, time2):
    if time2 - time1 == 0:
        return 0
    return dist / (time2 - time1)


model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

obj_hist = {}

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)[0]

    for b in results.boxes:

        if b.id is None:
            continue
        obj_id = int(b.id[0])


        seconds = time.perf_counter()
        x_center, y_center, w, h = b.xywh[0]
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        if obj_id not in obj_hist:
            obj_hist[obj_id] = []

        obj_hist[obj_id].append((seconds, float(x_center), float(y_center)))
        speed = 0


        if len(obj_hist[obj_id]) > 1:
            t1, x1c, y1c = obj_hist[obj_id][-2]
            t2, x2c, y2c = obj_hist[obj_id][-1]

            dist = compute_distance(x1c, y1c, x2c, y2c)
            speed = compute_speed(dist, t1, t2)

        label = f"{results.names[int(b.cls[0])]} ID:{obj_id} Speed:{int(speed)}px/s"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) == ord('q'):
        break
