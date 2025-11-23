import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0] 
    for det in results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        cls = int(det.cls[0])
        label = model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#next steps: group each object into a priority queue using c++ (pybind11)


cap.release()
cv2.destroyAllWindows()
