import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Hades\Documents\Face-Recognation\runs\detect\Train_Custom3\weights\best.pt"
VIDEO_PATH = "n.mp4"
CONF_THRESHOLD = 0.25  
TARGET_HEIGHT = 640

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

current_frame = 0
paused = False

cv2.namedWindow("YOLOv12 Player", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv12 Player", 1920, 1080)

def on_trackbar(val):
    global current_frame
    current_frame = val

cv2.createTrackbar("Seek", "YOLOv12 Player", 0, total_frames - 1, on_trackbar)

def draw_predictions(frame, results):
    total_boxes = 0
    filtered_boxes = 0
    for r in results:
        for box in r.boxes:
            total_boxes += 1
            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                filtered_boxes += 1
                continue

            cls_id = int(box.cls)
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
    return frame

while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, TARGET_HEIGHT))

    results = model(frame)
    frame = draw_predictions(frame, results)

    cv2.imshow("YOLOv12 Player", frame)
    cv2.setTrackbarPos("Seek", "YOLOv12 Player", current_frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE pause
        paused = not paused
    elif key == 83 or key == 2555904:  # → Forward
        current_frame = min(current_frame + 30, total_frames - 1)
    elif key == 81 or key == 2424832:  # ← Backward
        current_frame = max(current_frame - 30, 0)

    if not paused:
        current_frame += 1
        if current_frame >= total_frames:
            break

cap.release()
cv2.destroyAllWindows()