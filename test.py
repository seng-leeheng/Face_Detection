import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

BASE_DIR = r'C:\Users\Hades\Documents\Face-Recognation\runs\detect'

def letterbox(image, target_size=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    return canvas, scale, (pad_left, pad_top)

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def valid_face_box(x1, y1, x2, y2, w, h):
    bw = x2 - x1
    bh = y2 - y1
    if bw < 10 or bh < 10:
        return False
    return True

def draw_label(image, label, x1, y1, color, font=cv2.FONT_HERSHEY_DUPLEX):
    font_scale = 0.6
    font_thickness = 1
    
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = y1 - 8

    cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 6),
                 (text_x + text_size[0] + 2, text_y + 4), color, -1)
    cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 6),
                 (text_x + text_size[0] + 2, text_y + 4), (255, 255, 255), 1)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

def detect_and_draw_faces(image_path, model, model_name, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return None 

    results = model.predict(
        image, 
        imgsz=640,       
        conf=0.2,        
        iou=0.45,       
        verbose=False
    )
    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            conf = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1

            if not valid_face_box(x1, y1, x2, y2, bw, bh):
                continue
            
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': conf
            })

    for det_idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']

        color = (0, 255, 0)
        label = f"Face {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color)

    
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"  Saved to: {output_path}")
    
    print(f"  Detections: {len(detections)} faces\n")
    return image


def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video from {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        preprocessed, scale, (pad_left, pad_top) = letterbox(frame, target_size=640)

        results = model.predict(
            frame, 
            imgsz=640,      
            conf=0.1,      
            iou=0.45,      
            verbose=False
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1

                if not valid_face_box(x1, y1, x2, y2, bw, bh):
                    continue
                
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf
                })

        for det_idx, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['conf']

            color = (0, 255, 0)
            label = f"Face {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1, color)

        if output_path:
            out.write(frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total_frames} | Detections: {len(detections)}")
    
    cap.release()
    if output_path:
        out.release()
        print(f"\nOutput saved to: {output_path}")

def process_all_images(data_dir="Data/train/images", output_dir="detection_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    image_dir = Path(data_dir)
    if not image_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        return

    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {image_file.name}")
        output_path = os.path.join(output_dir, f"detected_{image_file.name}")
        detect_and_draw_faces(str(image_file), output_path)

if __name__ == "__main__":
    print("="*60)
    print("YOLO Face Detection — 4 Model Comparison")
    print("="*60)
    
    image_path = r"C:\Users\Hades\Documents\Face-Recognation\Data\valid\images\f_5_frame33_jpg.rf.1abcf8cf68b8eff4172663fc96d3e0d8.jpg"
    
    models = [
        ("Train_Custom",  "yolo12n"),
        ("Train_Custom2", "yolo11n"),
        ("Train_Custom3", "yolo12s"),
        ("Train_Custom4", "yolo11s"),
    ]
    
    for run_name, arch in models:
        model_path = os.path.join(BASE_DIR, run_name, "weights", "best.pt")
        model = YOLO(model_path)
        display_name = arch
        output_path = os.path.join(
            r"C:\Users\Hades\Documents\Face-Recognation",
            f"detected_{arch}.jpg"
        )
        print(f"\n[{arch}]")
        detect_and_draw_faces(image_path, model, display_name, output_path)
    
    print("\nDone! 4 images saved.")
