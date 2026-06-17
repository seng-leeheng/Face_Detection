# YOLO Face Detection

This repository contains a set of scripts for training, testing, and running YOLO-based models for Face Detection.

## Installation

Ensure you have the required dependencies installed (mainly `ultralytics` and `opencv-python`). You can install them via pip:

```bash
pip install ultralytics opencv-python numpy
```

## How to Use

### 1. Training the Model (`train_yolo.py`)
This script trains a YOLO model using a custom dataset defined in `Data/data.yaml`. By default, it loads `yolo12s.pt`.

To train the model, simply run:
```bash
python train_yolo.py
```

**Adding Custom Arguments:**
The `model.train()` function in `train_yolo.py` accepts many arguments. You can easily add or modify arguments to fit your specific needs (e.g., changing the number of epochs, image size, batch size, or learning rate). 

Example modifications:
```python
model.train(
    data="Data/data.yaml",
    imgsz=640,          # Image size
    epochs=100,         # Number of epochs
    batch=16,           # Batch size
    device=0,           # Set to 'cpu' if you don't have a GPU
    optimizer="AdamW",
    # Add any other Ultralytics YOLO arguments you need!
)
```

### 2. Testing & Comparing Models (`test.py`)
This script loads an image and compares the detection results across 4 different YOLO models (e.g., YOLOv11 and YOLOv12 variants). It will output the detected faces and draw bounding boxes.

To test the models:
```bash
python test.py
```
*Note: Make sure to update the `image_path` inside the script to point to a valid image on your computer.*

### 3. Video Inference Player (`video.py`)
This script processes a video and displays real-time face detection with an interactive video player.