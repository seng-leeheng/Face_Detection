from ultralytics import YOLO

def train():
    model = YOLO("yolo12s.pt")

    model.train(
        data="Data/data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        amp=True,
        name="Train_Custom",
    )

if __name__ == "__main__":
    train()
