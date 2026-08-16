import torch
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # GPU Check
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print(" CUDA not available. Using CPU.")

    #  Load a YOLOv8 model
    model = YOLO("yolo12m.pt")  # You can also try 'yolov8n.pt', 'yolov8m.pt', etc.

    # 🗂 Train the model
    model.train(
        data="data.yaml",
        epochs=150,  # More epochs to allow thorough learning
        imgsz=640,  # Higher resolution improves accuracy (check VRAM)
        batch=8,  # Automatically use max batch size GPU can handle
        device=0,  # GPU 0
        lr0=0.002,  # Slightly higher initial LR for faster convergence
        lrf=0.01,  # Final LR (lower = better fine-tuning)
        momentum=0.937,  # Momentum helps stabilize convergence
        weight_decay=0.0005,  # Regularization (reduce overfitting)
        warmup_epochs=5,  # Helps with stability at start
        warmup_bias_lr=0.1,  # Bias learning rate warmup
        close_mosaic=15,  # Turn off aggressive augmentation near end
        degrees=0.1,  # Minor rotation augmentation
        translate=0.1,  # Light translation
        scale=0.5,  # Scale variation
        shear=0.1,  # Shear augmentation
        perspective=0.0005,  # Perspective distortion
        flipud=0.0,  # No vertical flip
        fliplr=0.5,  # 50% horizontal flip
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,  # Saturation
        hsv_v=0.4,  # Brightness
        mixup=0.0,  # Disable mixup for SEM datasets (usually harms precision)
        dropout=0.1,  # Regularization
        patience=30,  # Early stopping if no improvement
        workers=8,  # CPU threads to load data
        cache=True,  # Cache images for speed (ensure enough RAM)
        amp=True,  # Mixed precision for faster training
        name="yolov12_optimized_v2",
        verbose = True

    )

    # Validate trained model
    metrics = model.val()
    print("Validation Metrics:", metrics)

    # 🔍 Run inference
    results = model("test.jpg")
    result = results[0]
    result.show()                       # Show result in OpenCV window
    result.save(filename="prediction.jpg")

    # Export model
    model.export(format='onnx')

# Required on Windows to avoid multiprocessing crash
if __name__ == '__main__':
    main()
