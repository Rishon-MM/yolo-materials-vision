# 🔬 YOLOv12 SEM Particle Defect Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-black.svg)](https://github.com/ultralytics/ultralytics)

This repository contains a specialized YOLO object detection pipeline optimized for **Scanning Electron Microscope (SEM) imagery**. It is designed to analyze powder quality (e.g., for additive manufacturing, metallurgy, or battery materials) by detecting and classifying individual particles based on their morphology and surface defects.

### 🎯 Classification Categories
The model is trained to detect three specific classes of particles:
- 🟢 **Good**: Well-formed, intact, and highly spherical particles.
- 🟡 **Irregular**: Misshapen, agglomerated, or satellite-bearing particles.
- 🔴 **Porous**: Particles exhibiting surface cracks, pores, or severe structural defects.

---

## ✨ Features

- **Domain-Specific Hyperparameters**: The training pipeline (`Training.py`) is heavily tuned for SEM images. For example, `mixup` augmentation is explicitly disabled as it is known to degrade precision on dense, grayscale microstructural datasets.
- **High-Resolution Processing**: Configured to process `640x640` images to capture fine morphological details like micro-pores.
- **GPU Acceleration Checks**: Includes a dedicated script (`GPU TEST.py`) to verify CUDA compatibility and Tensor Core matrix multiplication before starting heavy training.
- **Automated Validation & Inference**: Evaluates the model immediately after training and runs an inference test to produce visual bounding-box results.
- **ONNX Export**: Instantly exports the best PyTorch model (`.pt`) to ONNX (`.onnx`) format for deployment in industrial inspection pipelines (C++, C#, etc.).

---

## 🛠️ Project Structure

```bash
📦 Project Root
 ┣ 📜 Training.py       # Main script to train, validate, test, and export the model
 ┣ 📜 GPU TEST.py       # Script to verify CUDA availability and performance
 ┣ 📜 data.yaml         # Dataset configuration file mapping the 3 particle classes
 ┣ 📜 requirements.txt  # Python dependencies for the project
 ┣ 📜 LICENSE           # Open-source MIT License
 ┗ 📜 .gitignore        # Ignores heavy weights, datasets, and cache files
```

*(Note: Datasets and model weights are ignored in version control to keep the repository lightweight.)*

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Ensure you install the correct [PyTorch version with CUDA support](https://pytorch.org/get-started/locally/) if you plan to train on a GPU.*

---

## 🚀 Usage

### 1. Verify GPU Setup
Before running the main training loop, verify your PyTorch CUDA setup to ensure your GPU is fully utilized:
```bash
python "GPU TEST.py"
```

### 2. Prepare Your Dataset
1. Place your dataset in the standard YOLO structure (`train/`, `valid/`, `test/`).
2. Ensure your `data.yaml` defines `nc: 3` and includes your dataset paths.

### 3. Start Training
The `Training.py` script will automatically load the dataset, run an optimized training loop for up to 150 epochs (with early stopping at 30 epochs), validate the results, and run an inference test on `test.jpg`.

Run the training script:
```bash
python Training.py
```

### 4. Outputs
- **Trained Weights:** The best model will be saved inside the `runs/` directory (e.g., `runs/detect/yolov12_optimized_v2/weights/best.pt`).
- **Inference Results:** A sample prediction containing the detected "Good", "Irregular", and "Porous" particles will be saved as `prediction.jpg`.
- **Exported Model:** The model is exported to ONNX format automatically for integration into downstream analysis tools.

---

## 🧠 Training Insights for SEM Data

The `Training.py` script contains carefully selected hyperparameter choices for microstructural analysis:
- **Augmentation**: Employs moderate rotation (`degrees=0.1`), translation (`translate=0.1`), and scaling (`scale=0.5`).
- **No Mixup**: `mixup=0.0` is explicitly set. Blending dense SEM images creates ghosting artifacts that confuse the model when trying to detect hard boundaries or tiny surface pores.
- **Learning Rate**: `lr0=0.002` & `lrf=0.01` provides a slightly higher initial learning rate with a low final learning rate for excellent fine-tuning on subtle texture differences.

---

## 📄 License
This project is open-sourced under the [MIT License](LICENSE).
