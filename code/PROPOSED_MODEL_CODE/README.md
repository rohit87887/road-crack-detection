# TARP — Proposed DualBranchCrackNet Inference System
Architecture: **ResNet50 CNN + Swin-Tiny Transformer (Fused)**  
Trained Checkpoint: `crackdualnet_best.pth`

---

## 1. Installation & Setup

Make sure Python 3.8+ (preferably 3.10 to 3.12) is installed.

Open your terminal / command prompt in this folder and install dependencies:

```bash
pip install -r requirements.txt
```

*(Or install manually: `pip install torch torchvision timm opencv-python matplotlib Pillow numpy`)*

> **Note:** PyTorch automatically runs on GPU (CUDA) if available, or falls back seamlessly to CPU.

---

## 2. How to Run

### Mode A: Interactive Mode (Drag & Drop)
```bash
python run_inference.py
```
- The prompt will show: `>> Enter image path : `
- Drag and drop any image into the terminal (or type its path) and press **Enter**.
- Press **Enter** on an empty line to exit.

### Mode B: Direct Image Inference
```bash
python run_inference.py "sample_images/cracked_18_04_01.png"
```
Or for an uncracked road:
```bash
python run_inference.py "sample_images/uncracked_7_03_02.png"
```

---

## 3. Output Files

For every tested image, the outputs are automatically saved into the `manual test/` directory:

1. `crack_marking_<image_name>.png` — Pure black background with thick red crack centerlines.
2. `redline_overlay_<image_name>.png` — Original road surface overlaid with detected red crack lines.
3. `analysis_grid_<image_name>.png` — Complete 6-panel analysis grid:
   - Panel 1: Input Raw Image
   - Panel 2: IEAP Enhanced Frame (LAB CLAHE)
   - Panel 3: Grad-CAM Attention Map
   - Panel 4: Grad-CAM Otsu Binary Mask
   - Panel 5: Road Redline Overlay
   - Panel 6: Crack Marking [CRACKED / NORMAL]
