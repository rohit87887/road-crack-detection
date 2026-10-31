# Adaptive Image Enhancement for Robust Road Surface Crack Detection under Challenging Illumination Conditions

Road cracks are hard to detect automatically when the image has shadows, low light, glare or over-exposure. Most existing methods apply the same fixed enhancement to every image, whatever its lighting.

This project proposes **IAEP — an Illumination-Conditioned Adaptive Enhancement Pipeline**. It measures the quality of each image first and then adapts the enhancement and segmentation to it. The detector is **DualBranchCrackNet**, which fuses a **ResNet-50 CNN** branch with a **Swin-Tiny Transformer** branch.

![Architecture](trp%20picture%20results/architecture%20diagram.jpeg)

## How it works

1. **Image Quality Assessment (IQA) layer**: estimates scalar Quality Indices for illumination, shadow coverage, noise level and local contrast.
2. **Adaptive Enhancement (AE) layer**: enhances the image based on those indices (for example LAB-space CLAHE) before features are extracted.
3. **CNN + Transformer segmentation**: fuses multi-scale features with quality-conditioned gating to segment cracks.
4. **Crack severity analysis**: measures crack length, width, density and damage level.

## Datasets

| Role | Dataset | Samples | Resolution |
|---|---|---|---|
| Training | Asphalt Surface Crack and Uncracked Pavement (Mendeley) | 6,252 (70% of 8,932) | 800×800 |
| Cross-test | CrackForest (CFD) | 118 | 320×480 |
| Cross-test | Asphalt 360×360 | 360 | 360×360 |
| Cross-test | DeepCrack | 300 | 544×384 |

Training used AdamW (lr = 1e-4) with flips, rotations and ±15% brightness/contrast jitter, on Kaggle GPUs.

**Metrics:** Accuracy, Precision, Recall, F1, IoU, SSIM, HD95, illumination-stratified IoU, inference latency and FPS.

## Repository contents

| File | Description |
|---|---|
| [`code/`](code/) | All source code and the trained checkpoint (see below) |
| `Adaptive Image Enhancement ... Conditions.docx` | Research paper |
| `final document.docx` | Final project report |
| `Experimental Setup Results (2).docx` | Experimental setup, datasets and metrics |
| `TARP REVIEW 2 (2).docx` | Review 2 document |
| `lst_ppt.pptx` | Presentation slides |
| `trp picture results/` | Architecture diagrams, confusion matrices, training curves and ROC curves |

### Code layout

```
code/
  BASE_MODEL_CODE/            Baseline models for comparison
    base-model-1-cnn.ipynb
    base-model-2-lenet.ipynb
    base-model-3-alexnet.ipynb
    base-model-4-squeezenet.ipynb
    base-model-5-googlenet.ipynb
  PROPOSED_MODEL_CODE/        Proposed DualBranchCrackNet
    train-proposed.ipynb
    test-proposed.ipynb
    segmentation-test-train_FULL.ipynb
    run_inference.py
    crackdualnet_best.pth     Trained weights
    requirements.txt
```

## Running inference

```bash
cd code/PROPOSED_MODEL_CODE
pip install -r requirements.txt
python run_inference.py                  # interactive: enter or drag-drop an image path
python run_inference.py path/to/image.png
```

PyTorch uses the GPU (CUDA) if there is one, and otherwise runs on the CPU. The results go into `manual test/`:

- `crack_marking_<name>.png`: red crack centerlines on a black background
- `redline_overlay_<name>.png`: the detected cracks overlaid on the original image
- `analysis_grid_<name>.png`: a 6-panel grid showing the input, the enhanced image, the Grad-CAM map, the Otsu mask, the overlay and the verdict (CRACKED / NORMAL)

## Results

| Base vs. proposed model | ROC / AUC |
|---|---|
| ![Final output](trp%20picture%20results/final%20output%20for%20base%20%2C%20proposed%20model.jpg) | ![ROC AUC](trp%20picture%20results/ROC%20AUC%20Curves.jpg) |

| SDNET 2018 | SUT-Crack |
|---|---|
| ![SDNET 2018 confusion matrix](trp%20picture%20results/SNET%202018%20confusion%20matrix.jpg) | ![SUT-Crack confusion matrix](trp%20picture%20results/SUT%20CRACK%20confusion%20matrix.png) |

| Asphalt 360×360 | Asphalt 720×720 |
|---|---|
| ![Asphalt 360 confusion matrix](trp%20picture%20results/asphalt%20360%20x%20360%20confusion%20matrix.jpg) | ![Asphalt 720 confusion matrix](trp%20picture%20results/asphalt%20720%20x%20720%20Confusion%20matrix.jpg) |

More plots are in [`trp picture results/`](trp%20picture%20results/).

## License

This project is released under the [MIT License](LICENSE).
