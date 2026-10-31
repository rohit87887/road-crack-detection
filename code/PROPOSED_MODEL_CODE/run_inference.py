#!/usr/bin/env python3
# ==============================================================================
# TARP COURSE PROJECT — PROPOSED MODEL INFERENCE SYSTEM
# Architecture: DualBranchCrackNet (ResNet50 CNN + Swin-Tiny Transformer)
# Trained Weights: crackdualnet_best.pth
# Author: 23MIA1126
#
# USAGE OPTIONS:
#   1. Direct specific image:
#      python3 run_inference.py "path/to/image.png"
#
#   2. Interactive mode (prompts for image path):
#      python3 run_inference.py
# ==============================================================================

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Global Configuration ──────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE      = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Locate project root directory and candidate paths for trained checkpoint
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "manual test")
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

POSSIBLE_CKPTS = [
    os.path.join(SCRIPT_DIR, "crackdualnet_best.pth"),
    os.path.join(SCRIPT_DIR, "training", "crackdualnet_best.pth"),
    os.path.join(SCRIPT_DIR, "..", "training", "crackdualnet_best.pth"),
    os.path.join(SCRIPT_DIR, "..", "crackdualnet_best.pth"),
    os.path.join(SCRIPT_DIR, "CODE", "PROPOSED_MODEL_CODE", "crackdualnet_best.pth"),
]

CLASS_LABEL = {
    0: "CRACKED ROAD SURFACE",
    1: "UNCRACKED / NORMAL ROAD"
}


# ==============================================================================
# SECTION 1 — PROPOSED MODEL ARCHITECTURE (DualBranchCrackNet)
# ==============================================================================
class DualBranchCrackNet(nn.Module):
    """
    Proposed Dual-Branch Network:
      - CNN Branch   : ResNet50 (stem + layer1 + layer2 + layer3 -> 1024-ch)
      - Swin Branch  : Swin-Tiny Transformer (stage-2 feature map -> 384-ch)
      - Projections  : 1x1 convs mapping both to 512 channels at 16x16 grid
      - Fusion       : Element-wise addition (fused = f_cnn + f_swin)
      - Head 1       : Classifier (GAP -> Linear(512,128) -> ReLU -> Linear(128,2))
      - Head 2       : Heatmap Conv (1x1 conv -> Sigmoid -> Upsample to 256x256)
    """
    def __init__(self, num_classes=2, fused_channels=512, fused_size=16, img_size=256):
        super().__init__()
        self.fused_size = fused_size

        # CNN Branch (ResNet50)
        resnet = models.resnet50(weights=None)
        self.cnn_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        cnn_out_channels = 1024

        # Swin Transformer Branch
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(2,),
            img_size=img_size,
        )
        swin_out_channels = self.swin.feature_info.channels()[0]

        # Shared dimension projections
        self.cnn_proj  = nn.Conv2d(cnn_out_channels,  fused_channels, kernel_size=1)
        self.swin_proj = nn.Conv2d(swin_out_channels, fused_channels, kernel_size=1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fused_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Heatmap Head
        self.heatmap_conv = nn.Conv2d(fused_channels, 1, kernel_size=1)

    def forward(self, x):
        # CNN representation
        f_cnn = self.cnn_stem(x)
        f_cnn = self.cnn_proj(f_cnn)
        f_cnn = F.interpolate(f_cnn, (self.fused_size, self.fused_size),
                              mode="bilinear", align_corners=False)

        # Swin representation
        f_swin = self.swin(x)[0]
        if f_swin.shape[1] != self.swin_proj.in_channels:
            f_swin = f_swin.permute(0, 3, 1, 2).contiguous()
        f_swin = self.swin_proj(f_swin)
        f_swin = F.interpolate(f_swin, (self.fused_size, self.fused_size),
                               mode="bilinear", align_corners=False)

        # Dual-branch fusion
        fused = f_cnn + f_swin

        logits  = self.classifier(fused)
        heatmap = torch.sigmoid(self.heatmap_conv(fused))
        heatmap = F.interpolate(heatmap, (IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False)

        return logits, heatmap, fused


# ==============================================================================
# SECTION 2 — IEAP ENHANCEMENT (Adaptive CLAHE in LAB space)
# ==============================================================================
def ieap_enhance(bgr_image):
    """Applies CLAHE on the luminance channel (L) of LAB color space."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)


# ==============================================================================
# SECTION 3 — MORPHOLOGICAL SKELETONIZATION & MEASUREMENTS
# ==============================================================================
def morphological_skeleton(binary_mask_u8):
    """Extracts thin morphological centerlines via iterative erosion-opening."""
    skel   = np.zeros_like(binary_mask_u8)
    temp   = binary_mask_u8.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded  = cv2.erode(temp, kernel)
        opened  = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        contrib = cv2.subtract(eroded, opened)
        skel    = cv2.bitwise_or(skel, contrib)
        temp    = eroded
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def postprocess_model_heatmap(cam_uint8, is_cracked):
    """
    Computes Otsu threshold, morphological open/close cleanup,
    centerline skeleton, thick redline dilation, and EDT measurements.
    Faithfully mirrors experiment_pseudo_segmentation.py.
    """
    if is_cracked and np.max(cam_uint8) > 0:
        otsu_val, binary = cv2.threshold(
            cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_m)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_m)

        bool_mask = binary > 0
        skel = morphological_skeleton(binary)

        # Thicken lines using 5x5 elliptical structuring element (~3-4px line width)
        thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skel_thick = cv2.dilate(skel, thick_kernel)

        edt_map     = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        density_pct = float(100.0 * bool_mask.sum() / bool_mask.size)
        width_mm    = float(edt_map.max()) * 2.0 * 3.0  # approximate physical scale
    else:
        otsu_val    = 0
        binary      = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        skel        = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        skel_thick  = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        edt_map     = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        density_pct = 0.0
        width_mm    = 0.0

    return {
        "otsu_thresh":       float(otsu_val) / 255.0,
        "binary_mask":       binary,
        "skeleton":          skel,
        "skeleton_thick":    skel_thick,
        "edt_map":           edt_map,
        "crack_density_pct": density_pct,
        "crack_width_mm":    width_mm,
    }


def compute_image_quality(rgb_image):
    """Computes luminance, RMS contrast, noise sigma, and sharpness."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "mean_luminance": float(np.mean(gray)),
        "rms_contrast":   float(np.std(gray)),
        "noise_sigma":    float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(lap)) / 6.0),
        "sharpness":      float(np.var(lap)),
    }


def grade_severity(width_mm, is_cracked):
    if not is_cracked:   return "GRADE 0: NORMAL"
    if width_mm >= 5.0:  return "GRADE 3: SEVERE"
    if width_mm >= 2.0:  return "GRADE 2: MODERATE"
    return "GRADE 1: MINOR"


def compute_shannon_entropy(prob_f32):
    p = np.clip(prob_f32, 1e-6, 1.0 - 1e-6)
    ent = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.mean(ent))


# ==============================================================================
# SECTION 4 — VISUAL OUTPUT GENERATION
# ==============================================================================
def generate_visual_outputs(img_rgb, enh_rgb, hmap_u8, pp, pred_idx,
                            probs, iq, out_dir, base_name):
    """
    Generates:
      1. Complete 6-Panel Analysis Figure (.png)
      2. Standalone Pure Black Background with Thick Red Crack Lines (.png)
    """
    is_cracked   = (pred_idx == 0)
    conf_pct     = float(probs[pred_idx]) * 100.0
    severity     = grade_severity(pp["crack_width_mm"], is_cracked)
    shannon      = compute_shannon_entropy(hmap_u8.astype(np.float32) / 255.0)
    hm_f         = hmap_u8.astype(np.float32) / 255.0

    # ──────────────────────────────────────────────────────────────────────────
    # File 1: Standalone Black Background with Thick Red Crack Lines
    # ──────────────────────────────────────────────────────────────────────────
    marking_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if is_cracked and np.count_nonzero(pp["skeleton_thick"]) > 0:
        marking_img[pp["skeleton_thick"] > 0] = [255, 0, 0]  # Red in RGB
    marking_path = os.path.join(out_dir, f"crack_marking_{base_name}.png")
    Image.fromarray(marking_img).save(marking_path)

    # ──────────────────────────────────────────────────────────────────────────
    # File 2: Redline Overlay on Original Road Image
    # ──────────────────────────────────────────────────────────────────────────
    overlay_img = img_rgb.copy()
    if is_cracked and np.count_nonzero(pp["skeleton_thick"]) > 0:
        overlay_img[pp["skeleton_thick"] > 0] = [255, 0, 0]
    overlay_path = os.path.join(out_dir, f"redline_overlay_{base_name}.png")
    Image.fromarray(overlay_img).save(overlay_path)

    # ──────────────────────────────────────────────────────────────────────────
    # File 3: Comprehensive 6-Panel Scientific Analysis Grid
    # ──────────────────────────────────────────────────────────────────────────
    BG_COLOR   = "#0d1117"
    TEXT_COLOR = "#94a3b8"

    fig = plt.figure(figsize=(26, 4.6), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.05)
    axs = [fig.add_subplot(gs[0, i]) for i in range(6)]

    for ax in axs:
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")

    # Panel 1: Raw input image
    axs[0].imshow(img_rgb)
    axs[0].set_title("1. Input Raw Image", fontsize=10, color=TEXT_COLOR, pad=6, fontweight="bold")

    # Panel 2: IEAP Enhanced
    axs[1].imshow(enh_rgb)
    axs[1].set_title("2. IEAP Enhanced Frame", fontsize=10, color=TEXT_COLOR, pad=6)

    # Panel 3: Grad-CAM Probability Map
    axs[2].imshow(hm_f, cmap="jet", vmin=0, vmax=1)
    axs[2].set_title(f"3. Grad-CAM Map (max={hm_f.max():.3f})",
                     fontsize=10, color=TEXT_COLOR, pad=6, fontweight="bold")

    # Panel 4: Grad-CAM Otsu Mask
    mask_disp = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    mask_disp[:, :, 0] = pp["binary_mask"]
    axs[3].imshow(mask_disp)
    axs[3].set_title("4. Grad-CAM Otsu Mask", fontsize=10, color=TEXT_COLOR, pad=6, fontweight="bold")

    # Panel 5: Road Redline Overlay
    axs[4].imshow(overlay_img)
    axs[4].set_title("5. Road Redline Overlay", fontsize=10, color=TEXT_COLOR, pad=6, fontweight="bold")

    # Panel 6: Crack Marking on Black Background
    axs[5].imshow(marking_img)
    lbl_color = "#f87171" if is_cracked else "#4ade80"
    lbl_status = "CRACKED" if is_cracked else "NORMAL"
    axs[5].set_title(f"6. Crack Marking [{lbl_status}]", fontsize=10, color=lbl_color, pad=6, fontweight="bold")

    grid_path = os.path.join(out_dir, f"analysis_grid_{base_name}.png")
    fig.savefig(grid_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    return grid_path, marking_path, overlay_path


# ==============================================================================
# SECTION 5 — INFERENCE PIPELINE
# ==============================================================================
def find_checkpoint():
    for p in POSSIBLE_CKPTS:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def run_pipeline_on_image(image_path, model, out_dir=None):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Read and Resize
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. IEAP Enhancement
    enh_bgr = ieap_enhance(bgr)
    enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)

    # 3. Model Preprocessing & Grad-CAM Forward/Backward Pass (experiment_pseudo_segmentation.py)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = tfm(Image.fromarray(enh_rgb)).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    model.zero_grad()

    fused_grads = None
    def backward_hook(grad):
        nonlocal fused_grads
        fused_grads = grad

    logits, _, fused = model(tensor)
    h_hook = fused.register_hook(backward_hook)

    # Backpropagate targeting Class 0 (Cracked)
    score = logits[:, 0]
    score.backward()
    h_hook.remove()

    probs = F.softmax(logits.detach(), dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    is_cracked = (pred_idx == 0)

    # Extract Grad-CAM on fused feature tensor
    grads = fused_grads.detach().cpu().numpy()[0]
    acts  = fused.detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam)

    cam_resized = cv2.resize(cam_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    cam_u8 = np.uint8(255 * cam_resized)

    # 4. Classical Postprocessing (Otsu threshold, Morphological cleanup, Skeleton, EDT)
    pp = postprocess_model_heatmap(cam_u8, is_cracked)
    iq = compute_image_quality(img_rgb)
    sev = grade_severity(pp["crack_width_mm"], is_cracked)
    shannon = compute_shannon_entropy(cam_resized)

    # 5. Visual Output Files
    grid_path, marking_path, overlay_path = generate_visual_outputs(
        img_rgb=img_rgb, enh_rgb=enh_rgb, hmap_u8=cam_u8,
        pp=pp, pred_idx=pred_idx, probs=probs, iq=iq,
        out_dir=out_dir, base_name=base_name
    )

    # 6. Formatted Terminal Report
    bar = "─" * 60
    print()
    print(f"  ┌{bar}┐")
    print(f"  │  PROPOSED MODEL INFERENCE RESULT: {os.path.basename(image_path):<25}│")
    print(f"  ├{bar}┤")
    print(f"  │  Classification Status : {CLASS_LABEL[pred_idx]:<34}│")
    print(f"  │  Confidence Score      : {probs[pred_idx]*100:>7.2f}%{' '*27}│")
    print(f"  │  P(Cracked / Class 0)  : {probs[0]:>8.4f}{' '*27}│")
    print(f"  │  P(Uncrack / Class 1)  : {probs[1]:>8.4f}{' '*27}│")
    print(f"  │  Otsu Threshold (T)    : {pp['otsu_thresh']:>8.4f}{' '*27}│")
    print(f"  │  Crack Surface Density : {pp['crack_density_pct']:>7.4f}%{' '*27}│")
    print(f"  │  Est. Crack Width      : {pp['crack_width_mm']:>7.2f} mm{' '*26}│")
    print(f"  │  Pixel Shannon Entropy : {shannon:>7.4f} bits{' '*23}│")
    print(f"  │  Mean Luminance (IQ)   : {iq['mean_luminance']:>8.2f}{' '*27}│")
    print(f"  │  RMS Contrast (IQ)     : {iq['rms_contrast']:>8.2f}{' '*27}│")
    print(f"  │  Severity Assessment   : {sev:<34}│")
    print(f"  ├{bar}┤")
    print(f"  │  [OUTPUT FILES SAVED IN 'manual test/']:{' '*19}│")
    print(f"  │  1. Black BG Redline   : {os.path.basename(marking_path):<34}│")
    print(f"  │  2. Road Overlay       : {os.path.basename(overlay_path):<34}│")
    print(f"  │  3. 6-Panel Analysis   : {os.path.basename(grid_path):<34}│")
    print(f"  └{bar}┘")
    print()
    print(f"  ✓ Redline Marking (Black BG) : {marking_path}")
    print(f"  ✓ Road Redline Overlay       : {overlay_path}")
    print(f"  ✓ 6-Panel Analysis Grid      : {grid_path}")
    print()

    return {
        "pred_idx": pred_idx,
        "probs": probs,
        "metrics": pp,
        "marking_path": marking_path,
        "overlay_path": overlay_path,
        "grid_path": grid_path
    }


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run TARP DualBranchCrackNet inference on any road pavement image."
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to an input road image (optional; prompts interactively if omitted)."
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to crackdualnet_best.pth (auto-detected if omitted)."
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUT_DIR,
        help="Directory to save output PNG files (defaults to 'manual test' folder)."
    )
    args = parser.parse_args()

    # Locate checkpoint
    ckpt_path = args.checkpoint or find_checkpoint()
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"\n[ERROR] Model checkpoint not found! Checked locations:")
        for p in POSSIBLE_CKPTS:
            print(f"  - {p}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  TARP — ROAD CRACK DETECTION: PROPOSED DUAL-BRANCH MODEL")
    print("  Architecture : ResNet50 CNN + Swin-Tiny Transformer (Fused)")
    print(f"  Device       : {DEVICE}")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Output Dir   : {args.outdir}")
    print("=" * 70)

    # Load Model
    model = DualBranchCrackNet(num_classes=2, img_size=IMG_SIZE).to(DEVICE)
    checkpoint_data = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint_data, strict=True)
    model.eval()
    print("  ✓ Model loaded successfully with trained weights.\n")

    # Mode 1: Command-line direct image passed
    if args.image:
        img_path = args.image.strip('"').strip("'")
        try:
            run_pipeline_on_image(img_path, model, out_dir=args.outdir)
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)
        return

    # Mode 2: Interactive loop
    print("  [INTERACTIVE MODE]")
    print("  Type or drag-and-drop the path to an image file and press Enter.")
    print("  Press Enter on an empty line to exit.\n")

    while True:
        try:
            user_input = input("  >> Enter image path : ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if not user_input:
            print("  Exiting.")
            break

        if not os.path.isfile(user_input):
            print(f"  [ERROR] File does not exist: {user_input}\n")
            continue

        try:
            run_pipeline_on_image(user_input, model, out_dir=args.outdir)
        except Exception as e:
            print(f"  [ERROR processing image]: {e}\n")


if __name__ == "__main__":
    main()
