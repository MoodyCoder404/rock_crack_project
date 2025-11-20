import math
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# --- NVIDIA SegFormer (generic segmentation) ---
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
device = torch.device("cpu")

# Try to load SegFormer; if it fails (e.g., no net), we still proceed with CV pipeline
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    segformer = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).to(device)
    segformer.eval()
    torch.set_grad_enabled(False)
    _SEGFORMER_OK = True
except Exception as e:
    print("SegFormer warning:", str(e))
    processor, segformer = None, None
    _SEGFORMER_OK = False


def _clahe_bilateral(gray: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral filter to normalize contrast and preserve edges on rock surfaces."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.bilateralFilter(clahe, d=7, sigmaColor=50, sigmaSpace=50)


def _auto_canny(gray: np.ndarray, sigma: float = 0.33):
    """Auto canny thresholds based on image median."""
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    lower = max(10, lower)   # floor to avoid too-soft thresholds
    upper = max(lower + 30, upper)
    return cv2.Canny(gray, lower, upper)


def _structural_edges(rgb: np.ndarray) -> np.ndarray:
    """Return binary edges (uint8 0/255) highlighting structural cracks."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = _clahe_bilateral(gray)

    edges = _auto_canny(gray, sigma=0.33)
    # Close small gaps and remove tiny specks
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    # Slight dilate to stabilize coverage
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _keep_structural(edges: np.ndarray, minor_len=5, major_len=45) -> np.ndarray:
    """
    Keep only connected components with meaningful span (structural cracks).
    Returns binary mask (uint8 0/255).
    """
    mask = np.zeros_like(edges, dtype=np.uint8)
    # This is the line we fixed before, make sure it has no extra spaces
    num, labels = cv2.connectedComponents((edges > 0).astype(np.uint8))
    
    # This is the line from the error, make sure it's indented just like this
    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        if ys.size < 3:
            continue
        span = math.hypot(xs.max() - xs.min(), ys.max() - ys.min())
        if span >= minor_len:  # keep both minor and major; span threshold filters texture noise
            mask[labels == lab] = 255
    return mask

def analyze_image(image_path: str):
    """
    Returns:
        crack_percent (float, 0–100),
        condition (str: Good / Moderate / Poor)
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # (Optional) run SegFormer to keep model in pipeline (not used in scoring here).
    if _SEGFORMER_OK:
        try:
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
            _ = segformer(**inputs)
        except Exception as e:
            print("SegFormer forward warn:", str(e))

    # Improved crack proxy tuned for rocks
    edges = _structural_edges(rgb)
    structural = _keep_structural(edges, minor_len=12, major_len=45)

    crack_percent = round((np.sum(structural > 0) / (H * W)) * 100.0, 2)

    # Standard bands: <10 Good, 10–30 Moderate, >30 Poor
    if crack_percent < 10:
        condition = "Good"
    elif crack_percent < 30:
        condition = "Moderate"
    else:
        condition = "Poor"

    return crack_percent, condition
