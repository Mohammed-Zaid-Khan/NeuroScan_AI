import cv2
import numpy as np
from PIL import Image

def preprocess_for_classification(image_path, target_size=(224, 224)):
    """
    Preprocesses MRI scan for VGG16 Classification Model.
    - Reads image in RGB format
    - Resizes to target_size (default 224x224)
    - Normalizes pixel values to [0, 1]
    - Returns expanded tensor of shape (1, 224, 224, 3)
    """
    img = cv2.imread(image_path)
    if img is None:
        # Fallback using PIL
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized.astype('float32') / 255.0
    img_tensor = np.expand_dims(img_normalized, axis=0)
    return img_tensor, img_resized

def preprocess_for_segmentation(image_path, target_size=(256, 256)):
    """
    Preprocesses MRI scan for U-Net Segmentation Model.
    - Reads image in Grayscale format
    - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    - Resizes to target_size (default 256x256)
    - Normalizes pixel values to [0, 1]
    - Returns expanded tensor of shape (1, 256, 256, 1)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil_img = Image.open(image_path).convert('L')
        img = np.array(pil_img)

    # Enhance contrast using CLAHE for clearer tissue differentiation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)

    img_resized = cv2.resize(img_enhanced, target_size)
    img_normalized = img_resized.astype('float32') / 255.0
    img_tensor = np.expand_dims(img_normalized, axis=(0, -1))
    return img_tensor, img_resized

def generate_color_overlay(original_image_path, binary_mask, output_path, alpha=0.45):
    """
    Blends the binary segmentation mask over the original MRI scan.
    - Tumor mask colored in vibrant Red/Yellow
    - Saves composite overlay image to output_path
    - Returns tumor area (pixels) and tumor percentage relative to brain area
    """
    # Load original image in RGB
    orig_bgr = cv2.imread(original_image_path)
    if orig_bgr is None:
        pil_img = Image.open(original_image_path).convert('RGB')
        orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = orig_bgr.shape[:2]
    mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Color mask: Vibrant Red (BGR: 0, 0, 255)
    color_mask = np.zeros_like(orig_bgr)
    color_mask[mask_resized > 0] = [0, 40, 255] # Red/Orange

    # Apply boundary outline contour
    contours, _ = cv2.findContours((mask_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = orig_bgr.copy()
    # Blend color inside tumor mask
    mask_indices = mask_resized > 0
    overlay[mask_indices] = cv2.addWeighted(orig_bgr[mask_indices], 1 - alpha, color_mask[mask_indices], alpha, 0)
    
    # Draw contour line in neon cyan/yellow for clinical clarity
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    cv2.imwrite(output_path, overlay)

    # Calculate metrics relative to total slice area
    tumor_pixels = int(np.sum(mask_resized > 0))
    total_pixels = h * w
    
    # Calculate percentage relative to total slice resolution
    raw_pct = (tumor_pixels / max(1, total_pixels)) * 100.0
    tumor_percentage = round(min(100.0, raw_pct), 2)
    return tumor_pixels, tumor_percentage
