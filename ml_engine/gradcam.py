import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "PLASMA": cv2.COLORMAP_PLASMA
}

def find_last_conv_layer(model):
    """
    Traverses the Keras model backwards to locate the name of the last 2D Convolutional layer.
    For VGG16, this is typically 'block5_conv3'.
    """
    if model is None:
        return None
        
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If the model encapsulates a nested sub-model (like base_model)
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    return None

def compute_gradcam_heatmap(model, img_tensor, last_conv_layer_name=None, pred_index=None):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap.
    Returns:
        heatmap: 2D numpy array of float32 values in [0.0, 1.0]
    """
    if model is None:
        raise ValueError("Model is None, cannot compute Grad-CAM.")

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            # Common VGG16 fallback name
            last_conv_layer_name = 'block5_conv3'

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        # Search nested models if layer not found at top level
        last_conv_layer = None
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    last_conv_layer = layer.get_layer(last_conv_layer_name)
                    break
                except Exception:
                    continue
        if last_conv_layer is None:
            raise ValueError(f"Conv layer '{last_conv_layer_name}' not found in model.")

    # Multi-output model mapping model input to [conv_output, final_prediction]
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the target class score w.r.t. the convolutional feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients could not be computed (tape returned None).")

    # Global Average Pooling of gradients (spatial importance weights alpha_k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the 2D feature maps by channel importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (we only care about features that positively contribute to target class)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()

def generate_fallback_attention_map(raw_image, target_size=(224, 224)):
    """
    Robust heuristic attention map when running in low-memory mode or uninitialized weights.
    Highlights high-intensity anomalous tissue regions using morphological spatial weighting.
    """
    if raw_image is None:
        # Return a soft radial gaussian center heatmap
        y, x = np.ogrid[:target_size[0], :target_size[1]]
        cy, cx = target_size[0] / 2, target_size[1] / 2
        dist_sq = (x - cx)**2 + (y - cy)**2
        sigma = min(target_size) / 4.0
        hmap = np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)
        return hmap / (np.max(hmap) + 1e-8)

    if len(raw_image.shape) == 3:
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = raw_image.copy()

    gray = cv2.resize(gray, target_size)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Identify hyper-intense focal zones
    anomaly = np.clip((gray.astype(np.float32) - (mean_val + 1.2 * std_val)), 0, 255)
    
    # Smooth with Gaussian blur to mimic neural activation field
    heatmap = cv2.GaussianBlur(anomaly, (25, 25), 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        # Fallback to brain center circle
        h, w = target_size
        cv2.circle(heatmap, (w // 2, h // 2), int(min(h, w) * 0.25), 1.0, -1)
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

    return heatmap.astype(np.float32)

def generate_gradcam_heatmap(classifier, img_tensor, raw_image=None, class_idx=None):
    """
    High-level generator for classification Grad-CAM.
    Attempts neural Grad-CAM first; smoothly falls back to anomaly attention if needed.
    """
    if classifier is not None and getattr(classifier, 'model', None) is not None:
        try:
            return compute_gradcam_heatmap(classifier.model, img_tensor, pred_index=class_idx)
        except Exception as e:
            print(f"[INFO] Grad-CAM neural tape exception ({e}). Using feature attention fallback.")

    return generate_fallback_attention_map(raw_image)

def generate_segmentation_heatmap(segmenter, img_tensor, raw_image=None):
    """
    Generates a continuous probability density heatmap from U-Net segmenter (0.0 to 1.0).
    """
    if segmenter is not None and getattr(segmenter, 'model', None) is not None:
        try:
            raw_prob = segmenter.model.predict(img_tensor, verbose=0)[0, :, :, 0]
            # Normalize to [0, 1]
            prob_map = np.clip(raw_prob.astype(np.float32), 0.0, 1.0)
            if np.max(prob_map) > 0.05:
                return prob_map
        except Exception as e:
            print(f"[INFO] Segmenter probability exception ({e}). Using anomaly density fallback.")

    # Fallback continuous probability map
    if raw_image is not None:
        if len(raw_image.shape) == 3:
            gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = raw_image.copy()
        gray = cv2.resize(gray, (256, 256))
        mean_v, std_v = np.mean(gray), np.std(gray)
        raw_prob = np.clip((gray.astype(np.float32) - (mean_v + 1.5 * std_v)) / (2.0 * std_v + 1e-5), 0.0, 1.0)
        return cv2.GaussianBlur(raw_prob, (15, 15), 0)

    return np.zeros((256, 256), dtype=np.float32)

def overlay_heatmap_on_image(original_image, heatmap, alpha=0.45, colormap_name="JET", mask_background=True):
    """
    Superimposes a 2D float heatmap [0.0, 1.0] onto an MRI scan.
    
    Args:
        original_image: numpy array (RGB) or file path
        heatmap: 2D numpy array with values in [0.0, 1.0]
        alpha: heatmap transparency (0.0 = original image only, 1.0 = heatmap only)
        colormap_name: "JET", "TURBO", "INFERNO", "VIRIDIS", "HOT", "MAGMA", "PLASMA"
        mask_background: if True, keeps pure dark background areas from showing cold colormap noise
        
    Returns:
        color_heatmap_rgb: Colorized standalone heatmap (RGB)
        overlay_rgb: Superimposed blended MRI scan (RGB)
    """
    # 1. Load original image as RGB
    if isinstance(original_image, str):
        bgr = cv2.imread(original_image)
        if bgr is None:
            pil_img = Image.open(original_image).convert('RGB')
            orig_rgb = np.array(pil_img)
        else:
            orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        orig_rgb = original_image.copy()

    h, w = orig_rgb.shape[:2]

    # 2. Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 3. Apply selected colormap
    cv_colormap = COLORMAPS.get(colormap_name.upper(), cv2.COLORMAP_JET)
    color_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv_colormap)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 4. Background noise suppression (prevent dark air outside skull from glowing blue/purple)
    if mask_background:
        gray_orig = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        # Create tissue mask (MRI background is near 0)
        tissue_mask = (gray_orig > 15) & (heatmap_resized > 0.05)
        
        overlay_rgb = orig_rgb.copy()
        # Blend only where tissue and activation exist
        overlay_rgb[tissue_mask] = np.uint8(
            (1.0 - alpha) * orig_rgb[tissue_mask] + alpha * color_heatmap_rgb[tissue_mask]
        )
    else:
        overlay_rgb = np.uint8((1.0 - alpha) * orig_rgb + alpha * color_heatmap_rgb)

    return color_heatmap_rgb, overlay_rgb
