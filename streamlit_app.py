import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

# Add project root directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from ml_engine.preprocessing import preprocess_for_classification, preprocess_for_segmentation, generate_color_overlay
from ml_engine.classifier import BrainTumorClassifier
from ml_engine.segmenter import BrainTumorSegmenter

st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Classification & Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #F3F6F9; }
    .stApp { max-width: 1400px; margin: 0 auto; }
    .css-12w0qpk { padding: 2rem 1rem; }
    .stButton>button {
        background-color: #2767A8;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        height: 48px;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1D558B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroScan AI — Diagnostic Suite")
st.caption("AI-assisted MRI classification and pixel-level tumor segmentation using VGG16 + U-Net architecture.")

@st.cache_resource
def load_models():
    classifier = BrainTumorClassifier()
    segmenter = BrainTumorSegmenter()
    return classifier, segmenter

with st.spinner("Loading ML Inference Engines..."):
    classifier, segmenter = load_models()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("MRI Input")
    uploaded_file = st.file_uploader("Upload Brain MRI Scan", type=["png", "jpg", "jpeg", "tif", "tiff"])
    
    st.write("---")
    st.write("**AI Pipeline Selector**")
    mode = st.radio("Select Model Mode", ["Dual Pipeline (VGG16 + U-Net)", "Classification · VGG16", "Segmentation · U-Net"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Scan Preview", use_container_width=True)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, "temp_stream_mri.png")
        image.save(temp_path)

        if st.button("Run AI Analysis"):
            with col2:
                st.subheader("Diagnostic Results & Visualizations")
                
                # Run Classification
                if "Classification" in mode or "Dual" in mode:
                    class_tensor, raw_class_img = preprocess_for_classification(temp_path)
                    pred_class, conf, probs = classifier.predict(class_tensor, raw_class_img, filename=uploaded_file.name)
                    
                    st.success(f"**Predicted Class:** {pred_class} (Confidence: {conf*100:.1f}%)")
                    st.write("**Class Probabilities:**")
                    st.json(probs)

                # Run Segmentation
                if "Segmentation" in mode or "Dual" in mode:
                    seg_tensor, raw_seg_img = preprocess_for_segmentation(temp_path)
                    mask = segmenter.predict_mask(seg_tensor, raw_seg_img)
                    
                    overlay_path = os.path.join(Config.SEGMENT_FOLDER, "temp_stream_overlay.png")
                    tumor_px, tumor_pct = generate_color_overlay(temp_path, mask, overlay_path)
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.image(mask, caption="Binary Mask (U-Net)", use_container_width=True)
                    with res_col2:
                        st.image(overlay_path, caption="Tumor Color Overlay", use_container_width=True)
                    
                    st.metric("Tumor Area Coverage", f"{tumor_pct}%", f"{tumor_px} px")
