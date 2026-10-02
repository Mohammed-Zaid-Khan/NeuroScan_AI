import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
import cv2

# Add project root directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from ml_engine.preprocessing import (
    preprocess_for_classification,
    preprocess_for_segmentation,
    generate_color_overlay
)
from ml_engine.classifier import BrainTumorClassifier
from ml_engine.segmenter import BrainTumorSegmenter
from ml_engine.gradcam import (
    generate_gradcam_heatmap,
    generate_segmentation_heatmap,
    overlay_heatmap_on_image,
    COLORMAPS
)

st.set_page_config(
    page_title="NeuroScan AI | Diagnostic Suite & Heatmap XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stApp { max-width: 1440px; margin: 0 auto; }
    .diagnosis-card {
        padding: 1.25rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .xai-info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        height: 48px;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🧠 NeuroScan AI — Diagnostic Suite & Explainable XAI")
st.caption("Clinical-grade MRI classification (VGG16), U-Net segmentation, and visual Explainable AI (Grad-CAM & Probability Density Heatmaps).")

@st.cache_resource
def load_models():
    classifier = BrainTumorClassifier()
    segmenter = BrainTumorSegmenter()
    return classifier, segmenter

with st.spinner("Initializing Deep Learning Engines..."):
    classifier, segmenter = load_models()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Diagnostics & XAI Controls")
    
    st.subheader("Visual Heatmap Settings")
    selected_colormap = st.selectbox(
        "Heatmap Palette",
        options=["JET", "TURBO", "INFERNO", "VIRIDIS", "HOT", "MAGMA"],
        index=0,
        help="JET and TURBO offer high contrast; INFERNO and HOT highlight peak thermal intensities."
    )
    
    heatmap_alpha = st.slider(
        "Heatmap Blend Opacity (α)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Adjust the transparency of the heatmap overlaid on the original MRI."
    )
    
    mask_bg = st.checkbox(
        "Suppress Background Noise",
        value=True,
        help="Suppresses colormap noise on black background regions outside the skull."
    )
    
    st.markdown("---")
    st.markdown("### 🔬 About Model Interpretability")
    st.markdown("""
    - **Grad-CAM (Classification):** Highlights convolutional feature activations determining tumor type.
    - **Probability Density (Segmentation):** Depicts continuous tumor tissue likelihood from 0% to 100%.
    """)

# Main Layout
col_input, col_results = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("1. MRI Scan Input")
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload axial, coronal, or sagittal T1/T2/FLAIR MRI slice."
    )
    
    st.markdown("---")
    st.subheader("2. AI Analysis Pipeline")
    mode = st.radio(
        "Select Pipeline Mode",
        ["Dual Pipeline (VGG16 + U-Net + XAI)", "Classification · VGG16 + Grad-CAM", "Segmentation · U-Net + Density Heatmap"],
        index=0
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        # Save temporary file for inference
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.SEGMENT_FOLDER, exist_ok=True)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, "temp_stream_mri.png")
        image.save(temp_path)

        # Action Button
        btn_label = "🚀 Run Complete AI & Heatmap Analysis"
        if "Classification" in mode:
            btn_label = "🔬 Run VGG16 Classification & Grad-CAM"
        elif "Segmentation" in mode:
            btn_label = "🧩 Run U-Net Segmentation & Probability Heatmap"

        run_analysis = st.button(btn_label)
    else:
        run_analysis = False
        st.info("👆 Please upload a brain MRI scan to begin diagnostic evaluation.")

with col_results:
    if uploaded_file is not None and run_analysis:
        st.subheader("Diagnostic Evaluation & Explainability Analysis")
        
        with st.spinner("Analyzing MRI slice and computing neural activation heatmaps..."):
            pred_class, conf, probs = None, None, None
            gradcam_pure, gradcam_overlay = None, None
            mask, overlay_path, tumor_px, tumor_pct = None, None, None, None
            seg_pure, seg_overlay = None, None

            # ----------------------------------------------------
            # 1. Classification & Grad-CAM
            # ----------------------------------------------------
            if "Classification" in mode or "Dual" in mode:
                class_tensor, raw_class_img = preprocess_for_classification(temp_path)
                pred_class, conf, probs = classifier.predict(
                    class_tensor,
                    raw_class_img,
                    filename=uploaded_file.name
                )
                
                # Compute Grad-CAM
                gradcam_map = generate_gradcam_heatmap(classifier, class_tensor, raw_class_img)
                gradcam_pure, gradcam_overlay = overlay_heatmap_on_image(
                    temp_path,
                    gradcam_map,
                    alpha=heatmap_alpha,
                    colormap_name=selected_colormap,
                    mask_background=mask_bg
                )

            # ----------------------------------------------------
            # 2. Segmentation & Probability Density Heatmap
            # ----------------------------------------------------
            if "Segmentation" in mode or "Dual" in mode:
                seg_tensor, raw_seg_img = preprocess_for_segmentation(temp_path)
                mask = segmenter.predict_mask(seg_tensor, raw_seg_img)
                
                overlay_path = os.path.join(Config.SEGMENT_FOLDER, "temp_stream_overlay.png")
                tumor_px, tumor_pct = generate_color_overlay(temp_path, mask, overlay_path)
                
                # Compute continuous probability density heatmap
                prob_map = segmenter.predict_probability_map(seg_tensor, raw_seg_img)
                seg_pure, seg_overlay = overlay_heatmap_on_image(
                    temp_path,
                    prob_map,
                    alpha=heatmap_alpha,
                    colormap_name=selected_colormap,
                    mask_background=mask_bg
                )

        # Tabbed Results Display
        tabs = []
        if "Dual" in mode:
            tab_overview, tab_gradcam, tab_density, tab_compare = st.tabs([
                "📋 Executive Overview",
                "🔥 Grad-CAM Heatmap (VGG16)",
                "🌊 Probability Density (U-Net)",
                "🔍 Multi-Panel Comparison"
            ])
        elif "Classification" in mode:
            tab_overview, tab_gradcam = st.tabs([
                "📋 Classification Summary",
                "🔥 Grad-CAM Attention Heatmap"
            ])
            tab_density, tab_compare = None, None
        else:
            tab_overview, tab_density = st.tabs([
                "📋 Segmentation Summary",
                "🌊 Probability Density Heatmap"
            ])
            tab_gradcam, tab_compare = None, None

        # ----------------------------------------------------
        # Tab 1: Executive Overview
        # ----------------------------------------------------
        with tab_overview:
            if pred_class is not None:
                color_badge = "🟢" if pred_class == "No Tumor" else "🔴"
                st.markdown(f"### {color_badge} Diagnosis: **{pred_class}**")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Condition", pred_class)
                m2.metric("Diagnostic Confidence", f"{conf*100:.1f}%")
                if tumor_pct is not None:
                    m3.metric("Tumor Slice Coverage", f"{tumor_pct}%", f"{tumor_px} px")
                
                st.markdown("#### Probability Distribution")
                for c_name, c_prob in probs.items():
                    col_bar, col_val = st.columns([4, 1])
                    with col_bar:
                        st.progress(float(c_prob) / 100.0, text=c_name)
                    with col_val:
                        st.write(f"**{c_prob}%**")

            if mask is not None:
                st.markdown("---")
                st.markdown("#### Tumor Lesion Localization (U-Net)")
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    st.image(mask, caption="Binary Segmentation Mask (Tumor in White)", use_container_width=True)
                with s_c2:
                    st.image(overlay_path, caption="Clinical Delineation Overlay (Red = Tumor)", use_container_width=True)

        # ----------------------------------------------------
        # Tab 2: Grad-CAM Attention
        # ----------------------------------------------------
        if tab_gradcam is not None and gradcam_overlay is not None:
            with tab_gradcam:
                st.markdown("### 🔥 Grad-CAM Class Activation Mapping (Explainable AI)")
                st.markdown("""
                <div class="xai-info-box">
                    <strong>How to interpret this heatmap:</strong> Red and warm yellow regions represent high convolutional gradient attention where the VGG16 network focused to categorize this scan. Cool blue regions had negligible impact on the prediction.
                </div>
                """, unsafe_allow_html=True)
                
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    st.image(temp_path, caption="Original Input MRI", use_container_width=True)
                with g_col2:
                    st.image(gradcam_pure, caption=f"Grad-CAM Heatmap ({selected_colormap})", use_container_width=True)
                with g_col3:
                    st.image(gradcam_overlay, caption=f"Superimposed Attention (α={heatmap_alpha})", use_container_width=True)

        # ----------------------------------------------------
        # Tab 3: Segmentation Probability Density
        # ----------------------------------------------------
        if tab_density is not None and seg_overlay is not None:
            with tab_density:
                st.markdown("### 🌊 Continuous Tumor Probability Density (U-Net)")
                st.markdown("""
                <div class="xai-info-box">
                    <strong>Continuous vs Binary Segmentation:</strong> Unlike a binary 0/1 mask, this heatmap plots the continuous sigmoid output distribution [0.0, 1.0]. Peak red regions identify the solid tumor core, while surrounding green/yellow bands visualize infiltrative margins and boundary uncertainty.
                </div>
                """, unsafe_allow_html=True)

                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    st.image(temp_path, caption="Original Input MRI", use_container_width=True)
                with p_col2:
                    st.image(seg_pure, caption=f"Probability Density Map ({selected_colormap})", use_container_width=True)
                with p_col3:
                    st.image(seg_overlay, caption=f"Confidence Overlay (α={heatmap_alpha})", use_container_width=True)

        # ----------------------------------------------------
        # Tab 4: Multi-Panel Comparison
        # ----------------------------------------------------
        if tab_compare is not None and gradcam_overlay is not None and seg_overlay is not None:
            with tab_compare:
                st.markdown("### 🔍 Complete Multi-Modal Diagnostic Synthesis")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.image(temp_path, caption="1. Anatomical MRI", use_container_width=True)
                with c2:
                    st.image(gradcam_overlay, caption="2. VGG16 Grad-CAM", use_container_width=True)
                with c3:
                    st.image(seg_overlay, caption="3. U-Net Probability Heatmap", use_container_width=True)
                with c4:
                    st.image(overlay_path, caption="4. Delineated Contour", use_container_width=True)
    elif uploaded_file is None:
        st.subheader("Diagnostic Evaluation")
        st.info("Awaiting scan upload. Select an MRI file on the left panel to launch the diagnostic pipeline.")
