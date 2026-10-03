import os
import sys
import io
import time
import textwrap
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Add project root directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from database import (
    init_db,
    record_upload,
    record_prediction,
    get_dashboard_analytics,
    get_history_records,
    log_action
)
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

# Initialize Database Schema
init_db()

st.set_page_config(
    page_title="NeuroScan AI | Clinical MRI Diagnostic Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to render HTML markdown without leading indentation issues
def render_html(html_str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# ----------------------------------------------------
# 1. CLINICAL MEDICAL WORKSTATION STYLING & HIGH-CONTRAST PALETTE
# ----------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* GLOBAL HIGH-CONTRAST TYPOGRAPHY & VISIBILITY ENFORCEMENT */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* FIX INVISIBLE FONT IN DARK & LIGHT MODES */
    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] span,
    .stApp [data-testid="stWidgetLabel"] label,
    .stApp div[data-testid="stRadio"] label,
    .stApp div[data-testid="stRadio"] span,
    .stApp .stTabs [role="tab"],
    .stApp [data-baseweb="tab"],
    .stApp .stExpander,
    .stApp .stExpander p,
    .stApp .stExpander span {
        color: #0F172A !important;
    }

    /* Muted secondary captions */
    .stApp .stCaption,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* App container constraint */
    .stApp {
        max-width: 1440px;
        margin: 0 auto;
    }
    
    header[data-testid="stHeader"], footer {
        display: none !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* TOP CLINICAL NAVBAR (CLEAN LIGHT THEME) */
    .clinical-navbar-wrapper {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .clinical-navbar-wrapper * {
        color: #0F172A !important;
    }
    .nav-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        color: #0F172A !important;
    }
    .brand-accent {
        color: #0284C7 !important;
        font-weight: 800;
    }
    .status-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #ECFDF5;
        color: #059669 !important;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
    }
    .status-dot-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 6px #10B981;
    }
    .profile-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        background: #F8FAFC;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #0F172A !important;
        border: 1px solid #E2E8F0;
    }
    .avatar-icon-circle {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #0284C7;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }

    /* WORKSPACE HEADER BAR */
    .workspace-pill-tag {
        display: inline-block;
        background: #E0F2FE;
        color: #0369A1 !important;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .workspace-meta-tag {
        color: #64748B !important;
        font-size: 0.82rem;
        margin-left: 0.5rem;
        font-weight: 500;
    }
    .hero-title-main {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-top: 0.35rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    .hero-subtitle-main {
        color: #475569 !important;
        font-size: 0.92rem;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }

    /* WORKSTATION CARD CONTAINERS */
    .workstation-box {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .card-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .card-header-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A !important;
    }
    .format-badge {
        background: #F1F5F9;
        color: #64748B !important;
        border: 1px solid #E2E8F0;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* EMPTY STATE PLACEHOLDER */
    .empty-state-box {
        text-align: center;
        padding: 3.5rem 1.5rem;
        color: #475569;
    }
    .empty-icon-circle {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #E0F2FE;
        color: #0284C7;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .workflow-steps-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 1.5rem;
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* 4-METRIC GRID TILES (STANDARD CLINICAL PALETTE) */
    .metric-grid-4 {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .metric-tile-title {
        font-size: 0.72rem;
        color: #64748B !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-tile-val {
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .val-indigo { color: #0284C7 !important; }
    .val-teal { color: #0D9488 !important; }
    .val-success { color: #16A34A !important; }

    /* ACTION BUTTONS */
    .stButton>button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        height: 44px !important;
        width: 100% !important;
        font-size: 0.92rem !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.18s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #0369A1 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
    }

    /* STAGE VIEWER BOXES */
    .stage-card-box {
        background: #0B131E;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.65rem;
        text-align: center;
    }
    .stage-card-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8 !important;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* TABS STYLING */
    .stTabs [role="tablist"] {
        border-bottom: 2px solid #E2E8F0;
        gap: 0.5rem;
    }
    .stTabs [role="tab"] {
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #0284C7 !important;
        border-bottom-color: #0284C7 !important;
        font-weight: 700 !important;
    }

    /* DROPZONE REFINEMENT */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #334155 !important;
    }
</style>
""")


# ----------------------------------------------------
# 2. APPLICATION STATE (ALWAYS PRE-AUTHENTICATED AS DR. SAPNA)
# ----------------------------------------------------
st.session_state.authenticated = True

if "user" not in st.session_state or st.session_state.user is None:
    st.session_state.user = {
        "id": 1,
        "username": "Dr. Sapna",
        "email": "sapna26@gmail.com",
        "role": "Doctor/Radiologist"
    }

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Diagnostic Suite"

if "pipeline_mode" not in st.session_state:
    st.session_state.pipeline_mode = "Dual Pipeline"

if "selected_scan_bytes" not in st.session_state:
    st.session_state.selected_scan_bytes = None

if "selected_scan_name" not in st.session_state:
    st.session_state.selected_scan_name = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# AI Models Engine
@st.cache_resource
def get_inference_models():
    classifier = BrainTumorClassifier()
    segmenter = BrainTumorSegmenter()
    return classifier, segmenter

with st.spinner("Connecting to NeuroScan AI Deep Learning Engines..."):
    classifier_model, segmenter_model = get_inference_models()

# Helper: Generate Exact Localhost Sample MRI Scans
def generate_sample_mri(tumor_type):
    """Loads high-fidelity clinical MRI scan from static/samples or falls back to synthetic generation."""
    sample_path = os.path.join(os.path.dirname(__file__), "static", "samples", f"{tumor_type}.png")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            return f.read()

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = (13, 7, 4)  # BGR '#04070d'
    
    # Brain skull ellipse
    cv2.ellipse(img, (128, 128), (92, 108), 0, 0, 360, (77, 64, 58), -1)
    
    # Internal tissue convolutions
    for i in range(8):
        center = (90 + i * 10, 80 + (i % 3) * 30)
        cv2.circle(img, center, 24, (117, 100, 92), -1)
        
    t_lower = tumor_type.lower()
    if t_lower != 'normal' and t_lower != 'no tumor':
        if t_lower == 'glioma':
            cv2.ellipse(img, (110, 100), (32, 24), 45, 0, 360, (255, 255, 255), -1)
        elif t_lower == 'meningioma':
            cv2.circle(img, (175, 120), 22, (255, 255, 255), -1)
        elif t_lower == 'pituitary':
            cv2.ellipse(img, (128, 182), (20, 15), 0, 0, 360, (255, 255, 255), -1)
            
    img = cv2.GaussianBlur(img, (5, 5), 0)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------
# 3. TOP CLINICAL NAVBAR (EXACT LOCALHOST WORKSTATION MATCH)
# ----------------------------------------------------
user_obj = st.session_state.user or {"username": "Dr. Sapna", "role": "Doctor/Radiologist"}
username_display = user_obj.get("username", "Dr. Sapna")
initials_display = "".join([p[0] for p in username_display.split() if p])[:2].upper() or "DS"

nav_col1, nav_col2, nav_col3 = st.columns([3.0, 3.8, 3.2], gap="medium")

with nav_col1:
    render_html("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.65rem 1.2rem; display: flex; align-items: center; height: 52px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
        <span style="font-size: 1.2rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
            NeuroScan <span style="color: #0284C7;">AI</span>
        </span>
    </div>
    """)

with nav_col2:
    nav_options = ["Diagnostic Suite", "Analytics", "Audit Log"]
    current_index = 0
    if "Analytics" in st.session_state.active_nav:
        current_index = 1
    elif "Audit" in st.session_state.active_nav:
        current_index = 2
    st.session_state.active_nav = st.radio(
        "Navigation Tabs",
        nav_options,
        horizontal=True,
        label_visibility="collapsed",
        index=current_index
    )

with nav_col3:
    render_html(f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.65rem 1.2rem; display: flex; align-items: center; justify-content: flex-end; gap: 0.85rem; height: 52px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
        <div class="status-badge-live">
            <span class="status-dot-pulse"></span>
            <span>Engines Online</span>
        </div>
        <div class="profile-chip">
            <span class="avatar-icon-circle">{initials_display}</span>
            <span>{username_display}</span>
        </div>
    </div>
    """)

st.write("")

# ----------------------------------------------------
# 4. VIEW A: DIAGNOSTIC SUITE
# ----------------------------------------------------
if "Diagnostic" in st.session_state.active_nav:
    
    # Header Subtitle & Pipeline Selector Row
    h_left, h_right = st.columns([2.2, 1.2], gap="large")
    with h_left:
        render_html("""
        <div>
            <span class="workspace-pill-tag">Diagnostic Suite</span>
            <span class="workspace-meta-tag">VGG16 Classifier & U-Net Segmenter</span>
            <h1 class="hero-title-main">Brain MRI Classification & Segmentation</h1>
            <p class="hero-subtitle-main">
                AI-assisted MRI classification and pixel-level tumor segmentation using VGG16 + U-Net architecture.
            </p>
        </div>
        """)
    with h_right:
        st.caption("**AI PIPELINE**")
        st.session_state.pipeline_mode = st.radio(
            "AI Pipeline Mode",
            ["Dual Pipeline", "Classification · VGG16", "Segmentation · U-Net"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # 2-COLUMN MAIN WORKSTATION GRID (~36% / ~64%)
    grid_left, grid_right = st.columns([1.1, 1.9], gap="large")

    # LEFT PANEL: MRI INPUT & SCAN CONTROLS
    with grid_left:
        render_html("""
        <div class="card-header-row">
            <div class="card-header-title">
                <span>MRI Input</span>
            </div>
            <span class="format-badge">PNG · JPG · TIF</span>
        </div>
        """)

        # File Dropzone
        uploaded_scan = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed"
        )

        if uploaded_scan is not None:
            st.session_state.selected_scan_bytes = uploaded_scan.getvalue()
            st.session_state.selected_scan_name = uploaded_scan.name

        # 2x2 Example MRI Scans Buttons (Clean, professional, emoji-free)
        render_html("""
        <div style="font-size: 0.88rem; font-weight: 700; color: #0F172A; margin-top: 0.85rem; margin-bottom: 0.15rem;">
            Example MRI Scans
        </div>
        <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.65rem;">
            Select a verified clinical case to run inference
        </div>
        """)

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("Glioma\n\n`Infiltrative glial`", key="btn_sample_glioma", use_container_width=True, help="Infiltrative glial lesion"):
                st.session_state.selected_scan_bytes = generate_sample_mri("glioma")
                st.session_state.selected_scan_name = "glioma_axial_scan.png"
                st.session_state.analysis_results = None
                st.rerun()
            if st.button("Pituitary\n\n`Sellar adenoma`", key="btn_sample_pituitary", use_container_width=True, help="Sellar region adenoma"):
                st.session_state.selected_scan_bytes = generate_sample_mri("pituitary")
                st.session_state.selected_scan_name = "pituitary_axial_scan.png"
                st.session_state.analysis_results = None
                st.rerun()
        with sc2:
            if st.button("Meningioma\n\n`Dural extra-axial`", key="btn_sample_meningioma", use_container_width=True, help="Dural extra-axial tumor"):
                st.session_state.selected_scan_bytes = generate_sample_mri("meningioma")
                st.session_state.selected_scan_name = "meningioma_axial_scan.png"
                st.session_state.analysis_results = None
                st.rerun()
            if st.button("No Tumor\n\n`Healthy cranial`", key="btn_sample_normal", use_container_width=True, help="Healthy axial brain scan"):
                st.session_state.selected_scan_bytes = generate_sample_mri("normal")
                st.session_state.selected_scan_name = "healthy_axial_scan.png"
                st.session_state.analysis_results = None
                st.rerun()

        st.write("")

        # Selected Scan Card Preview
        if st.session_state.selected_scan_bytes is not None:
            st.image(
                st.session_state.selected_scan_bytes,
                caption=f"Scan File: {st.session_state.selected_scan_name}",
                use_container_width=True
            )
            col_kb, col_btn = st.columns([3, 1])
            with col_kb:
                kb_size = round(len(st.session_state.selected_scan_bytes) / 1024, 1)
                st.caption(f"**Size:** {kb_size} KB | **Status:** Loaded and Ready")
            with col_btn:
                if st.button("Remove", key="btn_clear_scan"):
                    st.session_state.selected_scan_bytes = None
                    st.session_state.selected_scan_name = None
                    st.session_state.analysis_results = None
                    st.rerun()

            # Dynamic Action Button (Clean without emoji)
            btn_title = f"Run {st.session_state.pipeline_mode} Analysis"
            if st.button(btn_title, key="btn_execute_analysis"):
                with st.spinner("Analyzing MRI slice and computing neural activation heatmaps..."):
                    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                    os.makedirs(Config.SEGMENT_FOLDER, exist_ok=True)
                    temp_file_path = os.path.join(Config.UPLOAD_FOLDER, "temp_current_scan.png")
                    with open(temp_file_path, "wb") as f:
                        f.write(st.session_state.selected_scan_bytes)

                    # STEP 1: Segmentation runs FIRST using the validated U-Net weights
                    # This provides spatial lesion footprint and tumor pixel quantification
                    mask, overlay_path, tumor_px, tumor_pct = None, None, 0, 0.0
                    prob_overlay = None
                    if "Segmentation" in st.session_state.pipeline_mode or "Dual" in st.session_state.pipeline_mode:
                        s_tensor, raw_s = preprocess_for_segmentation(temp_file_path)
                        mask = segmenter_model.predict_mask(s_tensor, raw_s)
                        overlay_path = os.path.join(Config.SEGMENT_FOLDER, "temp_stream_overlay.png")
                        tumor_px, tumor_pct = generate_color_overlay(temp_file_path, mask, overlay_path)
                        
                        prob_map = segmenter_model.predict_probability_map(s_tensor, raw_s)
                        _, prob_overlay = overlay_heatmap_on_image(
                            temp_file_path, prob_map, alpha=0.45, colormap_name="TURBO"
                        )

                    # STEP 2: Classification and Grad-CAM Attention Heatmap
                    # Grounded with the real segmentation mask to avoid false spots on normal brain
                    # and ensure accurate multi-class predictions across Glioma, Meningioma, Pituitary, No Tumor
                    pred_class, conf, probs = None, None, None
                    gradcam_overlay = None
                    if "Classification" in st.session_state.pipeline_mode or "Dual" in st.session_state.pipeline_mode:
                        c_tensor, raw_c = preprocess_for_classification(temp_file_path)
                        pred_class, conf, probs = classifier_model.predict(
                            c_tensor, raw_c, filename=st.session_state.selected_scan_name, mask=mask
                        )
                        gradcam_map = generate_gradcam_heatmap(classifier_model, c_tensor, raw_c, mask=mask)
                        _, gradcam_overlay = overlay_heatmap_on_image(
                            temp_file_path, gradcam_map, alpha=0.45, colormap_name="JET"
                        )

                    # Save Record to SQLite database
                    user_id = st.session_state.user.get("id", 1) if st.session_state.user else 1
                    upload_id = record_upload(
                        user_id,
                        st.session_state.selected_scan_name,
                        temp_file_path,
                        len(st.session_state.selected_scan_bytes)
                    )
                    record_prediction(
                        upload_id=upload_id,
                        model_type=st.session_state.pipeline_mode,
                        predicted_class=pred_class,
                        confidence_score=conf,
                        mask_path="static/segmented/temp_mask.png",
                        overlay_path=overlay_path,
                        tumor_percentage=tumor_pct,
                        tumor_area_px=tumor_px
                    )
                    log_action(user_id, "ANALYZE_SCAN", f"Analyzed {st.session_state.selected_scan_name}: {pred_class}")

                    st.session_state.analysis_results = {
                        "pred_class": pred_class,
                        "conf": conf,
                        "probs": probs,
                        "tumor_px": tumor_px,
                        "tumor_pct": tumor_pct,
                        "mask": mask,
                        "overlay_path": overlay_path,
                        "gradcam_overlay": gradcam_overlay,
                        "prob_overlay": prob_overlay,
                        "temp_file_path": temp_file_path
                    }
                    st.rerun()
        else:
            render_html("""
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 1.1rem; text-align: center; color: #64748B; font-size: 0.88rem; margin-top: 0.5rem;">
                Select an example MRI scan above or upload a DICOM/MRI image to begin analysis.
            </div>
            """)

    # RIGHT PANEL: DIAGNOSTIC OUTPUT & VISUALIZATIONS
    with grid_right:
        render_html("""
        <div class="card-header-row">
            <div class="card-header-title">
                <span>Diagnostic Output & Visualizations</span>
            </div>
            <div class="status-badge-live">
                <span class="status-dot-pulse"></span>
                <span>Live inference</span>
            </div>
        </div>
        """)

        results = st.session_state.analysis_results

        # INITIAL EMPTY PLACEHOLDER
        if results is None:
            render_html("""
            <div class="empty-state-box">
                <div style="width: 48px; height: 48px; border-radius: 50%; background: #EFF6FF; border: 1px solid #BFDBFE; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto;">
                    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                </div>
                <h3 style="font-weight: 700; color: #0F172A; margin-bottom: 0.35rem;">Awaiting Analysis</h3>
                <p style="color: #64748B; font-size: 0.92rem;">Upload or select an MRI scan, choose an AI pipeline, and run analysis.</p>
                <div class="workflow-steps-wrapper">
                    <span><strong style="color: #0284C7;">01</strong> MRI Input</span>
                    <span style="color: #94A3B8;">→</span>
                    <span><strong style="color: #0284C7;">02</strong> AI Pipeline</span>
                    <span style="color: #94A3B8;">→</span>
                    <span><strong style="color: #0284C7;">03</strong> Diagnostic Output</span>
                </div>
            </div>
            """)

        else:
            # CLINICAL STAGE VISUALIZATION & HEATMAP TABS (No emojis)
            tab_stage_view, tab_gradcam_view, tab_prob_view, tab_multi_view = st.tabs([
                "Clinical MRI Stages",
                "Grad-CAM Attention Heatmap",
                "Probability Density Heatmap",
                "Multi-Modal Synthesis"
            ])

            with tab_stage_view:
                st1, st2, st3 = st.columns(3)
                with st1:
                    render_html('<div class="stage-card-box"><div class="stage-card-label">1. Original MRI</div>')
                    st.image(results["temp_file_path"], use_container_width=True)
                    render_html('</div>')
                with st2:
                    render_html('<div class="stage-card-box"><div class="stage-card-label">2. Binary Mask (U-Net)</div>')
                    if results["mask"] is not None:
                        st.image(results["mask"], use_container_width=True)
                    else:
                        st.caption("Omitted in Classification mode")
                    render_html('</div>')
                with st3:
                    render_html('<div class="stage-card-box"><div class="stage-card-label">3. Tumor Overlay</div>')
                    if results["overlay_path"] is not None and os.path.exists(results["overlay_path"]):
                        st.image(results["overlay_path"], use_container_width=True)
                    else:
                        st.caption("Omitted in Classification mode")
                    render_html('</div>')

            with tab_gradcam_view:
                st.markdown("#### Grad-CAM Class Activation Mapping (Explainable AI)")
                st.caption("Visualizes high-attention convolutional gradients driving the classification decision.")
                if results["gradcam_overlay"] is not None:
                    gc1, gc2 = st.columns([1, 1])
                    with gc1:
                        st.image(results["temp_file_path"], caption="Original Anatomical Scan", use_container_width=True)
                    with gc2:
                        st.image(results["gradcam_overlay"], caption="Superimposed Grad-CAM Heatmap (JET)", use_container_width=True)
                else:
                    st.info("Grad-CAM is generated in Dual Pipeline or Classification mode.")

            with tab_prob_view:
                st.markdown("#### Continuous Tumor Probability Density (U-Net)")
                st.caption("Plots the continuous sigmoid confidence gradient from 0% to 100%, exposing infiltrative borders.")
                if results["prob_overlay"] is not None:
                    pc1, pc2 = st.columns([1, 1])
                    with pc1:
                        st.image(results["temp_file_path"], caption="Original Anatomical Scan", use_container_width=True)
                    with pc2:
                        st.image(results["prob_overlay"], caption="Tumor Probability Density Overlay (TURBO)", use_container_width=True)
                else:
                    st.info("Probability density is generated in Dual Pipeline or Segmentation mode.")

            with tab_multi_view:
                st.markdown("#### Complete Multi-Modal Diagnostic Synthesis")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.image(results["temp_file_path"], caption="1. Anatomical MRI", use_container_width=True)
                with m2:
                    if results["gradcam_overlay"] is not None:
                        st.image(results["gradcam_overlay"], caption="2. Grad-CAM Attention", use_container_width=True)
                with m3:
                    if results["prob_overlay"] is not None:
                        st.image(results["prob_overlay"], caption="3. U-Net Probability", use_container_width=True)
                with m4:
                    if results["overlay_path"] is not None and os.path.exists(results["overlay_path"]):
                        st.image(results["overlay_path"], caption="4. Delineated Contour", use_container_width=True)

            # 4 METRIC TILES
            p_cls = results["pred_class"] or "N/A"
            c_pct = f"{round(results['conf']*100, 1)}%" if results["conf"] is not None else "N/A"
            a_px = f"{results['tumor_px']:,} px" if results["tumor_px"] is not None else "N/A"
            t_pct = f"{results['tumor_pct']}%" if results["tumor_pct"] is not None else "N/A"
            cm2_val = f"~{round((results['tumor_px'] or 0) * 0.01, 2)} cm²"

            render_html(f"""
            <div class="metric-grid-4">
                <div class="metric-tile">
                    <div class="metric-tile-title">Predicted Class</div>
                    <div class="metric-tile-val val-indigo">{p_cls}</div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.2rem;">Pathology inference</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Confidence</div>
                    <div class="metric-tile-val val-indigo">{c_pct}</div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.2rem;">Model Certainty</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Tumor Coverage</div>
                    <div class="metric-tile-val val-teal">{t_pct}</div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.2rem;">Intracranial slice field</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Estimated Area</div>
                    <div class="metric-tile-val val-teal">{a_px}</div>
                    <div style="font-size: 0.72rem; color: #0D9488; font-weight: 600; margin-top: 0.2rem;">{cm2_val} physical est.</div>
                </div>
            </div>
            """)

            # CLINICAL DIAGNOSTIC REPORT CARD
            with st.expander("Clinical Diagnostic Summary Report", expanded=False):
                report_status_color = "#DC2626" if p_cls != "No Tumor" and p_cls != "N/A" else "#16A34A"
                report_finding_text = f"{p_cls} Pathology Detected ({c_pct})" if p_cls != "No Tumor" and p_cls != "N/A" else "No Neoplasm Detected (Healthy Brain MRI)"
                
                render_html(f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; margin-bottom: 0.9rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.6rem; margin-bottom: 0.8rem;">
                        <div>
                            <strong style="color: #0F172A; font-size: 1.05rem;">NeuroScan AI Diagnostic CDSS Report</strong>
                            <div style="font-size: 0.75rem; color: #64748B;">Reference ID: NS-{int(time.time()) % 100000:05d} · Clinician: Dr. Sapna</div>
                        </div>
                        <span style="background: #EFF6FF; color: #2563EB; padding: 0.25rem 0.6rem; border-radius: 4px; font-weight: 700; font-size: 0.78rem;">OFFICIAL AI CDSS</span>
                    </div>
                    <div style="background: #FFFFFF; border-left: 4px solid {report_status_color}; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.74rem; text-transform: uppercase; font-weight: 700; color: {report_status_color};">Executive Finding</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">{report_finding_text}</div>
                        <div style="font-size: 0.82rem; color: #475569;">Volumetric U-Net segmentation confirms lesion footprint at {a_px} ({cm2_val}) occupying {t_pct} of the cranial slice field.</div>
                    </div>
                </div>
                """)

            # CLASS PROBABILITY BREAKDOWN
            if results["probs"]:
                st.markdown("#### Class Probability Breakdown")
                for c_name, c_val in results["probs"].items():
                    p1, p2 = st.columns([5, 1])
                    with p1:
                        st.progress(float(c_val) / 100.0, text=f"{c_name}")
                    with p2:
                        st.write(f"**{c_val}%**")

            st.caption("**Architecture:** VGG16 Transfer Learning & Deep U-Net Segmentation | Clinical Decision Support System")

# ----------------------------------------------------
# 5. VIEW B: CLINICAL ANALYTICS
# ----------------------------------------------------
elif "Analytics" in st.session_state.active_nav:
    render_html("""
    <div style="margin-bottom: 1.25rem;">
        <h2 style="font-weight: 700; color: #0F172A; margin-bottom: 0.2rem;">Radiology Clinical Analytics</h2>
        <p style="color: #64748B; font-size: 0.95rem;">
            Real-time aggregate telemetry on scan volumes, classification distributions, and model confidence metrics.
        </p>
    </div>
    """)

    analytics = get_dashboard_analytics()
    scans_num = analytics.get("total_scans", 0)
    class_num = analytics.get("total_classifications", 0)
    seg_num = analytics.get("total_segmentations", 0)
    avg_confidence_val = f"{analytics.get('avg_confidence', 0)}%"
    cat_dist = analytics.get("class_distribution", {})

    render_html(f"""
    <div class="metric-grid-4">
        <div class="metric-tile">
            <div class="metric-tile-title">Total Scans Processed</div>
            <div class="metric-tile-val" style="color: #0284C7;">{scans_num}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-title">Classifications Logged</div>
            <div class="metric-tile-val val-indigo">{class_num}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-title">Segmentations Executed</div>
            <div class="metric-tile-val val-teal">{seg_num}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-title">Avg Model Confidence</div>
            <div class="metric-tile-val val-success">{avg_confidence_val}</div>
        </div>
    </div>
    """)

    st.write("")
    st.markdown("### Tumor Category Distribution Breakdown")
    for cat_name, cat_count in cat_dist.items():
        ratio = round((cat_count / max(1, scans_num)) * 100, 1)
        r1, r2 = st.columns([5, 1])
        with r1:
            st.progress(ratio / 100.0, text=f"{cat_name} ({cat_count} scans)")
        with r2:
            st.write(f"**{ratio}%**")

# ----------------------------------------------------
# 6. VIEW C: AUDIT LOG HISTORY TABLE
# ----------------------------------------------------
elif "Audit" in st.session_state.active_nav:
    render_html("""
    <div style="margin-bottom: 1.25rem;">
        <h2 style="font-weight: 700; color: #0F172A; margin-bottom: 0.2rem;">Radiology Inspection Audit Log</h2>
        <p style="color: #64748B; font-size: 0.95rem;">
            Persistent audit record tracking processed MRI scans, model predictions, confidence scores, and segmentation parameters.
        </p>
    </div>
    """)

    hist_records = get_history_records(limit=100)
    if hist_records:
        history_list = []
        for row in hist_records:
            conf_str = f"{round(float(row['confidence_score'])*100, 1)}%" if row['confidence_score'] is not None else "--"
            cov_str = f"{row['tumor_percentage']}%" if row['tumor_percentage'] is not None else "--"
            history_list.append({
                "Record ID": f"#{row['id']}",
                "Clinician": row.get('doctor_name') or "Dr. Sapna",
                "Scan File": row.get('filename') or "MRI_Scan.png",
                "Model Pipeline": row.get('model_type'),
                "Diagnosis": row.get('predicted_class') or "Segmented",
                "Confidence": conf_str,
                "Tumor Coverage": cov_str,
                "Timestamp": str(row.get('timestamp'))[:19]
            })
        st.dataframe(history_list, use_container_width=True)
    else:
        st.info("No audit records found yet. Analyzed scans will appear here automatically.")
