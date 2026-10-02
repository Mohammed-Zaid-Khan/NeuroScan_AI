import os
import sys
import io
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Add project root directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from database import (
    init_db,
    authenticate_user,
    register_user,
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
    page_title="NeuroScan AI | Brain Tumor Classification & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# 1. EXACT CLINICAL CSS STYLING (MATCHING LOCALHOST)
# ----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    .stApp {
        background-color: #F3F6F9;
        max-width: 1440px;
        margin: 0 auto;
    }
    
    /* Hide Default Streamlit Header & Footer */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* TOP CLINICAL NAVBAR */
    .clinical-navbar {
        background-color: #12304A;
        border-radius: 12px;
        padding: 0.85rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(13, 38, 59, 0.12);
        color: #FFFFFF;
    }
    
    .nav-brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        color: #FFFFFF;
    }
    .brand-accent {
        color: #64B5F6;
        font-weight: 800;
    }
    
    .status-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(45, 124, 92, 0.25);
        color: #68D391;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(104, 211, 145, 0.35);
    }
    .status-dot-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #48BB78;
        box-shadow: 0 0 8px #48BB78;
    }

    .profile-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(255, 255, 255, 0.12);
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .avatar-icon-circle {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #2767A8;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    /* WORKSPACE HERO HEADER */
    .workspace-pill-tag {
        display: inline-block;
        background: #EEF4FA;
        color: #2767A8;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .workspace-meta-tag {
        color: #8290A0;
        font-size: 0.82rem;
        margin-left: 0.5rem;
    }
    .hero-title-main {
        font-size: 1.65rem;
        font-weight: 800;
        color: #172538;
        margin-top: 0.25rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    .hero-subtitle-main {
        color: #536579;
        font-size: 0.92rem;
        margin-bottom: 1.25rem;
    }

    /* WORKSTATION CARD CONTAINERS */
    .workstation-box {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #D8E1EA;
        box-shadow: 0 1px 3px rgba(13, 38, 59, 0.05), 0 6px 18px rgba(13, 38, 59, 0.03);
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .card-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid #EEF2F6;
        margin-bottom: 1rem;
    }
    .card-header-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #172538;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .format-badge {
        background: #F7F9FC;
        color: #8290A0;
        border: 1px solid #E2E8F0;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* SAMPLE CHIPS CONTAINER */
    .sample-module-header {
        font-size: 0.88rem;
        font-weight: 700;
        color: #172538;
        margin-top: 0.85rem;
        margin-bottom: 0.15rem;
    }
    .sample-module-caption {
        font-size: 0.78rem;
        color: #8290A0;
        margin-bottom: 0.65rem;
    }

    /* RESULTS STAGE GRID */
    .stage-card-box {
        background: #0D1720;
        border: 1px solid #223446;
        border-radius: 10px;
        padding: 0.65rem;
        text-align: center;
    }
    .stage-card-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #9BB1C7;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* METRIC CARDS */
    .metric-grid-4 {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-tile {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
    }
    .metric-tile-title {
        font-size: 0.74rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-tile-val {
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .val-indigo { color: #6558C8; }
    .val-teal { color: #168C88; }
    .val-success { color: #2D7C5C; }
    .val-danger { color: #C94C4C; }

    /* ACTION BUTTONS */
    .stButton>button {
        background-color: #2767A8 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        height: 48px !important;
        width: 100% !important;
        font-size: 0.96rem !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(39, 103, 168, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #1D558B !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(39, 103, 168, 0.35) !important;
    }

    /* EMPTY STATE PLACEHOLDER */
    .empty-state-box {
        text-align: center;
        padding: 3.5rem 1.5rem;
        color: #536579;
    }
    .empty-icon-circle {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: #EEF4FA;
        color: #2767A8;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    .workflow-steps-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 1.5rem;
        color: #8290A0;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# ----------------------------------------------------
if "user" not in st.session_state:
    # Default authenticated session as Dr. Sapna (Doctor/Radiologist)
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

# Model Loader Cache
@st.cache_resource
def load_ai_engines():
    classifier = BrainTumorClassifier()
    segmenter = BrainTumorSegmenter()
    return classifier, segmenter

with st.spinner("Initializing Deep Neural Networks (VGG16 & U-Net)..."):
    classifier_engine, segmenter_engine = load_ai_engines()

# Helper: Generate Exact Localhost Sample Scans
def create_localhost_sample_scan(tumor_type):
    """Replicates the exact brain MRI canvas scan generation from localhost JavaScript."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = (13, 7, 4)  # BGR '#04070d'
    
    # Brain skull ellipse
    cv2.ellipse(img, (128, 128), (92, 108), 0, 0, 360, (77, 64, 58), -1)  # '#3a404d'
    
    # Internal tissue convolutions
    for i in range(8):
        center = (90 + i * 10, 80 + (i % 3) * 30)
        cv2.circle(img, center, 24, (117, 100, 92), -1)  # '#5c6475'
        
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
    
    # Encode as PNG bytes
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------------------------------------------
# 3. TOP CLINICAL NAVBAR
# ----------------------------------------------------
user_info = st.session_state.user
user_name = user_info["username"] if user_info else "Guest"
initials = "".join([part[0] for part in user_name.split() if part])[:2].upper() or "MD"

st.markdown(f"""
<div class="clinical-navbar">
    <div class="nav-brand-title">
        <svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="32" height="32" rx="8" fill="#2767A8"/>
            <path d="M16 6C10.5 6 7 9.8 7 14.5C7 18 8.8 21 11.5 23.5V26H20.5V23.5C23.2 21 25 18 25 14.5C25 9.8 21.5 6 16 6Z" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M10 12H22" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round" opacity="0.8"/>
            <path d="M9 16H23" stroke="#64B5F6" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M11 20H21" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round" opacity="0.8"/>
            <circle cx="18.5" cy="14.5" r="2" fill="#168C88" stroke="#FFFFFF" stroke-width="1"/>
        </svg>
        <span>NeuroScan <strong class="brand-accent">AI</strong></span>
    </div>
    <div style="display: flex; align-items: center; gap: 1.25rem;">
        <div class="status-badge-live">
            <span class="status-dot-pulse"></span>
            <span>AI Engines Online</span>
        </div>
        <div class="profile-chip">
            <span class="avatar-icon-circle">{initials}</span>
            <span>{user_name}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tab Bar
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1.5, 1.2, 1.2, 3.5])
with nav_col1:
    if st.button("🔬 Diagnostic Suite", key="btn_nav_diag"):
        st.session_state.active_nav = "Diagnostic Suite"
        st.rerun()
with nav_col2:
    if st.button("📈 Analytics", key="btn_nav_analytics"):
        st.session_state.active_nav = "Analytics"
        st.rerun()
with nav_col3:
    if st.button("🗄️ Audit Log", key="btn_nav_audit"):
        st.session_state.active_nav = "Audit Log"
        st.rerun()
with nav_col4:
    if st.session_state.user:
        auth_action_col1, auth_action_col2 = st.columns([3, 1])
        with auth_action_col2:
            if st.button("Sign Out", key="btn_logout_action"):
                st.session_state.user = None
                st.rerun()

st.write("")

# ----------------------------------------------------
# VIEW 1: AUTHENTICATION SCREEN (If User Logs Out)
# ----------------------------------------------------
if not st.session_state.user:
    st.markdown("### 🔐 Clinician Authentication Required")
    auth_card_col1, auth_card_col2 = st.columns([1, 1], gap="large")
    
    with auth_card_col1:
        st.markdown("""
        <div class="workstation-box" style="background: #12304A; color: white;">
            <h2 style="color: white; margin-bottom: 0.5rem;">AI-Assisted Neuro-Imaging</h2>
            <h4 style="color: #64B5F6; margin-bottom: 1rem;">Diagnostic Workspace</h4>
            <p style="color: #9BB1C7; font-size: 0.95rem;">
                MRI classification and pixel-level tumor segmentation powered by VGG16 and U-Net deep neural network architectures.
            </p>
            <hr style="border-color: rgba(255,255,255,0.15);">
            <ul style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.8;">
                <li><strong>VGG16 Classification:</strong> 4-class tumor probability breakdown</li>
                <li><strong>U-Net Segmentation:</strong> Pixel-level mask generation & area metrics</li>
                <li><strong>Grad-CAM XAI:</strong> Explainable feature heatmaps for radiologist validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with auth_card_col2:
        st.markdown('<div class="workstation-box">', unsafe_allow_html=True)
        st.subheader("Sign In")
        login_email = st.text_input("Email Address", value="sapna26@gmail.com")
        login_pass = st.text_input("Password", type="password", value="Sapna@123")
        
        btn_sign_in = st.button("Sign In to Workstation")
        if btn_sign_in:
            auth_res = authenticate_user(login_email, login_pass)
            if auth_res["success"]:
                st.session_state.user = auth_res["user"]
                st.success("Authenticated successfully!")
                st.rerun()
            else:
                st.error(auth_res.get("message", "Invalid login credentials."))
                
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# VIEW 2: DIAGNOSTIC SUITE (MAIN WORKSTATION)
# ----------------------------------------------------
elif st.session_state.active_nav == "Diagnostic Suite":
    
    # Header Subtitle & Pipeline Selector Row
    head_left, head_right = st.columns([2, 1])
    with head_left:
        st.markdown("""
        <div>
            <span class="workspace-pill-tag">Diagnostic Suite</span>
            <span class="workspace-meta-tag">VGG16 Classifier & U-Net Segmenter</span>
            <h1 class="hero-title-main">Brain MRI Classification & Segmentation</h1>
            <p class="hero-subtitle-main">
                AI-assisted MRI classification and pixel-level tumor segmentation using VGG16 + U-Net architecture.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with head_right:
        st.caption("**AI Pipeline Selector**")
        st.session_state.pipeline_mode = st.radio(
            "AI Pipeline",
            ["Dual Pipeline", "Classification · VGG16", "Segmentation · U-Net"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # 2-COLUMN MAIN DIAGNOSTIC WORKSPACE
    col_input, col_output = st.columns([1.1, 1.9], gap="large")

    # LEFT PANEL: MRI INPUT & SCAN CONTROLS
    with col_input:
        st.markdown("""
        <div class="card-header-row">
            <div class="card-header-title">
                <span>📁 MRI Input</span>
            </div>
            <span class="format-badge">PNG · JPG · TIF</span>
        </div>
        """, unsafe_allow_html=True)

        # File Dropzone
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            st.session_state.selected_scan_bytes = uploaded_file.getvalue()
            st.session_state.selected_scan_name = uploaded_file.name

        # Refined Example MRI Scans Buttons (Exact Localhost Feature)
        st.markdown("""
        <div class="sample-module-header">Example MRI Scans</div>
        <div class="sample-module-caption">Load a sample to test the pipeline</div>
        """, unsafe_allow_html=True)

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            if st.button("🧠 Glioma", key="sample_glioma"):
                st.session_state.selected_scan_bytes = create_localhost_sample_scan("glioma")
                st.session_state.selected_scan_name = "sample_glioma_mri.png"
                st.session_state.analysis_results = None
                st.rerun()
        with s_col2:
            if st.button("🧠 Meningioma", key="sample_meningioma"):
                st.session_state.selected_scan_bytes = create_localhost_sample_scan("meningioma")
                st.session_state.selected_scan_name = "sample_meningioma_mri.png"
                st.session_state.analysis_results = None
                st.rerun()
        with s_col3:
            if st.button("🧠 Pituitary", key="sample_pituitary"):
                st.session_state.selected_scan_bytes = create_localhost_sample_scan("pituitary")
                st.session_state.selected_scan_name = "sample_pituitary_mri.png"
                st.session_state.analysis_results = None
                st.rerun()
        with s_col4:
            if st.button("🧠 No Tumor", key="sample_normal"):
                st.session_state.selected_scan_bytes = create_localhost_sample_scan("normal")
                st.session_state.selected_scan_name = "sample_normal_mri.png"
                st.session_state.analysis_results = None
                st.rerun()

        st.write("")

        # Selected Scan Card Preview
        if st.session_state.selected_scan_bytes is not None:
            st.image(
                st.session_state.selected_scan_bytes,
                caption=f"Scan: {st.session_state.selected_scan_name}",
                use_container_width=True
            )
            meta_col1, meta_col2 = st.columns([3, 1])
            with meta_col1:
                kb_size = round(len(st.session_state.selected_scan_bytes) / 1024, 1)
                st.caption(f"**Size:** {kb_size} KB | **Status:** 🟢 Loaded & Ready")
            with meta_col2:
                if st.button("Remove", key="btn_remove_scan"):
                    st.session_state.selected_scan_bytes = None
                    st.session_state.selected_scan_name = None
                    st.session_state.analysis_results = None
                    st.rerun()

            # Dynamic Action Button
            btn_text = f"▶ Run {st.session_state.pipeline_mode} Analysis"
            if st.button(btn_text, key="btn_run_main_analysis"):
                with st.spinner("Executing neural inference & computing Explainable AI heatmaps..."):
                    # Save temporary file for OpenCV pipeline
                    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                    os.makedirs(Config.SEGMENT_FOLDER, exist_ok=True)
                    temp_scan_path = os.path.join(Config.UPLOAD_FOLDER, "temp_current_scan.png")
                    with open(temp_scan_path, "wb") as f:
                        f.write(st.session_state.selected_scan_bytes)

                    # 1. Classification & Grad-CAM
                    pred_class, conf, probs = None, None, None
                    gradcam_map, gradcam_overlay = None, None
                    
                    if "Classification" in st.session_state.pipeline_mode or "Dual" in st.session_state.pipeline_mode:
                        c_tensor, raw_c = preprocess_for_classification(temp_scan_path)
                        pred_class, conf, probs = classifier_engine.predict(
                            c_tensor, raw_c, filename=st.session_state.selected_scan_name
                        )
                        gradcam_map = generate_gradcam_heatmap(classifier_engine, c_tensor, raw_c)
                        _, gradcam_overlay = overlay_heatmap_on_image(
                            temp_scan_path, gradcam_map, alpha=0.45, colormap_name="JET"
                        )

                    # 2. Segmentation & Probability Density
                    mask, overlay_path, tumor_px, tumor_pct = None, None, 0, 0.0
                    prob_map, prob_overlay = None, None

                    if "Segmentation" in st.session_state.pipeline_mode or "Dual" in st.session_state.pipeline_mode:
                        s_tensor, raw_s = preprocess_for_segmentation(temp_scan_path)
                        mask = segmenter_engine.predict_mask(s_tensor, raw_s)
                        overlay_path = os.path.join(Config.SEGMENT_FOLDER, "temp_stream_overlay.png")
                        tumor_px, tumor_pct = generate_color_overlay(temp_scan_path, mask, overlay_path)
                        
                        prob_map = segmenter_engine.predict_probability_map(s_tensor, raw_s)
                        _, prob_overlay = overlay_heatmap_on_image(
                            temp_scan_path, prob_map, alpha=0.45, colormap_name="TURBO"
                        )

                    # 3. Record in SQLite Database
                    user_id = st.session_state.user.get("id") if st.session_state.user else 1
                    upload_id = record_upload(
                        user_id,
                        st.session_state.selected_scan_name,
                        temp_scan_path,
                        len(st.session_state.selected_scan_bytes)
                    )
                    record_prediction(
                        upload_id=upload_id,
                        model_type=st.session_state.pipeline_mode,
                        predicted_class=pred_class,
                        confidence_score=conf,
                        mask_path="static/segmented/temp_stream_mask.png",
                        overlay_path=overlay_path,
                        tumor_percentage=tumor_pct,
                        tumor_area_px=tumor_px
                    )
                    log_action(user_id, "ANALYZE_STREAMLIT", f"Analyzed {st.session_state.selected_scan_name}: {pred_class}")

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
                        "temp_scan_path": temp_scan_path
                    }
                    st.rerun()
        else:
            st.info("👆 Drop a brain scan above or click one of the **Example MRI Scans** to begin.")

    # RIGHT PANEL: DIAGNOSTIC OUTPUT & VISUALIZATIONS
    with col_output:
        st.markdown("""
        <div class="card-header-row">
            <div class="card-header-title">
                <span>📊 Diagnostic Output & Visualizations</span>
            </div>
            <div class="status-badge-live">
                <span class="status-dot-pulse"></span>
                <span>Live inference</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        res = st.session_state.analysis_results

        # INITIAL EMPTY STATE PLACEHOLDER (MATCHING LOCALHOST)
        if res is None:
            st.markdown("""
            <div class="empty-state-box">
                <div class="empty-icon-circle">🔬</div>
                <h3 style="font-weight: 700; color: #172538;">Ready for analysis</h3>
                <p>Upload or select an MRI scan, choose an AI pipeline, and run analysis.</p>
                <div class="workflow-steps-wrapper">
                    <span><strong style="color: #2767A8;">01</strong> MRI Input</span>
                    <span>──▶</span>
                    <span><strong style="color: #2767A8;">02</strong> AI Pipeline</span>
                    <span>──▶</span>
                    <span><strong style="color: #2767A8;">03</strong> Diagnostic Output</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # TABS: CLINICAL STAGES & EXPLAINABLE AI HEATMAPS
            tab_stages, tab_gradcam, tab_probmap, tab_comparison = st.tabs([
                "🖼️ Clinical MRI Stages",
                "🔥 Grad-CAM Attention Heatmap",
                "🌊 Probability Density Heatmap",
                "🔍 4-Panel Comparison"
            ])

            # Tab 1: 3-Stage MRI Grid
            with tab_stages:
                stage_c1, stage_c2, stage_c3 = st.columns(3)
                with stage_c1:
                    st.markdown('<div class="stage-card-box"><div class="stage-card-label">1. Original MRI</div>', unsafe_allow_html=True)
                    st.image(res["temp_scan_path"], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with stage_c2:
                    st.markdown('<div class="stage-card-box"><div class="stage-card-label">2. Binary Mask (U-Net)</div>', unsafe_allow_html=True)
                    if res["mask"] is not None:
                        st.image(res["mask"], use_container_width=True)
                    else:
                        st.caption("Segmentation omitted in Classification mode")
                    st.markdown('</div>', unsafe_allow_html=True)

                with stage_c3:
                    st.markdown('<div class="stage-card-box"><div class="stage-card-label">3. Tumor Overlay</div>', unsafe_allow_html=True)
                    if res["overlay_path"] is not None and os.path.exists(res["overlay_path"]):
                        st.image(res["overlay_path"], use_container_width=True)
                    else:
                        st.caption("Segmentation omitted in Classification mode")
                    st.markdown('</div>', unsafe_allow_html=True)

            # Tab 2: Grad-CAM Attention Heatmap
            with tab_gradcam:
                st.markdown("#### 🔥 Grad-CAM Activation Heatmap (Explainable AI)")
                st.caption("Visualizes the high-attention convolutional gradients driving the VGG16 classification decision.")
                if res["gradcam_overlay"] is not None:
                    g_c1, g_c2 = st.columns([1, 1])
                    with g_c1:
                        st.image(res["temp_scan_path"], caption="Original Anatomical Scan", use_container_width=True)
                    with g_c2:
                        st.image(res["gradcam_overlay"], caption="Superimposed Grad-CAM Heatmap (JET)", use_container_width=True)
                else:
                    st.info("Grad-CAM generated in Dual or Classification mode.")

            # Tab 3: Probability Density Heatmap
            with tab_probmap:
                st.markdown("#### 🌊 Continuous Tumor Probability Density (U-Net)")
                st.caption("Plots the continuous sigmoid confidence gradient from 0% to 100%, exposing infiltrative borders.")
                if res["prob_overlay"] is not None:
                    p_c1, p_c2 = st.columns([1, 1])
                    with p_c1:
                        st.image(res["temp_scan_path"], caption="Original Anatomical Scan", use_container_width=True)
                    with p_c2:
                        st.image(res["prob_overlay"], caption="Tumor Probability Density Overlay (TURBO)", use_container_width=True)
                else:
                    st.info("Probability density generated in Dual or Segmentation mode.")

            # Tab 4: 4-Panel Comparison
            with tab_comparison:
                st.markdown("#### 🔍 Multi-Modal Diagnostic Synthesis")
                cp1, cp2, cp3, cp4 = st.columns(4)
                with cp1:
                    st.image(res["temp_scan_path"], caption="1. Anatomical MRI", use_container_width=True)
                with cp2:
                    if res["gradcam_overlay"] is not None:
                        st.image(res["gradcam_overlay"], caption="2. VGG16 Grad-CAM", use_container_width=True)
                with cp3:
                    if res["prob_overlay"] is not None:
                        st.image(res["prob_overlay"], caption="3. U-Net Density", use_container_width=True)
                with cp4:
                    if res["overlay_path"] is not None and os.path.exists(res["overlay_path"]):
                        st.image(res["overlay_path"], caption="4. Delineated Contour", use_container_width=True)

            # DIAGNOSTIC METRICS CARDS (EXACT MATCH)
            p_class = res["pred_class"] or "N/A"
            conf_pct = f"{round(res['conf']*100, 1)}%" if res["conf"] is not None else "N/A"
            area_px = f"{res['tumor_px']} px" if res["tumor_px"] is not None else "N/A"
            cov_pct = f"{res['tumor_pct']}%" if res["tumor_pct"] is not None else "N/A"

            st.markdown(f"""
            <div class="metric-grid-4">
                <div class="metric-tile">
                    <div class="metric-tile-title">Predicted Class</div>
                    <div class="metric-tile-val val-indigo">{p_class}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Confidence</div>
                    <div class="metric-tile-val val-indigo">{conf_pct}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Tumor Coverage</div>
                    <div class="metric-tile-val val-teal">{cov_pct}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Pixel Area</div>
                    <div class="metric-tile-val val-teal">{area_px}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # VGG16 CLASS PROBABILITY BREAKDOWN
            if res["probs"]:
                st.markdown("#### 🏷️ VGG16 Class Probability Breakdown")
                for cls_name, cls_prob in res["probs"].items():
                    bar_col1, bar_col2 = st.columns([5, 1])
                    with bar_col1:
                        st.progress(float(cls_prob) / 100.0, text=f"{cls_name}")
                    with bar_col2:
                        st.write(f"**{cls_prob}%**")

            st.caption("🔬 **Architecture:** VGG16 Transfer Learning & Deep U-Net Segmentation | Clinical Decision Support System")

# ----------------------------------------------------
# VIEW 3: CLINICAL ANALYTICS DASHBOARD
# ----------------------------------------------------
elif st.session_state.active_nav == "Analytics":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h2 style="font-weight: 700; color: #172538; margin-bottom: 0.2rem;">📈 Radiology Clinical Analytics</h2>
        <p style="color: #536579; font-size: 0.95rem;">
            Real-time aggregate telemetry on scan volumes, classification distributions, and model confidence metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    analytics_data = get_dashboard_analytics()
    total_scans = analytics_data.get("total_scans", 0)
    total_classifications = analytics_data.get("total_classifications", 0)
    total_segmentations = analytics_data.get("total_segmentations", 0)
    avg_conf = f"{analytics_data.get('avg_confidence', 0)}%"
    class_dist = analytics_data.get("class_distribution", {})

    st.markdown(f"""
    <div class="metric-grid-4">
        <div class="metric-tile">
            <div class="metric-tile-title">Total Scans Processed</div>
            <div class="metric-tile-val" style="color: #2767A8;">{total_scans}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-title">Classifications Logged</div>
            <div class="metric-tile-val val-indigo">{total_classifications}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-title">Segmentations Executed</div>
            <div class="metric-tile-val val-teal">{total_segmentations}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-title">Avg Model Confidence</div>
            <div class="metric-tile-val val-success">{avg_conf}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 📊 Tumor Category Distribution Breakdown")
    for category, count in class_dist.items():
        pct = round((count / max(1, total_scans)) * 100, 1)
        c_col1, c_col2 = st.columns([5, 1])
        with c_col1:
            st.progress(pct / 100.0, text=f"{category} ({count} scans)")
        with c_col2:
            st.write(f"**{pct}%**")

# ----------------------------------------------------
# VIEW 4: AUDIT LOG HISTORY TABLE
# ----------------------------------------------------
elif st.session_state.active_nav == "Audit Log":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h2 style="font-weight: 700; color: #172538; margin-bottom: 0.2rem;">🗄️ Radiology Inspection Audit Log</h2>
        <p style="color: #536579; font-size: 0.95rem;">
            Persistent audit record tracking processed MRI scans, model predictions, confidence scores, and segmentation parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    records = get_history_records(limit=100)
    if records:
        table_rows = []
        for r in records:
            conf_val = f"{round(float(r['confidence_score'])*100, 1)}%" if r['confidence_score'] is not None else "--"
            cov_val = f"{r['tumor_percentage']}%" if r['tumor_percentage'] is not None else "--"
            table_rows.append({
                "Record ID": f"#{r['id']}",
                "Clinician": r.get('doctor_name') or "Dr. Sapna",
                "Scan File": r.get('filename') or "MRI_Scan.png",
                "Model Pipeline": r.get('model_type'),
                "Diagnosis": r.get('predicted_class') or "Segmented",
                "Confidence": conf_val,
                "Tumor Coverage": cov_val,
                "Timestamp": str(r.get('timestamp'))[:19]
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No audit records found yet. Analyzed scans will appear here automatically.")
