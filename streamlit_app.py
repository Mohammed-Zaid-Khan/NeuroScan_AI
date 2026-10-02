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
    page_title="NeuroScan AI | Brain Tumor Classification & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to render HTML markdown without leading indentation issues
def render_html(html_str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# ----------------------------------------------------
# 1. EXACT CLINICAL CSS STYLING (LOCAL WORKSTATION MATCH)
# ----------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    .stApp {
        background-color: #F3F6F9 !important;
        max-width: 1440px;
        margin: 0 auto;
    }
    
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
    .clinical-navbar-wrapper {
        background-color: #12304A;
        border-radius: 12px;
        padding: 0.7rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 16px rgba(13, 38, 59, 0.12);
        color: #FFFFFF;
    }
    .nav-brand-title {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.65rem;
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
        gap: 0.55rem;
        background: rgba(255, 255, 255, 0.12);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.86rem;
        font-weight: 600;
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .avatar-icon-circle {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #2767A8;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    /* WORKSPACE HERO HEADER */
    .workspace-pill-tag {
        display: inline-block;
        background: #EEF4FA;
        color: #2767A8;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .workspace-meta-tag {
        color: #8290A0;
        font-size: 0.82rem;
        margin-left: 0.5rem;
        font-weight: 500;
    }
    .hero-title-main {
        font-size: 1.65rem;
        font-weight: 800;
        color: #172538;
        margin-top: 0.35rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    .hero-subtitle-main {
        color: #536579;
        font-size: 0.92rem;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }

    /* WORKSTATION CARD CONTAINERS */
    .workstation-box {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #D8E1EA;
        box-shadow: 0 1px 3px rgba(13, 38, 59, 0.04), 0 6px 18px rgba(13, 38, 59, 0.02);
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
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* METRIC TILES */
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

    /* ACTION BUTTONS */
    .stButton>button {
        background-color: #2767A8 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        height: 44px !important;
        width: 100% !important;
        font-size: 0.92rem !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(39, 103, 168, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #1D558B !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(39, 103, 168, 0.35) !important;
    }

    /* STAGE GRID */
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

    /* TOP BAR RADIO BUTTONS */
    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
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
    """Replicates the brain MRI canvas scan generation from localhost JavaScript."""
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
    <div style="background-color: #12304A; border-radius: 12px; padding: 0.65rem 1.2rem; display: flex; align-items: center; height: 52px; box-shadow: 0 4px 16px rgba(13, 38, 59, 0.12);">
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
    </div>
    """)

with nav_col2:
    st.session_state.active_nav = st.radio(
        "Navigation Tabs",
        ["🔬 Diagnostic Suite", "📈 Analytics", "🗄️ Audit Log"],
        horizontal=True,
        label_visibility="collapsed",
        index=["🔬 Diagnostic Suite", "📈 Analytics", "🗄️ Audit Log"].index(
            f"🔬 Diagnostic Suite" if "Diagnostic" in st.session_state.active_nav
            else (f"📈 Analytics" if "Analytics" in st.session_state.active_nav else "🗄️ Audit Log")
        )
    )

with nav_col3:
    render_html(f"""
    <div style="background-color: #12304A; border-radius: 12px; padding: 0.65rem 1.2rem; display: flex; align-items: center; justify-content: flex-end; gap: 0.85rem; height: 52px; box-shadow: 0 4px 16px rgba(13, 38, 59, 0.12);">
        <div class="status-badge-live">
            <span class="status-dot-pulse"></span>
            <span>AI Engines Online</span>
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
    
    # Header Subtitle & Pipeline Selector Row (Matching Localhost)
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
                <span>📁 MRI Input</span>
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

        # EXACT LOCALHOST FEATURE: 2x2 Example MRI Scans Buttons
        render_html("""
        <div style="font-size: 0.88rem; font-weight: 700; color: #172538; margin-top: 0.85rem; margin-bottom: 0.15rem;">
            Example MRI Scans
        </div>
        <div style="font-size: 0.78rem; color: #8290A0; margin-bottom: 0.65rem;">
            Load a sample to test the pipeline
        </div>
        """)

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("🧠 Glioma", key="btn_sample_glioma"):
                st.session_state.selected_scan_bytes = generate_sample_mri("glioma")
                st.session_state.selected_scan_name = "sample_glioma_mri.png"
                st.session_state.analysis_results = None
                st.rerun()
            if st.button("🧠 Pituitary", key="btn_sample_pituitary"):
                st.session_state.selected_scan_bytes = generate_sample_mri("pituitary")
                st.session_state.selected_scan_name = "sample_pituitary_mri.png"
                st.session_state.analysis_results = None
                st.rerun()
        with sc2:
            if st.button("🧠 Meningioma", key="btn_sample_meningioma"):
                st.session_state.selected_scan_bytes = generate_sample_mri("meningioma")
                st.session_state.selected_scan_name = "sample_meningioma_mri.png"
                st.session_state.analysis_results = None
                st.rerun()
            if st.button("🧠 No Tumor", key="btn_sample_normal"):
                st.session_state.selected_scan_bytes = generate_sample_mri("normal")
                st.session_state.selected_scan_name = "sample_normal_mri.png"
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
                st.caption(f"**Size:** {kb_size} KB | **Status:** 🟢 Loaded & Ready")
            with col_btn:
                if st.button("Remove", key="btn_clear_scan"):
                    st.session_state.selected_scan_bytes = None
                    st.session_state.selected_scan_name = None
                    st.session_state.analysis_results = None
                    st.rerun()

            # Dynamic Action Button
            btn_title = f"▶ Run {st.session_state.pipeline_mode} Analysis"
            if st.button(btn_title, key="btn_execute_analysis"):
                with st.spinner("Analyzing MRI slice and computing neural activation heatmaps..."):
                    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                    os.makedirs(Config.SEGMENT_FOLDER, exist_ok=True)
                    temp_file_path = os.path.join(Config.UPLOAD_FOLDER, "temp_current_scan.png")
                    with open(temp_file_path, "wb") as f:
                        f.write(st.session_state.selected_scan_bytes)

                    # 1. Classification & Grad-CAM Heatmap
                    pred_class, conf, probs = None, None, None
                    gradcam_overlay = None
                    if "Classification" in st.session_state.pipeline_mode or "Dual" in st.session_state.pipeline_mode:
                        c_tensor, raw_c = preprocess_for_classification(temp_file_path)
                        pred_class, conf, probs = classifier_model.predict(
                            c_tensor, raw_c, filename=st.session_state.selected_scan_name
                        )
                        gradcam_map = generate_gradcam_heatmap(classifier_model, c_tensor, raw_c)
                        _, gradcam_overlay = overlay_heatmap_on_image(
                            temp_file_path, gradcam_map, alpha=0.45, colormap_name="JET"
                        )

                    # 2. Segmentation & Continuous Probability Density Heatmap
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

                    # 3. Save Record to SQLite database.db
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
            <div style="background: #F1F5F9; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 0.9rem; text-align: center; color: #64748B; font-size: 0.85rem; margin-top: 0.5rem;">
                Select an MRI scan to begin.
            </div>
            """)

    # RIGHT PANEL: DIAGNOSTIC OUTPUT & VISUALIZATIONS
    with grid_right:
        render_html("""
        <div class="card-header-row">
            <div class="card-header-title">
                <span>📊 Diagnostic Output & Visualizations</span>
            </div>
            <div class="status-badge-live">
                <span class="status-dot-pulse"></span>
                <span>Live inference</span>
            </div>
        </div>
        """)

        results = st.session_state.analysis_results

        # INITIAL EMPTY PLACEHOLDER (EXACT MATCH TO LOCALHOST SCREENSHOT)
        if results is None:
            render_html("""
            <div class="empty-state-box">
                <div class="empty-icon-circle">🔬</div>
                <h3 style="font-weight: 700; color: #172538; margin-bottom: 0.35rem;">Ready for analysis</h3>
                <p style="color: #536579; font-size: 0.92rem;">Upload or select an MRI scan, choose an AI pipeline, and run analysis.</p>
                <div class="workflow-steps-wrapper">
                    <span><strong style="color: #2767A8;">01</strong> MRI Input</span>
                    <span>──▶</span>
                    <span><strong style="color: #2767A8;">02</strong> AI Pipeline</span>
                    <span>──▶</span>
                    <span><strong style="color: #2767A8;">03</strong> Diagnostic Output</span>
                </div>
            </div>
            """)

        else:
            # CLINICAL STAGE VISUALIZATION & HEATMAP TABS
            tab_stage_view, tab_gradcam_view, tab_prob_view, tab_multi_view = st.tabs([
                "🖼️ Clinical MRI Stages",
                "🔥 Grad-CAM Attention Heatmap",
                "🌊 Probability Density Heatmap",
                "🔍 4-Panel Multi-Modal Synthesis"
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
                st.markdown("#### 🔥 Grad-CAM Class Activation Mapping (Explainable AI)")
                st.caption("Visualizes the high-attention convolutional gradients driving the VGG16 classification decision.")
                if results["gradcam_overlay"] is not None:
                    gc1, gc2 = st.columns([1, 1])
                    with gc1:
                        st.image(results["temp_file_path"], caption="Original Anatomical Scan", use_container_width=True)
                    with gc2:
                        st.image(results["gradcam_overlay"], caption="Superimposed Grad-CAM Heatmap (JET)", use_container_width=True)
                else:
                    st.info("Grad-CAM is generated in Dual Pipeline or Classification mode.")

            with tab_prob_view:
                st.markdown("#### 🌊 Continuous Tumor Probability Density (U-Net)")
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
                st.markdown("#### 🔍 Complete Multi-Modal Diagnostic Synthesis")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.image(results["temp_file_path"], caption="1. Anatomical MRI", use_container_width=True)
                with m2:
                    if results["gradcam_overlay"] is not None:
                        st.image(results["gradcam_overlay"], caption="2. VGG16 Grad-CAM", use_container_width=True)
                with m3:
                    if results["prob_overlay"] is not None:
                        st.image(results["prob_overlay"], caption="3. U-Net Density", use_container_width=True)
                with m4:
                    if results["overlay_path"] is not None and os.path.exists(results["overlay_path"]):
                        st.image(results["overlay_path"], caption="4. Delineated Contour", use_container_width=True)

            # 4 METRIC TILES (EXACT LOCALHOST MATCH)
            p_cls = results["pred_class"] or "N/A"
            c_pct = f"{round(results['conf']*100, 1)}%" if results["conf"] is not None else "N/A"
            a_px = f"{results['tumor_px']:,} px" if results["tumor_px"] is not None else "N/A"
            t_pct = f"{results['tumor_pct']}%" if results["tumor_pct"] is not None else "N/A"

            render_html(f"""
            <div class="metric-grid-4">
                <div class="metric-tile">
                    <div class="metric-tile-title">Predicted Class</div>
                    <div class="metric-tile-val val-indigo">{p_cls}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Confidence</div>
                    <div class="metric-tile-val val-indigo">{c_pct}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Tumor Coverage</div>
                    <div class="metric-tile-val val-teal">{t_pct}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-tile-title">Pixel Area</div>
                    <div class="metric-tile-val val-teal">{a_px}</div>
                </div>
            </div>
            """)

            # VGG16 CLASS PROBABILITY BREAKDOWN
            if results["probs"]:
                st.markdown("#### 🏷️ VGG16 Class Probability Breakdown")
                for c_name, c_val in results["probs"].items():
                    p1, p2 = st.columns([5, 1])
                    with p1:
                        st.progress(float(c_val) / 100.0, text=f"{c_name}")
                    with p2:
                        st.write(f"**{c_val}%**")

            st.caption("🔬 **Architecture:** VGG16 Transfer Learning & Deep U-Net Segmentation | Clinical Decision Support System")

# ----------------------------------------------------
# 5. VIEW B: CLINICAL ANALYTICS
# ----------------------------------------------------
elif "Analytics" in st.session_state.active_nav:
    render_html("""
    <div style="margin-bottom: 1.25rem;">
        <h2 style="font-weight: 700; color: #172538; margin-bottom: 0.2rem;">📈 Radiology Clinical Analytics</h2>
        <p style="color: #536579; font-size: 0.95rem;">
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
            <div class="metric-tile-val" style="color: #2767A8;">{scans_num}</div>
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
    st.markdown("### 📊 Tumor Category Distribution Breakdown")
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
        <h2 style="font-weight: 700; color: #172538; margin-bottom: 0.2rem;">🗄️ Radiology Inspection Audit Log</h2>
        <p style="color: #536579; font-size: 0.95rem;">
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
