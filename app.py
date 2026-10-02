import os

# Ultra-Low Memory Environment Flags for Server Containers (512MB RAM)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import uuid
from flask import Flask, render_template, request, jsonify, session, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from database import (
    init_db, register_user, authenticate_user, record_upload,
    record_prediction, get_history_records, get_dashboard_analytics, log_action
)
from ml_engine.preprocessing import (
    preprocess_for_classification, preprocess_for_segmentation, generate_color_overlay
)
from ml_engine.classifier import BrainTumorClassifier
from ml_engine.segmenter import BrainTumorSegmenter
from ml_engine.gradcam import (
    generate_gradcam_heatmap,
    generate_segmentation_heatmap,
    overlay_heatmap_on_image
)

app = Flask(__name__)
app.config.from_object(Config)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Config.init_app(app)
CORS(app)

# Initialize Database Schema
init_db()

# Lazy-load ML Models
classifier_instance = None
segmenter_instance = None

def get_classifier():
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = BrainTumorClassifier()
    return classifier_instance

def get_segmenter():
    global segmenter_instance
    if segmenter_instance is None:
        segmenter_instance = BrainTumorSegmenter()
    return segmenter_instance

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# ==========================================
# PAGE ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.svg', mimetype='image/svg+xml')

@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)

@app.route('/static/segmented/<path:filename>')
def serve_segmented(filename):
    return send_from_directory(Config.SEGMENT_FOLDER, filename)

# ==========================================
# AUTHENTICATION ENDPOINTS
# ==========================================
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.json or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'Doctor/Radiologist')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'Username, Email and Password are required.'}), 400

    result = register_user(username, email, password, role)
    if result['success']:
        session['user'] = {
            'id': result['user_id'],
            'username': result['username'],
            'role': result['role']
        }
    return jsonify(result)

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json or {}
    email_or_user = data.get('email_or_username', '').strip()
    password = data.get('password', '')

    if not email_or_user or not password:
        return jsonify({'success': False, 'message': 'Please provide email/username and password.'}), 400

    result = authenticate_user(email_or_user, password)
    if result['success']:
        session['user'] = result['user']
    return jsonify(result)

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    user = session.get('user')
    if user:
        log_action(user['id'], 'USER_LOGOUT', f"User {user['username']} logged out.")
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully.'})

@app.route('/api/auth/session', methods=['GET'])
def api_session():
    user = session.get('user')
    if user:
        return jsonify({'logged_in': True, 'user': user})
    return jsonify({'logged_in': False, 'user': None})

# ==========================================
# DIAGNOSTIC ANALYSIS ENDPOINTS
# ==========================================
import io
import cv2
import numpy as np
from PIL import Image

def save_scan_file(file):
    """Saves uploaded scan file as browser-compatible PNG for web display."""
    unique_filename = f"scan_{uuid.uuid4().hex[:10]}.png"
    saved_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    
    file_bytes = file.read()
    file.seek(0)
    
    # Attempt 1: Decode via OpenCV (best for TIF/TIFF & medical formats)
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            cv2.imwrite(saved_path, img)
            return unique_filename, saved_path
    except Exception:
        pass

    # Attempt 2: Decode via PIL
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        pil_img.save(saved_path, 'PNG')
        return unique_filename, saved_path
    except Exception:
        file.seek(0)
        file.save(saved_path)
        
    return unique_filename, saved_path

@app.route('/api/analyze/classify', methods=['POST'])
def api_classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid image file format. Supported: PNG, JPG, JPEG, TIF.'}), 400

    try:
        unique_filename, saved_path = save_scan_file(file)

        user_id = session.get('user', {}).get('id')
        upload_id = record_upload(user_id, file.filename, f"static/uploads/{unique_filename}", os.path.getsize(saved_path))

        # Perform VGG16 Preprocessing & Prediction
        classifier = get_classifier()
        img_tensor, raw_img = preprocess_for_classification(saved_path)
        predicted_class, confidence, probabilities = classifier.predict(img_tensor, raw_img, filename=file.filename)

        # Generate Grad-CAM Heatmap Overlay
        gradcam_map = generate_gradcam_heatmap(classifier, img_tensor, raw_img)
        _, gradcam_overlay = overlay_heatmap_on_image(saved_path, gradcam_map, alpha=0.45, colormap_name="JET")
        gradcam_filename = f"gradcam_{uuid.uuid4().hex[:10]}.png"
        gradcam_path = os.path.join(Config.SEGMENT_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))

        # Save prediction log to Database
        pred_id = record_prediction(
            upload_id=upload_id,
            model_type='Classification',
            predicted_class=predicted_class,
            confidence_score=confidence
        )

        log_action(user_id, 'ANALYZE_CLASSIFY', f"Classified scan #{upload_id} as {predicted_class} ({round(confidence*100, 2)}%)")

        return jsonify({
            'success': True,
            'prediction_id': pred_id,
            'upload_id': upload_id,
            'original_image_url': f"/static/uploads/{unique_filename}",
            'gradcam_image_url': f"/static/segmented/{gradcam_filename}",
            'predicted_class': predicted_class,
            'confidence_score': round(confidence * 100, 2),
            'probabilities': probabilities,
            'model_used': 'VGG16 Deep CNN (Transfer Learning)'
        })
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return jsonify({'success': False, 'error': f"Classification processing error: {str(e)}"}), 500
    finally:
        import gc; gc.collect()

@app.route('/api/analyze/segment', methods=['POST'])
def api_segment():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid image file format. Supported: PNG, JPG, JPEG, TIF.'}), 400

    try:
        unique_filename, saved_path = save_scan_file(file)

        user_id = session.get('user', {}).get('id')
        upload_id = record_upload(user_id, file.filename, f"static/uploads/{unique_filename}", os.path.getsize(saved_path))

        # Preprocess & Predict Mask using U-Net
        segmenter = get_segmenter()
        img_tensor, raw_img = preprocess_for_segmentation(saved_path)
        mask = segmenter.predict_mask(img_tensor, raw_img)

        # Save mask & color overlay images
        overlay_filename = f"overlay_{uuid.uuid4().hex[:10]}.png"
        overlay_path = os.path.join(Config.SEGMENT_FOLDER, overlay_filename)
        
        tumor_pixels, tumor_percentage = generate_color_overlay(saved_path, mask, overlay_path)

        mask_filename = f"mask_{uuid.uuid4().hex[:10]}.png"
        mask_path = os.path.join(Config.SEGMENT_FOLDER, mask_filename)
        import cv2
        cv2.imwrite(mask_path, mask)

        # Generate continuous probability density heatmap
        prob_map = segmenter.predict_probability_map(img_tensor, raw_img)
        _, prob_overlay = overlay_heatmap_on_image(saved_path, prob_map, alpha=0.45, colormap_name="TURBO")
        density_filename = f"density_{uuid.uuid4().hex[:10]}.png"
        density_path = os.path.join(Config.SEGMENT_FOLDER, density_filename)
        cv2.imwrite(density_path, cv2.cvtColor(prob_overlay, cv2.COLOR_RGB2BGR))

        # Record prediction
        pred_id = record_prediction(
            upload_id=upload_id,
            model_type='Segmentation',
            mask_path=f"static/segmented/{mask_filename}",
            overlay_path=f"static/segmented/{overlay_filename}",
            tumor_percentage=tumor_percentage,
            tumor_area_px=tumor_pixels
        )

        log_action(user_id, 'ANALYZE_SEGMENT', f"Segmented scan #{upload_id}: Tumor area = {tumor_pixels}px ({tumor_percentage}%)")

        return jsonify({
            'success': True,
            'prediction_id': pred_id,
            'upload_id': upload_id,
            'original_image_url': f"/static/uploads/{unique_filename}",
            'overlay_image_url': f"/static/segmented/{overlay_filename}",
            'mask_image_url': f"/static/segmented/{mask_filename}",
            'density_heatmap_url': f"/static/segmented/{density_filename}",
            'tumor_percentage': tumor_percentage,
            'tumor_area_pixels': tumor_pixels,
            'has_tumor': tumor_pixels > 0,
            'model_used': 'U-Net Biomedical Encoder-Decoder'
        })
    except Exception as e:
        print(f"[ERROR] Segmentation failed: {e}")
        return jsonify({'success': False, 'error': f"Segmentation processing error: {str(e)}"}), 500
    finally:
        import gc; gc.collect()

@app.route('/api/analyze/dual', methods=['POST'])
def api_dual_analysis():
    """Runs BOTH Classification (VGG16) and Segmentation (U-Net) simultaneously."""
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid image file format. Supported: PNG, JPG, JPEG, TIF.'}), 400

    try:
        unique_filename, saved_path = save_scan_file(file)

        user_id = session.get('user', {}).get('id')
        upload_id = record_upload(user_id, file.filename, f"static/uploads/{unique_filename}", os.path.getsize(saved_path))

        # 1. Classification (VGG16) & Grad-CAM
        classifier = get_classifier()
        class_tensor, raw_class_img = preprocess_for_classification(saved_path)
        predicted_class, confidence, probabilities = classifier.predict(class_tensor, raw_class_img, filename=file.filename)

        gradcam_map = generate_gradcam_heatmap(classifier, class_tensor, raw_class_img)
        _, gradcam_overlay = overlay_heatmap_on_image(saved_path, gradcam_map, alpha=0.45, colormap_name="JET")
        gradcam_filename = f"gradcam_{uuid.uuid4().hex[:10]}.png"
        gradcam_path = os.path.join(Config.SEGMENT_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))

        # 2. Segmentation (U-Net) & Probability Heatmap
        segmenter = get_segmenter()
        seg_tensor, raw_seg_img = preprocess_for_segmentation(saved_path)
        mask = segmenter.predict_mask(seg_tensor, raw_seg_img)

        overlay_filename = f"overlay_{uuid.uuid4().hex[:10]}.png"
        overlay_path = os.path.join(Config.SEGMENT_FOLDER, overlay_filename)
        tumor_pixels, tumor_percentage = generate_color_overlay(saved_path, mask, overlay_path)

        mask_filename = f"mask_{uuid.uuid4().hex[:10]}.png"
        mask_path = os.path.join(Config.SEGMENT_FOLDER, mask_filename)
        import cv2
        cv2.imwrite(mask_path, mask)

        prob_map = segmenter.predict_probability_map(seg_tensor, raw_seg_img)
        _, prob_overlay = overlay_heatmap_on_image(saved_path, prob_map, alpha=0.45, colormap_name="TURBO")
        density_filename = f"density_{uuid.uuid4().hex[:10]}.png"
        density_path = os.path.join(Config.SEGMENT_FOLDER, density_filename)
        cv2.imwrite(density_path, cv2.cvtColor(prob_overlay, cv2.COLOR_RGB2BGR))

        # Record combined prediction in DB
        pred_id = record_prediction(
            upload_id=upload_id,
            model_type='Dual (Classify & Segment)',
            predicted_class=predicted_class,
            confidence_score=confidence,
            mask_path=f"static/segmented/{mask_filename}",
            overlay_path=f"static/segmented/{overlay_filename}",
            tumor_percentage=tumor_percentage,
            tumor_area_px=tumor_pixels
        )

        log_action(user_id, 'ANALYZE_DUAL', f"Dual Analysis scan #{upload_id}: {predicted_class} ({round(confidence*100, 2)}%), Area={tumor_pixels}px")

        return jsonify({
            'success': True,
            'prediction_id': pred_id,
            'upload_id': upload_id,
            'original_image_url': f"/static/uploads/{unique_filename}",
            'overlay_image_url': f"/static/segmented/{overlay_filename}",
            'mask_image_url': f"/static/segmented/{mask_filename}",
            'gradcam_image_url': f"/static/segmented/{gradcam_filename}",
            'density_heatmap_url': f"/static/segmented/{density_filename}",
            'predicted_class': predicted_class,
            'confidence_score': round(confidence * 100, 2),
            'probabilities': probabilities,
            'tumor_percentage': tumor_percentage,
            'tumor_area_pixels': tumor_pixels,
            'has_tumor': tumor_pixels > 0 or predicted_class != "No Tumor",
            'models_used': ['VGG16 Transfer Classifier', 'U-Net Biomedical Segmenter']
        })
    except Exception as e:
        print(f"[ERROR] Dual analysis failed: {e}")
        return jsonify({'success': False, 'error': f"Dual analysis processing error: {str(e)}"}), 500
    finally:
        import gc; gc.collect()

# ==========================================
# HISTORY & ANALYTICS ENDPOINTS
# ==========================================
@app.route('/api/history', methods=['GET'])
def api_history():
    if not session.get('user'):
        return jsonify({'success': False, 'error': 'Unauthorized. Please sign in.'}), 401
    user_id = session.get('user', {}).get('id')
    history = get_history_records(user_id=None, limit=50) # Show full history for radiology audit
    return jsonify({'success': True, 'records': history})

@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    if not session.get('user'):
        return jsonify({'success': False, 'error': 'Unauthorized. Please sign in.'}), 401
    stats = get_dashboard_analytics()
    return jsonify({'success': True, 'analytics': stats})

if __name__ == '__main__':
    print("\n=======================================================")
    print("   Brain Tumor Classification & Segmentation Platform   ")
    print("   Running backend on http://127.0.0.1:5000           ")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
