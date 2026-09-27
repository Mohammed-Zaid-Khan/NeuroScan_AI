import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Doctor/Radiologist',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_size INTEGER,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            model_type TEXT NOT NULL,
            predicted_class TEXT,
            confidence_score REAL,
            mask_path TEXT,
            overlay_path TEXT,
            tumor_percentage REAL,
            tumor_area_px INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
        )
    ''')
    
    # System logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action_type TEXT NOT NULL,
            description TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Model Metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            accuracy REAL,
            f1_score REAL,
            dice_score REAL,
            weights_path TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user if none exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        default_hash = generate_password_hash('admin123')
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ('admin', 'admin@medai.org', default_hash, 'Administrator')
        )

    # Create default demo user sapna26@gmail.com
    cursor.execute("SELECT * FROM users WHERE email = 'sapna26@gmail.com'")
    if not cursor.fetchone():
        demo_hash = generate_password_hash('Sapna@123')
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ('Sapna', 'sapna26@gmail.com', demo_hash, 'Doctor/Radiologist')
        )

    # Insert initial model metadata records if empty
    cursor.execute("SELECT COUNT(*) as count FROM model_metadata")
    if cursor.fetchone()['count'] == 0:
        cursor.execute('''
            INSERT INTO model_metadata (model_name, model_type, version, accuracy, f1_score, dice_score, weights_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('VGG16 Transfer Classifier', 'Classification', 'v1.0-Pretrained', 0.965, 0.962, None, 'models/best_model2.h5'))
        
        cursor.execute('''
            INSERT INTO model_metadata (model_name, model_type, version, accuracy, f1_score, dice_score, weights_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('Deep U-Net Segmenter', 'Segmentation', 'v1.0-Pretrained', 0.981, None, 0.924, 'models/brain_tumor_model.h5'))

    conn.commit()
    conn.close()

def register_user(username, email, password, role='Doctor/Radiologist'):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, pw_hash, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        log_action(user_id, 'USER_REGISTER', f"User {username} registered successfully.")
        return {'success': True, 'user_id': user_id, 'username': username, 'role': role}
    except sqlite3.IntegrityError as e:
        return {'success': False, 'message': 'Username or Email already registered.'}
    finally:
        conn.close()

def authenticate_user(email_or_username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE email = ? OR username = ?",
        (email_or_username, email_or_username)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        log_action(user['id'], 'USER_LOGIN', f"User {user['username']} logged in.")
        return {
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        }
    return {'success': False, 'message': 'Invalid username/email or password.'}

def record_upload(user_id, filename, filepath, file_size):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (user_id, filename, filepath, file_size) VALUES (?, ?, ?, ?)",
        (user_id, filename, filepath, file_size)
    )
    conn.commit()
    upload_id = cursor.lastrowid
    conn.close()
    return upload_id

def record_prediction(upload_id, model_type, predicted_class=None, confidence_score=None,
                      mask_path=None, overlay_path=None, tumor_percentage=None, tumor_area_px=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions 
        (upload_id, model_type, predicted_class, confidence_score, mask_path, overlay_path, tumor_percentage, tumor_area_px)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (upload_id, model_type, predicted_class, confidence_score, mask_path, overlay_path, tumor_percentage, tumor_area_px))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id

def get_history_records(user_id=None, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        query = '''
            SELECT p.*, u.filename, u.filepath, u.upload_time, usr.username
            FROM predictions p
            JOIN uploads u ON p.upload_id = u.id
            LEFT JOIN users usr ON u.user_id = usr.id
            WHERE u.user_id = ?
            ORDER BY p.timestamp DESC LIMIT ?
        '''
        params = (user_id, limit)
    else:
        query = '''
            SELECT p.*, u.filename, u.filepath, u.upload_time, usr.username
            FROM predictions p
            JOIN uploads u ON p.upload_id = u.id
            LEFT JOIN users usr ON u.user_id = usr.id
            ORDER BY p.timestamp DESC LIMIT ?
        '''
        params = (limit,)
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_dashboard_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM uploads")
    total_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE model_type LIKE '%Classification%' OR model_type LIKE '%Dual%'")
    total_classifications = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE model_type LIKE '%Segmentation%' OR model_type LIKE '%Dual%'")
    total_segmentations = cursor.fetchone()[0]
    
    cursor.execute("SELECT predicted_class, COUNT(*) as cnt FROM predictions WHERE predicted_class IS NOT NULL AND predicted_class != '' GROUP BY predicted_class")
    class_distribution = {row['predicted_class']: row['cnt'] for row in cursor.fetchall()}
    
    cursor.execute("SELECT AVG(confidence_score) FROM predictions WHERE confidence_score IS NOT NULL AND confidence_score > 0")
    avg_confidence = cursor.fetchone()[0] or 0.0
    
    conn.close()
    return {
        'total_scans': total_scans,
        'total_classifications': total_classifications,
        'total_segmentations': total_segmentations,
        'class_distribution': class_distribution,
        'avg_confidence': round(avg_confidence * 100, 2)
    }

def log_action(user_id, action_type, description):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO system_logs (user_id, action_type, description) VALUES (?, ?, ?)",
            (user_id, action_type, description)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Logging error: {e}")
