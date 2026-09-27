import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'brain-tumor-ai-secret-key-2026-secure')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    SEGMENT_FOLDER = os.path.join(BASE_DIR, 'static', 'segmented')
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
    DATABASE_PATH = os.path.join(BASE_DIR, 'database.db')
    
    # Datasets Folders
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    CLASSIFICATION_DATASET_DIR = os.path.join(DATASETS_DIR, 'classification')
    SEGMENTATION_DATASET_DIR = os.path.join(DATASETS_DIR, 'segmentation')
    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm'}
    
    # Model parameters
    CLASSIFICATION_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    CLASSIFICATION_INPUT_SHAPE = (224, 224, 3)
    SEGMENTATION_INPUT_SHAPE = (256, 256, 1)

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.SEGMENT_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        
        # Datasets directories
        for cls_name in Config.CLASSIFICATION_CLASSES:
            os.makedirs(os.path.join(Config.CLASSIFICATION_DATASET_DIR, cls_name.replace(' ', '_')), exist_ok=True)
            os.makedirs(os.path.join(Config.CLASSIFICATION_DATASET_DIR, cls_name), exist_ok=True)

        os.makedirs(os.path.join(Config.SEGMENTATION_DATASET_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(Config.SEGMENTATION_DATASET_DIR, 'masks'), exist_ok=True)
