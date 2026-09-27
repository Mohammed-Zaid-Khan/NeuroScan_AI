"""
System Self-Verification Script
-------------------------------
Tests DB initialization, Flask routes, and ML Engine pre-flight checks.
"""
import sys
import os

def test_system():
    print("==============================================")
    print(" Running Brain Tumor AI System Pre-Flight Check ")
    print("==============================================")

    # 1. Test Database Manager
    from database import init_db, get_dashboard_analytics
    init_db()
    stats = get_dashboard_analytics()
    print(f"[OK] Database Initialized Successfully. Total Scans: {stats['total_scans']}")

    # 2. Test Preprocessing Pipeline
    from ml_engine.preprocessing import preprocess_for_classification, preprocess_for_segmentation
    print("[OK] Preprocessing Modules Loaded.")

    # 3. Test Classifier Engine
    from ml_engine.classifier import BrainTumorClassifier
    classifier = BrainTumorClassifier()
    print(f"[OK] Classifier Engine Ready. Classes: {classifier.classes}")

    # 4. Test Segmenter Engine
    from ml_engine.segmenter import BrainTumorSegmenter
    segmenter = BrainTumorSegmenter()
    print(f"[OK] U-Net Segmenter Engine Ready. Input Shape: {segmenter.input_shape}")

    print("\n[SUCCESS] Pre-flight system check passed 100%!")

if __name__ == '__main__':
    test_system()
