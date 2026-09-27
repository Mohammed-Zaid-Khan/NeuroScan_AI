"""
Brain Tumor Model Trainer Script
--------------------------------
Use this script to train VGG16 (Classification) or U-Net (Segmentation) models
when datasets are provided.

Usage Examples:
  python ml_engine/trainer.py --type classification --data_dir ./dataset/classification --epochs 25
  python ml_engine/trainer.py --type segmentation --images_dir ./dataset/images --masks_dir ./dataset/masks --epochs 30
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf

try:
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import Config
from ml_engine.classifier import BrainTumorClassifier
from ml_engine.segmenter import BrainTumorSegmenter, dice_loss, dice_coefficient

def train_classifier(data_dir, epochs=25, batch_size=32):
    print(f"\n==========================================")
    print(f" Starting VGG16 Classifier Model Training ")
    print(f" Data Directory: {data_dir}")
    print(f"==========================================\n")

    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset directory '{data_dir}' was not found on your system.")
        print("\nPlease replace './path/to/dataset' with the actual path to your MRI dataset folder.")
        print("Required Folder Structure for Classification:")
        print("  dataset_folder/")
        print("    ├── Glioma/")
        print("    ├── Meningioma/")
        print("    ├── No Tumor/")
        print("    └── Pituitary/\n")
        return

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=Config.CLASSIFICATION_INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    if train_generator.samples == 0:
        print(f"\n[NOTICE] Found 0 images inside '{data_dir}'.")
        print("Please place your MRI scan image files (.png / .jpg) into the category subfolders:")
        print(f"  {data_dir}\\Glioma\\")
        print(f"  {data_dir}\\Meningioma\\")
        print(f"  {data_dir}\\No_Tumor\\")
        print(f"  {data_dir}\\Pituitary\\\n")
        return

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=Config.CLASSIFICATION_INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    classifier = BrainTumorClassifier()
    save_path = os.path.join(Config.MODEL_FOLDER, 'best_model2.h5')

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = classifier.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"\n[SUCCESS] Classification training complete. Model saved to '{save_path}'.")
    return history

class SegmentationDataSequence(tf.keras.utils.Sequence):
    """Thread-safe Keras Sequence data generator for U-Net segmentation training."""
    def __init__(self, image_paths, mask_paths, batch_size=16, target_size=(256, 256), shuffle=True):
        self.image_paths = np.array(image_paths)
        self.mask_paths = np.array(mask_paths)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs_paths = self.image_paths[batch_indices]
        batch_masks_paths = self.mask_paths[batch_indices]

        batch_x = []
        batch_y = []

        for img_path, mask_path in zip(batch_imgs_paths, batch_masks_paths):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img_arr = np.zeros((*self.target_size, 1), dtype=np.float32)
            else:
                img_arr = cv2.resize(img, self.target_size).astype(np.float32) / 255.0
                img_arr = np.expand_dims(img_arr, axis=-1)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask_arr = np.zeros((*self.target_size, 1), dtype=np.float32)
            else:
                mask_arr = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                mask_arr = (mask_arr > 0).astype(np.float32)
                mask_arr = np.expand_dims(mask_arr, axis=-1)

            batch_x.append(img_arr)
            batch_y.append(mask_arr)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

def create_tf_segmentation_dataset(image_paths, mask_paths, batch_size=16, target_size=(256, 256), is_training=True):
    return SegmentationDataSequence(image_paths, mask_paths, batch_size=batch_size, target_size=target_size, shuffle=is_training)

def train_segmenter(images_dir, masks_dir, epochs=30, batch_size=16):
    print(f"\n==========================================")
    print(f" Starting U-Net Segmenter Model Training  ")
    print(f" Images: {images_dir} | Masks: {masks_dir}")
    print(f"==========================================\n")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"[ERROR] Images directory '{images_dir}' or Masks directory '{masks_dir}' missing.")
        return

    # Match raw image scans with corresponding mask files
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
    mask_files = set(os.listdir(masks_dir))

    paired_imgs = []
    paired_masks = []

    for img_f in img_files:
        name, ext = os.path.splitext(img_f)
        possible_mask_names = [img_f, f"{name}_mask{ext}", f"{name}_mask.tif", f"{name}_mask.png"]
        found_mask = None
        for m_name in possible_mask_names:
            if m_name in mask_files:
                found_mask = m_name
                break
        
        if found_mask:
            paired_imgs.append(os.path.join(images_dir, img_f))
            paired_masks.append(os.path.join(masks_dir, found_mask))

    if len(paired_imgs) == 0:
        print(f"[NOTICE] Found 0 paired MRI images and masks in '{images_dir}' and '{masks_dir}'.")
        return

    print(f"[INFO] Successfully paired {len(paired_imgs)} MRI scans with ground-truth masks.")

    # 80/20 Train-Validation Split
    split_idx = int(len(paired_imgs) * 0.8)
    train_x, val_x = paired_imgs[:split_idx], paired_imgs[split_idx:]
    train_y, val_y = paired_masks[:split_idx], paired_masks[split_idx:]

    train_ds = create_tf_segmentation_dataset(train_x, train_y, batch_size=batch_size, is_training=True)
    val_ds = create_tf_segmentation_dataset(val_x, val_y, batch_size=batch_size, is_training=False)

    segmenter = BrainTumorSegmenter()
    save_path = os.path.join(Config.MODEL_FOLDER, 'brain_tumor_model.h5')

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_dice_coefficient', mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print(f"[INFO] Training U-Net model on {len(train_x)} training samples, {len(val_x)} validation samples over {epochs} epochs...\n")

    history = segmenter.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"\n[SUCCESS] U-Net Segmentation training complete. Model weights saved to '{save_path}'.")
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Brain Tumor AI Models')
    parser.add_argument('--type', choices=['classification', 'segmentation'], required=True)
    parser.add_argument('--data_dir', type=str, default=Config.CLASSIFICATION_DATASET_DIR, help='Path to classification dataset directory')
    parser.add_argument('--images_dir', type=str, default=os.path.join(Config.SEGMENTATION_DATASET_DIR, 'images'), help='Path to segmentation images directory')
    parser.add_argument('--masks_dir', type=str, default=os.path.join(Config.SEGMENTATION_DATASET_DIR, 'masks'), help='Path to segmentation masks directory')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    if args.type == 'classification':
        train_classifier(args.data_dir, args.epochs, args.batch_size)
    elif args.type == 'segmentation':
        train_segmenter(args.images_dir, args.masks_dir, args.epochs, args.batch_size)
