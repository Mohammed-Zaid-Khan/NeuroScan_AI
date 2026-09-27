import os
import zipfile
import shutil

zip_path = r'C:\Users\Strato\Downloads\archive.zip'
target_dir = r'c:\Users\Strato\Documents\brain-tumor\datasets\segmentation'

images_dir = os.path.join(target_dir, 'images')
masks_dir = os.path.join(target_dir, 'masks')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

print(f"Extracting '{zip_path}' into segmentation dataset directory...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    count_imgs = 0
    count_masks = 0
    
    for file_info in zip_ref.infolist():
        filename = file_info.filename
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            base_name = os.path.basename(filename)
            if '_mask' in base_name:
                # Target segmentation mask
                dst = os.path.join(masks_dir, base_name)
                with zip_ref.open(file_info) as source, open(dst, "wb") as target:
                    shutil.copyfileobj(source, target)
                count_masks += 1
            else:
                # Target raw scan image
                dst = os.path.join(images_dir, base_name)
                with zip_ref.open(file_info) as source, open(dst, "wb") as target:
                    shutil.copyfileobj(source, target)
                count_imgs += 1

print(f"[SUCCESS] Extracted {count_imgs} raw MRI scan images to '{images_dir}'")
print(f"[SUCCESS] Extracted {count_masks} ground-truth binary masks to '{masks_dir}'")
