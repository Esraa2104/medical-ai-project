import os
import shutil
import random

random.seed(42)  # pour reproductibilité

SOURCE = r"C:\Users\User\Downloads\archive\chest_xray\chest_xray"
DEST = r"C:\Users\User\medical-ai-project\data"

CLASSES = ["NORMAL", "PNEUMONIA"]
VAL_RATIO = 0.15  # 15% du train ira en validation

def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg", ".png"))]

def copy_files(files, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(src_folder, f), os.path.join(dst_folder, f))

def main():
    for cls in CLASSES:
        # --- TRAIN + VALIDATION (split à partir du train Kaggle) ---
        train_src = os.path.join(SOURCE, "train", cls)
        images = list_images(train_src)
        random.shuffle(images)

        n_val = int(len(images) * VAL_RATIO)
        val_images = images[:n_val]
        train_images = images[n_val:]

        copy_files(train_images, train_src, os.path.join(DEST, "train", cls))
        copy_files(val_images, train_src, os.path.join(DEST, "validation", cls))

        print(f"[{cls}] train: {len(train_images)} | validation: {len(val_images)}")

        # --- TEST (copié tel quel) ---
        test_src = os.path.join(SOURCE, "test", cls)
        test_images = list_images(test_src)
        copy_files(test_images, test_src, os.path.join(DEST, "test", cls))

        print(f"[{cls}] test: {len(test_images)}")

if __name__ == "__main__":
    main()