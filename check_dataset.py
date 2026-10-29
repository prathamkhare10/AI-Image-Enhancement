
import os
import glob
from PIL import Image

folders = [
    "data/DIV2K_train_HR",
    "data/Set5"
]

for folder in folders:
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(
            glob.glob(
                os.path.join(folder, "**", ext),
                recursive=True
            )
        )

    print(f"\nFolder: {folder}")
    print(f"Images found: {len(images)}")

    if images:
        with Image.open(images[0]) as img:
            print(f"Sample image: {os.path.basename(images[0])}")
            print(f"Dimensions: {img.size}")
