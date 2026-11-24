
from pathlib import Path

project = Path(__file__).parent
folders = {
    "DIV2K": project / "data" / "DIV2K_train_HR",
    "Flickr2K": project / "data" / "Flickr2K",
}

extensions = {".png", ".jpg", ".jpeg"}

for name, folder in folders.items():
    images = [
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    ]
    print(f"{name}: {len(images)} images")

    if not images:
        print(f"WARNING: No images found in {folder}")
