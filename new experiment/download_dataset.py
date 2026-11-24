
import csv
import io
import time
import requests
from PIL import Image
from pathlib import Path

CSV_PATH = Path.home() / "Downloads" / "image_ids_and_rotation.csv"
OUTPUT_DIR = Path(__file__).parent / "data" / "train"

TARGET_IMAGES = 100
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

downloaded = 0
failed = 0

print("Starting download of 100 training images...")

with open(CSV_PATH, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["Subset"] != "train":
            continue

        url = row["OriginalURL"]

        try:
            response = requests.get(
                url,
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image.load()
            image = image.convert("RGB")

            filename = OUTPUT_DIR / f"{row['ImageID']}.jpg"
            image.save(filename, "JPEG", quality=95)

            downloaded += 1
            print(f"Downloaded {downloaded}/{TARGET_IMAGES}")

            if downloaded >= TARGET_IMAGES:
                break

        except Exception as error:
            failed += 1
            print(f"Skipped image: {error}")

        time.sleep(0.1)

print("\nDownload finished!")
print("Successful:", downloaded)
print("Failed:", failed)
print("Saved in:", OUTPUT_DIR)
