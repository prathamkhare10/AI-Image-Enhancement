
import os
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------- CONFIGURATION ----------------

DATASET_PATHS = [
    "data/DIV2K_train_HR",
    "data/Flickr2K"
]

OUTPUT_PATH = "outputs"
IMAGE_SIZE = 128
SCALE_FACTOR = 4
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_NAME = "srcnn_div2k_flickr2k.pth"

os.makedirs(OUTPUT_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ---------------- IMAGE TO TENSOR ----------------

def to_tensor(image):
    image = np.array(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image


# ---------------- DATASET ----------------

class SRDataset(Dataset):

    def __init__(self, folders):

        self.paths = []

        for folder in folders:

            for ext in ("*.png", "*.jpg", "*.jpeg"):

                self.paths.extend(
                    glob.glob(
                        os.path.join(folder, "**", ext),
                        recursive=True
                    )
                )

        self.paths = sorted(set(self.paths))

        if not self.paths:
            raise FileNotFoundError(
                "No images found. Check dataset folder paths."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image = Image.open(self.paths[index]).convert("RGB")

        width, height = image.size

        if width < IMAGE_SIZE or height < IMAGE_SIZE:

            image = image.resize(
                (max(width, IMAGE_SIZE), max(height, IMAGE_SIZE)),
                Image.Resampling.BICUBIC
            )

            width, height = image.size

        # Random HR crop
        left = random.randint(0, width - IMAGE_SIZE)
        top = random.randint(0, height - IMAGE_SIZE)

        image = image.crop((
            left,
            top,
            left + IMAGE_SIZE,
            top + IMAGE_SIZE
        ))

        # Generate LR image
        low_size = IMAGE_SIZE // SCALE_FACTOR

        low = image.resize(
            (low_size, low_size),
            Image.Resampling.BICUBIC
        )

        low = low.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BICUBIC
        )

        return to_tensor(low), to_tensor(image)


# ---------------- SRCNN MODEL ----------------

class SRCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=5, padding=2)

        )

    def forward(self, x):
        return self.net(x)


# ---------------- TRAINING ----------------

def main():

    dataset = SRDataset(DATASET_PATHS)

    print("Total images found:", len(dataset))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training images:", train_size)
    print("Validation images:", val_size)

    model = SRCNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        # Training
        model.train()
        total_train_loss = 0.0

        for low, high in train_loader:

            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()

            prediction = model(low)
            loss = criterion(prediction, high)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():

            for low, high in val_loader:

                low = low.to(device)
                high = high.to(device)

                prediction = model(low)
                loss = criterion(prediction, high)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training Loss: {avg_train_loss:.6f} | "
            f"Validation Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_PATH, MODEL_NAME)
            )

            print("Best model saved!")

    print("\nTraining completed!")
    print("Best validation loss:", best_val_loss)
    print("Saved model:", MODEL_NAME)


if __name__ == "__main__":
    main()
