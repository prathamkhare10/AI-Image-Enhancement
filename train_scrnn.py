
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor


# Configuration
DATASET_PATH = "data/DIV2K_train_HR"
OUTPUT_PATH = "outputs"
IMAGE_SIZE = 128
SCALE_FACTOR = 4
BATCH_SIZE = 4
EPOCHS = 2

os.makedirs(OUTPUT_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Dataset: create low-resolution/high-resolution pairs
class SRDataset(Dataset):
    def __init__(self, folder):
        self.paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.paths.extend(
                glob.glob(os.path.join(folder, "**", ext),
                          recursive=True)
            )

        if not self.paths:
            raise FileNotFoundError(
                f"No images found. Check this path: {folder}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BICUBIC
        )

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


# SRCNN-style convolutional neural network
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


def main():
    dataset = SRDataset(DATASET_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print("Images found:", len(dataset))

    model = SRCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for low, high in loader:
            low, high = low.to(device), high.to(device)

            optimizer.zero_grad()
            prediction = model(low)
            loss = criterion(prediction, high)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {total_loss / len(loader):.6f}"
        )

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_PATH, "srcnn_model.pth")
    )
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
