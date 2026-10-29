
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Paths and settings
MODEL_PATH = "outputs/srcnn_model.pth"
IMAGE_PATH = "data/Set5/baby.png"
OUTPUT_PATH = "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Must match the architecture used during training
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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    model = SRCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # Load original image
    original = Image.open(IMAGE_PATH).convert("RGB")

    # Create a 4x lower-resolution input
    low_size = (original.width // 4, original.height // 4)
    low = original.resize(low_size, Image.Resampling.BICUBIC)

    # Ordinary resizing baseline
    bicubic = low.resize(original.size, Image.Resampling.BICUBIC)

    # Model expects the enlarged low-resolution image
    input_tensor = to_tensor(bicubic).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).clamp(0, 1)

    result = prediction.squeeze(0).cpu().permute(1, 2, 0).numpy()

    # Save AI-enhanced output
    output_file = os.path.join(OUTPUT_PATH, "baby_enhanced.png")
    plt.imsave(output_file, result)

    # Display comparisons
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(bicubic)
    plt.title("Bicubic Baseline")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(result)
    plt.title("AI Model Output")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(original)
    plt.title("Original Reference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Comparison saved:", output_file)


if __name__ == "__main__":
    main()
