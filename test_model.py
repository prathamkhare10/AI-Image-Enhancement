
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio


# ---------------- CONFIGURATION ----------------

MODEL_PATH = "outputs/srcnn_div2k_flickr2k.pth"
TEST_FOLDER = "data/Set5"
OUTPUT_FOLDER = "outputs/set5_results"

SCALE_FACTOR = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Testing device:", device)


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


# ---------------- IMAGE CONVERSION ----------------

def image_to_tensor(image):

    array = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)

    return tensor.unsqueeze(0)


def tensor_to_image(tensor):

    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(array)


# ---------------- LOAD TRAINED MODEL ----------------

model = SRCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("New SRCNN model loaded successfully!")


# ---------------- TEST DATASET ----------------

image_files = sorted([

    f for f in os.listdir(TEST_FOLDER)

    if f.lower().endswith((".png", ".jpg", ".jpeg"))

])

bicubic_scores = []
model_scores = []


# ---------------- TEST ALL SET5 IMAGES ----------------

for image_name in image_files:

    print("\nTesting:", image_name)

    image_path = os.path.join(TEST_FOLDER, image_name)

    original = Image.open(image_path).convert("RGB")

    width, height = original.size

    # Create low-resolution image
    low_width = max(1, width // SCALE_FACTOR)
    low_height = max(1, height // SCALE_FACTOR)

    low_res = original.resize(
        (low_width, low_height),
        Image.Resampling.BICUBIC
    )

    # Bicubic upscaling
    bicubic = low_res.resize(
        (width, height),
        Image.Resampling.BICUBIC
    )

    # SRCNN enhancement
    input_tensor = image_to_tensor(bicubic).to(device)

    with torch.no_grad():

        prediction = model(input_tensor)

    prediction = prediction.clamp(0, 1)

    enhanced = tensor_to_image(prediction)

    # Convert images to NumPy
    original_np = np.asarray(original).astype(np.float32) / 255.0

    bicubic_np = np.asarray(bicubic).astype(np.float32) / 255.0

    enhanced_np = np.asarray(enhanced).astype(np.float32) / 255.0

    # Calculate PSNR
    bicubic_psnr = peak_signal_noise_ratio(
        original_np,
        bicubic_np,
        data_range=1.0
    )

    srcnn_psnr = peak_signal_noise_ratio(
        original_np,
        enhanced_np,
        data_range=1.0
    )

    bicubic_scores.append(bicubic_psnr)
    model_scores.append(srcnn_psnr)

    # Save enhanced image
    enhanced.save(
        os.path.join(OUTPUT_FOLDER, image_name)
    )

    # Create comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bicubic)
    axes[0].set_title(
        f"Bicubic\nPSNR: {bicubic_psnr:.2f} dB"
    )

    axes[1].imshow(enhanced)
    axes[1].set_title(
        f"SRCNN\nPSNR: {srcnn_psnr:.2f} dB"
    )

    axes[2].imshow(original)
    axes[2].set_title("Original HR Reference")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(image_name)
    plt.tight_layout()

    comparison_name = (
            os.path.splitext(image_name)[0] + "_comparison.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, comparison_name),
        dpi=150
    )

    plt.close()

    print(
        f"{image_name}: "
        f"Bicubic = {bicubic_psnr:.2f} dB | "
        f"SRCNN = {srcnn_psnr:.2f} dB"
    )


# ---------------- FINAL RESULTS ----------------

if image_files:

    average_bicubic = np.mean(bicubic_scores)
    average_srcnn = np.mean(model_scores)

    print("\n========== FINAL SET5 RESULTS ==========")

    print(f"Average Bicubic PSNR: {average_bicubic:.2f} dB")

    print(f"Average SRCNN PSNR:   {average_srcnn:.2f} dB")

    print(f"PSNR Improvement:     {average_srcnn - average_bicubic:+.2f} dB")

    print("\nTesting completed successfully!")

    print("Results saved in:", OUTPUT_FOLDER)

else:

    print("No test images found!")


