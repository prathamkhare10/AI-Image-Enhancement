
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path



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



class ModelService:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Locate the trained model in the main project outputs folder
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "outputs" / "srcnn_div2k_flickr2k.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        self.model = SRCNN().to(self.device)

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()


    def enhance_image(self, image):
        """Enhance an image in small patches to reduce GPU memory usage."""

        width, height = image.size

        # Upscale using bicubic interpolation
        upscaled_image = image.resize(
            (width * 4, height * 4),
            Image.Resampling.BICUBIC
        )

        # Convert image to NumPy array
        image_array = np.asarray(
            upscaled_image,
            dtype=np.float32
        ) / 255.0

        # Image dimensions
        img_height, img_width, _ = image_array.shape

        # Prepare output and overlap counters
        output = np.zeros(
            (img_height, img_width, 3),
            dtype=np.float32
        )

        count = np.zeros(
            (img_height, img_width, 1),
            dtype=np.float32
        )

        patch_size = 128
        overlap = 16
        stride = patch_size - 2 * overlap

        # Process image patch by patch
        with torch.no_grad():

            for y in range(0, img_height, stride):
                for x in range(0, img_width, stride):

                    y_end = min(y + patch_size, img_height)
                    x_end = min(x + patch_size, img_width)

                    patch = image_array[y:y_end, x:x_end]

                    patch = np.transpose(patch, (2, 0, 1))
                    patch_tensor = torch.from_numpy(
                        patch.copy()
                    ).unsqueeze(0).to(self.device)

                    enhanced = self.model(patch_tensor)

                    enhanced = enhanced.squeeze(0).cpu().numpy()
                    enhanced = np.transpose(enhanced, (1, 2, 0))

                    output[y:y_end, x:x_end] += enhanced
                    count[y:y_end, x:x_end] += 1

                    del patch_tensor, enhanced

        # Combine overlapping patches
        output = output / np.maximum(count, 1)

        # Convert back to image
        output = np.clip(output, 0, 1) * 255
        output = output.astype(np.uint8)

        return Image.fromarray(output)
