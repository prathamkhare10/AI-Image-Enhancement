
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def evaluate_images(enhanced_image, reference_image):
    """
    Calculate PSNR and SSIM against a genuine
    high-resolution reference image.
    """

    enhanced = np.asarray(
        enhanced_image.convert("RGB"),
        dtype=np.uint8
    )

    reference = np.asarray(
        reference_image.convert("RGB"),
        dtype=np.uint8
    )

    if enhanced.shape != reference.shape:
        raise ValueError(
            "Enhanced image and reference image must have "
            "exactly the same dimensions."
        )

    psnr = peak_signal_noise_ratio(
        reference,
        enhanced,
        data_range=255
    )

    ssim = structural_similarity(
        reference,
        enhanced,
        channel_axis=2,
        data_range=255
    )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim)
    }
