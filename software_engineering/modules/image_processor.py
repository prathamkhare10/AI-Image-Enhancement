
from PIL import Image


SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")


def validate_image(image_path):
    """Check whether the selected file is a valid supported image."""

    if not image_path.lower().endswith(SUPPORTED_FORMATS):
        raise ValueError("Only JPG, JPEG and PNG images are supported.")

    try:
        image = Image.open(image_path)
        image.verify()
    except Exception:
        raise ValueError("The selected file is not a valid image.")

    return True


def load_image(image_path):
    """Load an image and convert it to RGB format."""

    validate_image(image_path)

    image = Image.open(image_path).convert("RGB")
    return image
