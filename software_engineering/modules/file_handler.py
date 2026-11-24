
from pathlib import Path


def save_image(image, original_path):
    """Save the enhanced image in the application's outputs folder."""

    # Get the software_engineering folder
    project_folder = Path(__file__).resolve().parents[1]

    # Create outputs folder if it doesn't exist
    output_folder = project_folder / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    original_name = Path(original_path).stem
    output_path = output_folder / f"{original_name}_enhanced.png"

    # Save enhanced image
    image.save(output_path, format="PNG")

    return output_path
