import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path

from siren_arch import SIREN_IMG_MODEL
256

SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "demo_images"
MODELS_DIR = SCRIPT_DIR / "demo_models"

IMAGE_DIM = 256


def render_siren_img(model_fp: Path):
    pixel_grid = torch.meshgrid(
        torch.linspace(-1, 1, IMAGE_DIM),
        torch.linspace(-1, 1, IMAGE_DIM),
        indexing="ij",
    )
    print(pixel_grid)

    model = SIREN_IMG_MODEL()
    model.load_state_dict(torch.load(model_fp))
    model.eval()

    with torch.no_grad():
        pixel_grid = torch.stack(pixel_grid, dim=-1).reshape(-1, 2)
        pixel_grid = pixel_grid

        rgb = model(pixel_grid)
        rgb = rgb.cpu().numpy().reshape(IMAGE_DIM, IMAGE_DIM, 3)
        rgb = np.clip(rgb, 0, 1)

    return rgb


if __name__ == "__main__":
    img_name = "demo_image_00"
    img_fp = IMAGES_DIR / f"{img_name}.jpg"
    model_fp = MODELS_DIR / f"{img_name}.pt"

    rgb = render_siren_img(model_fp)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb)
    axs[1].imshow(np.array(Image.open(img_fp)))

    axs[0].set_title("INR Rendered Image")
    axs[1].set_title("Original Image")
    
    plt.savefig(f"{img_name}_render.png")
