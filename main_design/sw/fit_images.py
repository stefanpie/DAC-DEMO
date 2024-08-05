from siren_arch import SIREN_IMG_MODEL

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import tqdm


from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "demo_images"
MODELS_DIR = SCRIPT_DIR / "demo_models"

IMAGE_DIM = 1024

PIXEL_RANGE = torch.linspace(-1, 1, IMAGE_DIM)
PIXEL_GRID = torch.meshgrid(PIXEL_RANGE, PIXEL_RANGE, indexing="ij")


class ImageINRDataset(Dataset):
    def __init__(self, img_fp: Path):
        self.img_fp = img_fp

        tensor = torch.from_numpy(np.array(Image.open(img_fp))) / 255
        tensor = tensor.permute(2, 0, 1)
        self.tensor = tensor

        self.img_dim = tensor.shape[1]
        self.pixel_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.img_dim),
            torch.linspace(-1, 1, self.img_dim),
            indexing="ij",
        )

    def __len__(self):
        return self.img_dim**2

    def __getitem__(self, idx):
        # convert liner index to 2D index
        i,j = idx // self.img_dim, idx % self.img_dim
        i_scaled, j_scaled = self.pixel_grid[0][i, j], self.pixel_grid[1][i, j]
        rgb = self.tensor[:, i, j]
        return {
            "x_in": torch.tensor([i_scaled, j_scaled]).float(),
            "x_out": rgb,
        }


def fit_image(img_fp: Path):
    BATCH_SIZE = 64
    EPOCHS = 50

    img_dataset = ImageINRDataset(img_fp)
    img_dataloader = DataLoader(img_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    num_batches = len(img_dataloader)

    print(f"Loaded {len(img_dataset)} pixels from {img_fp}")

    model = SIREN_IMG_MODEL().to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # loss_fn = lambda y_true, y_pred: torch.nn.MSELoss()(torch.clamp(y_pred, 0, 1), torch.clamp(y_true, 0, 1))
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        with tqdm.tqdm(total=num_batches, unit=" batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            l = 0
            count = 0

            for batch in img_dataloader:
                tepoch.update(1)
                
                optimizer.zero_grad()

                x_in = batch["x_in"].to("cuda")
                x_out = batch["x_out"].to("cuda")

                y = model(x_in)
                loss = loss_fn(y, x_out)

                loss.backward()
                optimizer.step()

                count += 1
                l += loss.item()

                tepoch.set_postfix({'Loss': l / count})

        print(f"Epoch {epoch}: {l / count}")
    
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{img_fp.stem}.pt")

if __name__ == "__main__":
    fit_image(IMAGES_DIR / "demo_image_00.jpg")
