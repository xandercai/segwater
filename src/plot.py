from typing import Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import torch
from torchgeo.datasets import unbind_samples


def plot_images(
    images: Iterable, axs: Iterable, chnls: List[int] = [2, 1, 0], bright: float = 3.0
):
    for img, ax in zip(images, axs):
        arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis("off")


def plot_masks(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy(), cmap="Blues")
        ax.axis("off")


def plot_batch(
    batch: Union[dict, torch.Tensor],
    masks: Optional[torch.Tensor] = None,
    bright: float = 3.0,
    cols: int = 4,
    width: int = 5,
    chnls: List[int] = [2, 1, 0],
):
    if isinstance(batch, dict):
        # Get the samples and the number of items in the batch
        samples = unbind_samples(batch)
        # if batch contains images and masks, the number of images will be doubled
        n = (
            2 * len(samples)
            if ("image" in batch) and ("mask" in batch)
            else len(samples)
        )
        # calculate the number of rows in the grid
        img_samples = map(lambda x: x["image"], samples) if "image" in batch else None
        msk_samples = map(lambda x: x["mask"], samples) if "mask" in batch else None
    else:
        n = 2 * len(batch) if masks is not None else len(batch)
        img_samples = batch
        msk_samples = masks if masks is not None else None
    rows = n // cols + (1 if n % cols != 0 else 0)
    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if img_samples is not None and msk_samples is not None:
        # plot the images on the even axis
        plot_images(
            images=img_samples,
            axs=axs.reshape(-1)[::2],
            chnls=chnls,
            bright=bright,
        )
        # plot the masks on the odd axis
        plot_masks(masks=msk_samples, axs=axs.reshape(-1)[1::2])
    else:
        if img_samples is not None:
            plot_images(
                images=img_samples,
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )
        elif msk_samples is not None:
            plot_masks(masks=msk_samples, axs=axs.reshape(-1))
    plt.show()
