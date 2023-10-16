import rasterio as rio
from torchgeo.datasets import RasterDataset
from typing import List
import torch


def scale_image(item: dict):
    item["image"] = item["image"] / 10000
    return item


def calc_statistics(dset: RasterDataset):
    """
    Calculate the statistics (mean and std) for the entire dataset
    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.
    For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    files = [
        item.object for item in dset.index.intersection(dset.index.bounds, objects=True)
    ]

    # Reseting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        img = rio.open(file).read() / 10000  # type: ignore
        accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
        accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with lenght n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)


class Normalizer(torch.nn.Module):
    def __init__(self, mean: List[float], stdev: List[float]):
        super().__init__()
        self.mean = torch.Tensor(mean)[:, None, None]
        self.std = torch.Tensor(stdev)[:, None, None]

    def forward(self, inputs: torch.Tensor):
        """
        Normalize the batch.
        """
        x = inputs[..., : len(self.mean), :, :]
        # if batch
        if inputs.ndim == 4:
            x = (x - self.mean[None, ...]) / self.std[None, ...]
        else:
            x = (x - self.mean) / self.std
        inputs[..., : len(self.mean), :, :] = x
        return inputs

    def revert(self, inputs: torch.Tensor):
        """
        De-normalize the batch.
        """
        x = inputs[..., : len(self.mean), :, :]
        # if batch
        if x.ndim == 4:
            x = inputs[:, : len(self.mean), ...]
            x = x * self.std[None, ...] + self.mean[None, ...]
        else:
            x = x * self.std + self.mean
        inputs[..., : len(self.mean), :, :] = x
        return inputs
