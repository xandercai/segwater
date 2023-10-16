from pathlib import Path

import torch
from osgeo import gdal
from pyproj import CRS
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units
from torchgeo.transforms import indices

from src.data import scale_image, calc_statistics, Normalizer
from src.loss import loss, oa, iou
from src.model import Model, init_weights
from src.plot import plot_batch
from src.train import train_loop
from src.utils import get_env_variable

gdal.PushErrorHandler(
    "CPLQuietErrorHandler"
)  # for suppressing "TIFFReadDirectory..." warning


# Part 1. Build dataloader
# data source https://zenodo.org/record/5205674
dataset_dir = Path(get_env_variable("DATASET_DIR"))
resolution = int(get_env_variable("RESOLUTION"))
crs = CRS.from_epsg(get_env_variable("CRS"))

train_images = RasterDataset(
    paths=(dataset_dir / "tra_scene").as_posix(),
    crs=crs,
    res=resolution,
    transforms=scale_image,
)
train_masks = RasterDataset(
    paths=(dataset_dir / "tra_truth").as_posix(), crs=crs, res=resolution
)

valid_images = RasterDataset(
    paths=(dataset_dir / "val_scene").as_posix(),
    crs=crs,
    res=resolution,
    transforms=scale_image,
)
valid_masks = RasterDataset(
    paths=(dataset_dir / "val_truth").as_posix(), crs=crs, res=resolution
)

train_masks.is_image = False
valid_masks.is_image = False

# intersection of the two datasets
train_set = train_images & train_masks
valid_set = valid_images & valid_masks

train_size = len(train_set)
valid_size = len(valid_set)

sample_scale = int(get_env_variable("SAMPLE_SCALE"))
sample_size = int(get_env_variable("SAMPLE_SIZE"))

train_sampler = RandomGeoSampler(
    train_images, size=sample_size, length=train_size * sample_scale, units=Units.PIXELS
)
valid_sampler = RandomGeoSampler(
    valid_images, size=sample_size, length=valid_size * sample_scale, units=Units.PIXELS
)

batch_size = int(get_env_variable("BATCH_SIZE"))

train_dataloader = DataLoader(
    train_set, sampler=train_sampler, batch_size=batch_size, collate_fn=stack_samples
)
valid_dataloader = DataLoader(
    valid_set, sampler=valid_sampler, batch_size=batch_size, collate_fn=stack_samples
)
train_batch = next(iter(train_dataloader))
valid_batch = next(iter(valid_dataloader))

# verify
print("verify a train_batch", train_batch.keys(), train_batch["image"].shape)
plot_batch(train_batch.copy())
print("verify a valid_batch", valid_batch.keys(), valid_batch["image"].shape)
plot_batch(valid_batch.copy())


# Part 2. Build normalizer
mean, std = calc_statistics(train_images)
print("train dataset mean: ", mean, "train dataset std: ", std)

normalizer = Normalizer(mean=mean, stdev=std)

# verify
norm_batch = normalizer(train_batch["image"].clone())
print("verify a norm_batch", norm_batch.shape)
plot_batch(norm_batch, train_batch["mask"].clone())
revert_batch = normalizer.revert(norm_batch)
print("verify a revert_batch", revert_batch.shape)
plot_batch(revert_batch, train_batch["mask"].clone())


# Part 3. Feature engineering
transformers = torch.nn.Sequential(
    indices.AppendNDWI(index_green=1, index_nir=3),
    indices.AppendNDWI(index_green=1, index_nir=5),
    indices.AppendNDVI(index_nir=3, index_red=2),
    normalizer,
)


# Part 4. Model
# weights is defined in src/weights.py
# DeepLabV3 with a ResNet-50 backbone
# from scratch
model = init_weights(Model)
print(model)


# Part 5. Loss, metrics, and optimizer
# loss is defined in src/loss.py, cross entropy
# metrics are overall-accuracy and intersection-over-union, oa and iou are defined in src/loss.py
epochs = 30
learning_rate = 0.1
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=len(train_dataloader),
    epochs=epochs,
)


# Part 6. Training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

acc = train_loop(
    epochs,
    train_dataloader,
    valid_dataloader,
    model,
    loss,
    optimizer,
    scheduler,
    acc_fns=[oa, iou],
    batch_tfms=transformers,
    device=device,
)


# Part 7. Save weights
file_name = "acc_" + "_".join([str(i)[:5] for i in acc]) + ".pth"
file_path = Path("./weights/" + file_name)
file_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "./weights/" + file_name)
