import torch.nn.init as init
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50


def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
    return model


# Do not load the pre-trained weight
Model = deeplabv3_resnet50(weights=None, num_classes=2)

backbone = Model.get_submodule("backbone")
conv = nn.modules.conv.Conv2d(
    in_channels=9,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)
backbone.register_module("conv1", conv)
