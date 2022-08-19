from turtle import forward
from typing import Any, Dict, List, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import make_mlp

cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self,
        cfg: List[Any],
        num_classes: int = 2,
        in_chans: int = 1,
        output_stride: int = 32,
        mlp_ratio: float = 1.0,
        act_layer: nn.Module = nn.ReLU,
        conv_layer: nn.Module = nn.Conv3d,
        norm_layer: nn.Module = None,
        drop_rate: float = 0,
    ):
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.use_norm = norm_layer is not None
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool3d

        layers: List[nn.Module] = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == "M":
                layers += [pool_layer(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv3d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv3d, norm_layer(v), act_layer(inplace=True)]
                else:
                    layers += [conv3d, act_layer(inplace=True)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = make_mlp(
            prev_chs * 8,
            [self.num_features, self.num_features],
            self.num_classes,
            act_layer,
            drop_rate,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.shape[2] == 5 or x.shape[2] == 20:
            x = F.interpolate(x, (64, 64, 64), mode="nearest")
            x = self.features(x)
        elif x.shape[2] == 25:
            x1 = F.interpolate(x[:, :, 0:20, :, :], (64, 64, 64), mode="nearest")
            x2 = F.interpolate(x[:, :, 20:, :, :], (64, 64, 64), mode="nearest")
            x = self.features(x1) + self.features(x2)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def vgg11_3d(**kwargs: Any) -> VGG:
    return VGG(cfg=cfgs["vgg11"], **kwargs)


def vgg13_3d(**kwargs: Any) -> VGG:
    return VGG(cfg=cfgs["vgg13"], **kwargs)


def vgg16_3d(**kwargs: Any) -> VGG:
    return VGG(cfg=cfgs["vgg16"], **kwargs)


def vgg19_3d(**kwargs: Any) -> VGG:
    return VGG(cfg=cfgs["vgg19"], **kwargs)


# x = torch.randn(1, 1, 64, 64, 64)
# model = vgg19()
# out = model(x)
# print(0)

