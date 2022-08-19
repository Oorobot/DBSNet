from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    # DenseBlock： BN+ReLU+1x1Conv + BN+ReLU+3x3Conv
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm3d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv3d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv3d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


# DenseBlock ，内部是密集连接方式（输入特征数线性增长）
class _DenseBlock(nn.Sequential):
    """DenseBlock"""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module("denselayer%d" % (i + 1,), layer)


# Transition Layer
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm3d(num_input_features))  # bn
        self.add_module("relu", nn.ReLU(inplace=True))  # relu
        self.add_module(
            "conv",
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))  # avg pool


class DenseNet(nn.Module):
    "DenseNet-BC model"

    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression_rate=0.5,
        drop_rate=0,
        num_classes=1000,
    ):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv3d(
                            1,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(3, stride=2, padding=1)),
                ]
            )
        )

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers, num_features, bn_size, growth_rate, drop_rate
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(
                    num_features, int(num_features * compression_rate)
                )
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        # BN2d -> BN3d
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)
        # self.classifier = nn.Linear(30608, num_classes)

        # params initialization
        for m in self.modules():
            # Conv2d -> Conv3d
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            # BN2d -> BN3d
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.shape[2] == 5 or x.shape[2] == 20:
            x = F.interpolate(x, (64, 64, 64), mode="nearest")
            features = self.features(x)
        elif x.shape[2] == 25:
            x1 = F.interpolate(x[:, :, 0:20, :, :], (64, 64, 64), mode="nearest")
            x2 = F.interpolate(x[:, :, 20:, :, :], (64, 64, 64), mode="nearest")
            features = self.features(x1) + self.features(x2)
        # AvgPool2d -> AvgPool3d
        out = F.adaptive_avg_pool3d(features, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121_3d(**kwargs):
    model = DenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
    )
    return model


def densenet169(**kwargs):
    model = DenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs
    )
    return model


def densenet201(**kwargs):
    model = DenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs
    )
    return model


def densenet161(**kwargs):
    model = DenseNet(
        num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs
    )
    return model


class DenseNet_F1(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression_rate=0.5,
        drop_rate=0,
        num_classes=1000,
    ):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet_F1, self).__init__()
        self.downsample_layers = nn.ModuleList()
        self.features = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    1,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm3d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, stride=2, padding=1),
            )
        )
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    1,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm3d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, stride=2, padding=1),
            )
        )
        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers, num_features, bn_size, growth_rate, drop_rate
            )
            self.features.append(block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition_1 = _Transition(
                    num_features, int(num_features * compression_rate)
                )
                transition_2 = _Transition(
                    num_features, int(num_features * compression_rate)
                )
                self.downsample_layers.append(transition_1)
                self.downsample_layers.append(transition_2)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.append(nn.BatchNorm3d(num_features))
        self.features.append(nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert x.shape[2] == 25
        x1 = F.interpolate(x[:, :, 0:20, :, :], (64, 64, 64), mode="nearest")
        x2 = F.interpolate(x[:, :, 20:, :, :], (64, 64, 64), mode="nearest")
        for i in range(4):
            x1 = self.downsample_layers[2 * i](x1)
            x2 = self.downsample_layers[2 * i + 1](x2)
            x1 = self.features[i](x1)
            x2 = self.features[i](x2)
        x1 = self.features[4](x1)
        x1 = self.features[5](x1)
        x2 = self.features[4](x2)
        x2 = self.features[5](x2)
        # x = torch.cat([x1, x2], dim=1)
        x = x1 + x2
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        # print("features shape:", features.shape)
        # print("here!!! the shape: ", out.shape)
        out = self.classifier(x)
        return out


def densenet121_f1(**kwargs):
    model = DenseNet_F1(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
    )
    return model


class DenseNet_F2(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression_rate=0.5,
        drop_rate=0,
        num_classes=1000,
    ):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet_F2, self).__init__()
        self.downsample_layers = nn.ModuleList()
        self.features = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    1,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm3d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, stride=2, padding=1),
            )
        )
        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers, num_features, bn_size, growth_rate, drop_rate
            )
            block_ = _DenseBlock(
                num_layers, num_features, bn_size, growth_rate, drop_rate
            )
            self.features.append(block)
            self.features.append(block_)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(
                    num_features, int(num_features * compression_rate)
                )
                self.downsample_layers.append(transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.append(nn.BatchNorm3d(num_features))
        self.features.append(nn.BatchNorm3d(num_features))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert x.shape[2] == 25
        x1 = F.interpolate(x[:, :, 0:20, :, :], (64, 64, 64), mode="nearest")
        x2 = F.interpolate(x[:, :, 20:, :, :], (64, 64, 64), mode="nearest")
        for i in range(4):
            x1 = self.downsample_layers[i](x1)
            x2 = self.downsample_layers[i](x2)
            x1 = self.features[2 * i](x1)
            x2 = self.features[2 * i + 1](x2)
        x1 = self.features[8](x1)
        x2 = self.features[9](x2)
        x1 = self.features[10](x1)
        x2 = self.features[11](x2)
        # x = torch.cat([x1, x2], dim=1)
        x = x1 + x2
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        # print("features shape:", features.shape)
        # print("here!!! the shape: ", out.shape)
        out = self.classifier(x)
        return out


def densenet121_f2(**kwargs):
    model = DenseNet_F2(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
    )
    return model


class DenseNet_F3(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression_rate=0.5,
        drop_rate=0,
        num_classes=1000,
    ):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet_F3, self).__init__()
        self.features1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv3d(
                            1,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(3, stride=2, padding=1)),
                ]
            )
        )
        self.features2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv3d(
                            1,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(3, stride=2, padding=1)),
                ]
            )
        )

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block1 = _DenseBlock(
                num_layers, num_features, bn_size, growth_rate, drop_rate
            )
            block2 = _DenseBlock(
                num_layers, num_features, bn_size, growth_rate, drop_rate
            )
            self.features1.add_module("denseblock%d" % (i + 1), block1)
            self.features2.add_module("denseblock%d" % (i + 1), block2)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition1 = _Transition(
                    num_features, int(num_features * compression_rate)
                )
                transition2 = _Transition(
                    num_features, int(num_features * compression_rate)
                )
                self.features1.add_module("transition%d" % (i + 1), transition1)
                self.features2.add_module("transition%d" % (i + 1), transition2)
                num_features = int(num_features * compression_rate)

        self.features1.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features1.add_module("relu5", nn.ReLU(inplace=True))
        self.features2.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features2.add_module("relu5", nn.ReLU(inplace=True))

        self.classifier = nn.Linear(num_features, num_classes)
        # params initialization
        for m in self.modules():
            # Conv2d -> Conv3d
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            # BN2d -> BN3d
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        assert x.shape[2] == 25
        x1 = F.interpolate(x[:, :, 0:20, :, :], (64, 64, 64), mode="nearest")
        x2 = F.interpolate(x[:, :, 20:, :, :], (64, 64, 64), mode="nearest")
        features = self.features1(x1) + self.features2(x2)
        # AvgPool2d -> AvgPool3d
        out = F.adaptive_avg_pool3d(features, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121_f3(**kwargs):
    model = DenseNet_F3(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
    )
    return model
