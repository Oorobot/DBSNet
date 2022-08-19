import torch
import torch.nn as nn
from timm.models.layers import Mlp

from .convlstm import ConvLSTM
from .modules import ConvNeXtBlock, ConvNeXtBlock_1, DeformConv2d, LayerNorm, make_mlp


CFG_DPB = {
    "0": {
        "cfg": {
            "C1": {"dim": 8, "kernel_size": 7, "stride": 2, "padding": 3},
            "M1": {"kernel_size": 3, "stride": 2, "padding": 1},
        },
        "convlstm_dims": [16, 32, 64],
        "convlstm_kernel_size": [(7, 7), (3, 3), (3, 3)],
    },
}


class DPBNet(nn.Module):
    def __init__(
        self,
        cfg: dict,
        in_channels=1,
        num_classes=2,
        # LSTM
        convlstm_dims=[16, 32, 64],
        convlstm_kernel_size=[(7, 7), (3, 3), (3, 3)],
        # MLP
        mlp_hidden_feature=[1024],
        drop_rate=0.2,
        # 其他
        img_size=128,
    ) -> None:
        super(DPBNet, self).__init__()
        r = img_size
        cnn_layers = []
        pre_channels = in_channels
        for key, value in cfg.items():
            if key[0] == "C":
                cnn_layers += [
                    nn.Conv3d(
                        in_channels=pre_channels,
                        out_channels=value["dim"],
                        kernel_size=(1, value["kernel_size"], value["kernel_size"]),
                        stride=(1, value["stride"], value["stride"]),
                        padding=(0, value["padding"], value["padding"]),
                    ),
                    nn.BatchNorm3d(value["dim"]),
                    nn.ReLU(inplace=True),
                ]
                pre_channels = value["dim"]
                r = (r - value["kernel_size"] + 2 * value["padding"]) // value[
                    "stride"
                ] + 1
            elif key[0] == "M":
                cnn_layers += [
                    nn.MaxPool3d(
                        kernel_size=(1, value["kernel_size"], value["kernel_size"]),
                        stride=(1, value["stride"], value["stride"]),
                        padding=(0, value["padding"], value["padding"]),
                    )
                ]
                r = (r - value["kernel_size"] + 2 * value["padding"]) // value[
                    "stride"
                ] + 1
            else:
                raise Exception("there is undefined cfg content.")
        self.features = nn.Sequential(*cnn_layers)

        convlstm_num_layers = len(convlstm_dims)
        self.convlstm1 = ConvLSTM(
            pre_channels,
            convlstm_dims,
            convlstm_kernel_size,
            convlstm_num_layers,
            batch_first=True,
        )
        self.convlstm2 = ConvLSTM(
            pre_channels,
            convlstm_dims,
            convlstm_kernel_size,
            convlstm_num_layers,
            batch_first=True,
        )

        self.classifier = make_mlp(
            in_features=convlstm_dims[-1] * r ** 2,
            hidden_feature_list=mlp_hidden_feature,
            out_features=num_classes,
            drop=drop_rate,
        )

        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        # shape: (B, C, D, H, W) -> (B, T, C, H, W)
        features = features.permute(0, 2, 1, 3, 4)
        assert features.shape[1] == 25
        _, last_state1 = self.convlstm1(features[:, :20, :, :, :])
        _, last_state2 = self.convlstm2(features[:, 20:, :, :, :])
        x = last_state1[0][0] + last_state2[0][0]
        # x = torch.cat((last_state1[0][0], last_state2[0][0]), dim=1)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
