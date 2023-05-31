import torch
import torch.nn as nn

CONFIG = {
    "cfg": {
        "C1": {"dim": 8, "kernel_size": 7, "stride": 2, "padding": 3},
        "M1": {"kernel_size": 3, "stride": 2, "padding": 1},
    },
    "convlstm_dims": [16, 32, 64],
    "convlstm_kernel_size": [(7, 7), (3, 3), (3, 3)],
}


class DBSeNet(nn.Module):
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
        super(DBSeNet, self).__init__()
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
            in_features=convlstm_dims[-1] * r**2,
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
        x = self.flatten(x)
        x = self.classifier(x)
        return x


from collections import OrderedDict

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def make_mlp(
    in_features: int,
    hidden_feature_list: list = None,
    out_features=None,
    act_layer=nn.GELU,
    drop=0.0,
    init_method=None,
):
    if hidden_feature_list is None:
        return nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, out_features)),
                    ("drop1", nn.Dropout(p=drop)),
                ]
            )
        )
    else:
        mlp = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, hidden_feature_list[0])),
                    ("act1", act_layer()),
                    ("drop1", nn.Dropout(p=drop)),
                ]
            )
        )
        for i in range(1, len(hidden_feature_list)):
            mlp.add_module(
                f"fc{i+1}",
                nn.Linear(hidden_feature_list[i - 1], hidden_feature_list[i]),
            )
            mlp.add_module(f"act{i+1}", act_layer())
            mlp.add_module(f"drop{i+1}", nn.Dropout(p=drop))
        num_linear = len(hidden_feature_list) + 1
        mlp.add_module(
            f"fc{num_linear}",
            nn.Linear(hidden_feature_list[-1], out_features),
        )
        mlp.add_module(
            f"drop{num_linear}",
            nn.Dropout(p=drop),
        )
        if init_method is not None:
            mlp.apply(init_method)
        return mlp
