import random

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


# data augmentation
def random_horizontal_flip(imgs: np.ndarray):
    return np.flip(imgs, -1)


def random_rotate(imgs: np.ndarray):
    angle = np.random.randint(-20, 20)
    imgs = ndimage.rotate(imgs, angle, axes=(1, 2), order=0, reshape=False)
    return imgs


def random_rotate_flip(imgs: np.ndarray):
    k = random.randint(0, 3)
    axis = random.randint(1, 2)
    imgs = np.rot90(imgs, k, axes=(1, 2))
    imgs = np.flip(imgs, axis)
    return imgs


def data_augmentation(augmentation: int, is_train: bool):
    augmentations = []
    if not is_train or augmentation == 0:
        return augmentations
    if augmentation == 1:
        augmentations.append(random_horizontal_flip)
    elif augmentation == 2:
        augmentations.append(random_rotate_flip)
    elif augmentations == 3:
        augmentations.append(random_rotate)
    elif augmentations == -1:
        augmentations.append(random_horizontal_flip)
        augmentations.append(random_rotate)
    return augmentations


# based dicom_window_ratio
def normalize(
    input: np.ndarray,
    mask: np.ndarray = None,
    dicom_window_ratio: float = 0.5,
    reduction_ratio: float = 1.0 / 6.0,
):
    if input.shape[0] == 25:
        flow_max = np.max(input[00:20]) * dicom_window_ratio
        pool_max = np.max(input[20:25]) * dicom_window_ratio
        if mask is not None:
            input = input * (1 - mask) + input * (mask * reduction_ratio)
        normalized_flow = np.clip(input[00:20], 0, flow_max) / flow_max
        normalized_pool = np.clip(input[20:25], 0, pool_max) / pool_max
        return np.concatenate((normalized_flow, normalized_pool), axis=0)
    else:
        max = np.max(input) * dicom_window_ratio
        if mask is not None:
            input = input * (1 - mask) + input * (mask * reduction_ratio)
        normalized_input = np.clip(input, 0, max) / max
        return normalized_input


# based normal hip
def preprocess_based_normal_hip(
    input: np.ndarray,
    side: str = "l" or "r",
    normal_hip_file: str = "data/hip_roi/avg_normal_hip.npz",
):
    normal_hip = np.load(normal_hip_file)
    normal_hip = normal_hip["left"] if side == "l" else normal_hip["right"]
    if input.shape[0] == 20:
        normal_hip = normal_hip[00:20]
    if input.shape[0] == 5:
        normal_hip = normal_hip[20:25]
    return input - normal_hip


def customed_transform(
    input: np.ndarray,
    target: np.ndarray,
    dicom_window_ratio: float = 0.5,
    mask=None,
    boundary=None,
    side: str = None,
    augmentations: list = [],
    num_classes=None,
):
    if boundary is None:
        input = normalize(input, mask, dicom_window_ratio)
    else:
        if num_classes == 2:
            input = input[
                :, boundary[0] : boundary[1] + 1, boundary[2] : boundary[3] + 1
            ]
            # hip: ROI
            target = np.array(target - 1)
            input = preprocess_based_normal_hip(input, side)
        else:
            input = normalize(input, mask, dicom_window_ratio)
            input = input[
                :, boundary[0] : boundary[1] + 1, boundary[2] : boundary[3] + 1
            ]

    for aug in augmentations:
        if random.random() > 0.5:
            input = aug(input)

    # (B, T, H, W) -> (B, 1, T, H, W)
    input = torch.from_numpy(input.astype(np.float32)).unsqueeze(0)
    target = torch.from_numpy(target.astype(np.long))
    return input, target


class ThreePhaseBone(Dataset):
    def __init__(
        self,
        file_list,
        data_type,
        dicom_window_ratio=0.5,
        is_train=True,
        augmentation_type=-1,
        num_classes=None,
    ) -> None:
        super(ThreePhaseBone, self).__init__()
        self.file_list = file_list
        self.data_type = data_type
        self.dicom_window_ratio = dicom_window_ratio
        self.augmentations = data_augmentation(augmentation_type, is_train)
        self.num_classes = num_classes

    def __getitem__(self, index):
        # medical images already were converted to npz
        file_name = self.file_list[index]
        data = np.load(file_name)

        if self.data_type == "flow":
            input = data["data"][:20]
        elif self.data_type == "pool":
            input = data["data"][20:]
        elif self.data_type == "none":
            input = data["data"]
        target = data["label"]
        mask = None if "mask" not in data else data["mask"]

        # filename: no_side_label.npz. e.g. 001_l_3.npz
        (boundary, side) = (
            (None, None)
            if "boundary" not in data
            else (data["boundary"], file_name[-7])
        )

        input, target = customed_transform(
            input=input,
            target=target,
            dicom_window_ratio=self.dicom_window_ratio,
            mask=mask,
            boundary=boundary,
            side=side,
            augmentations=self.augmentations,
            num_classes=self.num_classes,
        )

        return input, target

    def __len__(self):
        return len(self.file_list)
