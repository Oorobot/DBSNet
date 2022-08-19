import argparse
import io
import os
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from utils.dataset import ThreePhaseBone
from torch.utils.data import DataLoader

from model.convnext_3d import convnext_base, convnext_tiny
from model.customed_model import DPBNet, CFG_DPB

from model.densenet_3d import densenet121_3d

from model.resnet_3d import resnet18, resnet50, resnet101
from model.vgg_3d import vgg11_3d, vgg19_3d
from utils.utils import calculate_metrics, load_json


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def create_model(args):
    model = None
    if args.dataset == "knee" or args.dataset == "hip" or args.dataset == "hip_":
        img_size = 128
    elif args.dataset == "hip_focus":
        img_size = 40

    models = {
        "tpb": DPBNet, # DBS-eNet
        "densenet121_3d": densenet121_3d,
        "resnet18_3d": resnet18,
        "resnet50_3d": resnet50,
        "resnet101_3d": resnet101,
        "vgg11_3d": vgg11_3d,
        "vgg19_3d": vgg19_3d,
        "convnext_tiny_3d": convnext_tiny,
        "convnext_base_3d": convnext_base,
    }
    # create model
    if args.net.startswith("tpb"):
        model = models[args.net[:-1]](
            **CFG_DPB[args.net[3:]],
            num_classes=args.num_classes,
            drop_rate=args.drop_rate,
            img_size=img_size,
        )
    elif args.net.startswith("densenet"):
        model = models[args.net](num_classes=args.num_classes, drop_rate=args.drop_rate)
    else:
        model = models[args.net](num_classes=args.num_classes)

    return model


def load_model(model: nn.Module, trained_model_path: str, fusion=True):
    model_params = torch.load(trained_model_path)["model_params"]
    if fusion:
        new_model_params = OrderedDict()
        for key, value in model_params.items():
            if key.find("convlstm") != -1:
                key = key.replace("convlstm.", "convlstm1.")
                key = key.replace("convlstm_other.", "convlstm2.")
            new_model_params[key] = value
        model.load_state_dict(new_model_params)
    else:
        model.load_state_dict(model_params)
    return model


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_lossFunc: nn.Module,
    test_lossFunc: nn.Module,
    optimizer,
    device: str,
    trainloader: DataLoader,
    testloader: DataLoader,
    tf_writer,
    log_training: io.TextIOWrapper,
):
    """train model using loss_fn and optimizer. When this function is called, model trains for one epoch.
    Args:
        train_loader: train data
        model: prediction model
        loss_fn: loss function to judge the distance between target and output
        optimizer: optimize the loss function
    """
    model.train()  # enter train mode
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    for _, (input, target) in enumerate(trainloader):
        input = input.to(device)
        # if _ == 0:
        #     image = torchvision.utils.make_grid(torch.squeeze(input), 5)
        #     tf_writer.add_image("train/image", image, epoch)
        if isinstance(target, (tuple, list)):
            target1, target2, lam = target
            target = (target1.to(device), target2.to(device), lam)
        else:
            target = target.to(device)
        # 梯度
        optimizer.zero_grad()
        output = model(input)
        loss = train_lossFunc(output, target)  # compute loss
        loss.backward()  # compute gradient of loss over parameters
        optimizer.step()  # update parameters with gradient descent
        # loss, acc

        num = input.size(0)
        _, prediction = torch.max(output, dim=1)
        if isinstance(target, (tuple, list)):
            target1, target2, lam = target
            correct1 = torch.eq(prediction, target1).sum().item()
            correct2 = torch.eq(prediction, target2).sum().item()
            accuracy = (lam * correct1 + (1 - lam) * correct2) / num
        else:
            correct = torch.eq(prediction, target).sum().item()
            accuracy = correct / num
        train_loss_meter.update(loss.item(), num)
        train_accuracy_meter.update(accuracy, num)

    model.eval()  # enter test mode
    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    with torch.no_grad():
        for _, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = test_lossFunc(output, target)

            num = input.size(0)
            _, prediction = torch.max(output, dim=1)
            correct = torch.eq(prediction, target).sum().item()
            accuracy = correct / num
            test_loss_meter.update(loss.item(), num)
            test_accuracy_meter.update(accuracy, num)

    # log
    tf_writer.add_scalars(
        "loss", {"train": train_loss_meter.avg, "test": test_loss_meter.avg}, epoch
    )
    tf_writer.add_scalars(
        "accuracy",
        {"train": train_accuracy_meter.avg, "test": test_accuracy_meter.avg},
        epoch,
    )

    output = "Epoch {0} - loss: {1:.4f} - accuracy: {2:.2%}({5:.1f}/{6:.1f}) - test loss: {3:.4f} - test accuracy: {4:.2%}({7:.0f}/{8:.0f})".format(
        epoch + 1,
        train_loss_meter.avg,
        train_accuracy_meter.avg,
        test_loss_meter.avg,
        test_accuracy_meter.avg,
        train_accuracy_meter.sum,
        train_accuracy_meter.count,
        test_accuracy_meter.sum,
        test_accuracy_meter.count,
    )
    log_training.write(output + "\n")
    log_training.flush()
    print(output)

    return (
        train_loss_meter.avg,
        train_accuracy_meter.avg,
        test_loss_meter.avg,
        test_accuracy_meter.avg,
    )


def validate(
    model: nn.Module, device: str, dataloader: DataLoader, is_train: bool = False,
):

    model.eval()  # enter test mode
    ground_truth, predicted_score = [], []
    with torch.no_grad():
        for _, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            ground_truth.append(target.cpu().numpy())
            predicted_score.append(output.cpu().numpy())
    ground_truth = np.concatenate(ground_truth, axis=0)
    predicted_score = np.concatenate(predicted_score, axis=0)
    predicted_label = np.argmax(predicted_score, axis=1)
    if is_train:
        metrics = calculate_metrics(ground_truth, predicted_label)
        return metrics
    else:
        return ground_truth, predicted_score, predicted_label


def trained_model_output(log: str, file_types: List[str]):
    # get folder path
    checkpoint_dir = os.path.dirname(log)
    log = load_json(log)
    arguments = argparse.Namespace(**log["arguments"])
    model = create_model(arguments)
    data_file = load_json(arguments.data_file)
    fusion = arguments.data_type == "none"

    five = []
    for fold_name, files in data_file.items():
        model = load_model(
            model,
            os.path.join(checkpoint_dir, "model", f"bestacc_{fold_name}.pth"),
            fusion,
        )
        dataloaders = [
            DataLoader(
                dataset=ThreePhaseBone(
                    files[f],
                    arguments.data_type,
                    arguments.dicom_window_ratio,
                    is_train=False,
                    num_classes=arguments.num_classes,
                )
            )
            for f in file_types
        ]
        five.append([validate(model, "cpu", dataloder) for dataloder in dataloaders])
    return five

