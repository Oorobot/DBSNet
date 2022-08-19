import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from utils.dataset import ThreePhaseBone
from model.cutmix import CutMixCollator, CutMixCriterion
from model.loss import MultiFocalLoss
from utils.utils import load_json, mkdir, save_json
from utils.model import (
    create_model,
    train_one_epoch,
    validate,
)

parser = argparse.ArgumentParser()

# data
parser.add_argument("--dataset", type=str)  # knee, hip, hip_ , hip_focus
parser.add_argument("--data_type", type=str, default="none")  #  flow, pool, none = flow + pool
parser.add_argument("--data_file", type=str, default="data/dicom_hip_roi.json")
parser.add_argument("--dicom_window_ratio", type=float, default=0.5)

# model
parser.add_argument("--net", type=str, default="DBS-eNet")
parser.add_argument("--drop_rate", type=float, default=0.2)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--evaluate", type=bool, default=True)

# optimizer
# [sgd] momentum=0.8, weight_decay=5e-4
# [adam] betas=[0.9, 0.98], eps=1e-9
# [adamw] betas=[0.9, 0.999], weight_decay=5e-2
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--betas", type=list, default=[0.9, 0.999])
parser.add_argument("--momentum", type=float, default=0.8)
parser.add_argument("--weight_decay", type=float, default=5e-4)
# parser.add_argument("--eps", type=float, default=1e-9)

# data augmentation
parser.add_argument("--augmentation_type", type=int, default=-1)
parser.add_argument("--use_cutmix", action="store_true")
parser.add_argument("--cutmix_alpha", type=float, default=1.0)


def main():
    args = parser.parse_args()
    if args.seed is not None:  # Fixed seeds can be reproduced
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cuda.manual_seed(args.seed)
        cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
    checkpoint_dir = "./checkpoint/%s/%s_%s/%s" % (
        args.dataset,
        args.data_type,
        args.net,
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    )

    log = {"arguments": vars(args), "test": {}, "validate": {}}
    print(str(args))

    five_fold = load_json(args.data_file)
    # five-fold cross validation
    for fold_name, each_fold in five_fold.items():

        # create model
        model = create_model(args)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # create dataset
        train_files = each_fold["train"]
        test_files = each_fold["test"]
        if each_fold.get("validate"):
            validate_files = each_fold["validate"]
        else:
            validate_files = test_files
        if args.use_cutmix:
            collator = CutMixCollator(args.cutmix_alpha)
        else:
            collator = torch.utils.data.dataloader.default_collate
        trainloader = DataLoader(
            dataset=ThreePhaseBone(
                train_files,
                args.data_type,
                args.dicom_window_ratio,
                augmentation_type=args.augmentation_type,
                num_classes=args.num_classes,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        testloader = DataLoader(
            dataset=ThreePhaseBone(
                test_files,
                args.data_type,
                args.dicom_window_ratio,
                augmentation_type=args.augmentation_type,
                is_train=False,
                num_classes=args.num_classes,
            )
        )
        validateloader = DataLoader(
            dataset=ThreePhaseBone(
                validate_files,
                args.data_type,
                args.dicom_window_ratio,
                augmentation_type=args.augmentation_type,
                is_train=False,
                num_classes=args.num_classes,
            )
        )

        # log
        tf_writer = SummaryWriter(
            logdir=os.path.join(checkpoint_dir, f"visualization/{fold_name}")
        )
        log_training = open(os.path.join(checkpoint_dir, "log_train.log"), "a")

        model_save_folder = os.path.join(checkpoint_dir, "model")
        mkdir(model_save_folder)

        # loss function and optimizer
        if args.dataset == "knee":
            lossFunction = nn.CrossEntropyLoss()
        else:
            lossFunction = MultiFocalLoss(
                num_class=2,
                alpha=28.0 / (28 + 16),
                gamma=1,
                balance_index=1,
                smooth=0.15,
            )
        # lossFunction = nn.CrossEntropyLoss()
        if args.use_cutmix:
            train_lossFunc = CutMixCriterion(lossFunction)
        else:
            train_lossFunc = lossFunction
        # n_samples = np.array([172, 99])
        # class_weight_1 = 1 / n_samples
        # class_weight_2 = 1 - n_samples / sum(n_samples)
        # class_weight_3 = max(n_samples) / n_samples
        # class_weight_4 = sum(n_samples) / (2 * n_samples)
        # norm_weight = torch.FloatTensor(class_weight_4).to(device)
        # log_training.write(str(norm_weight) + "\n")
        # log_training.flush()
        # lossFunction = nn.CrossEntropyLoss(weight=norm_weight)
        # lossFunction = nn.CrossEntropyLoss()

        if args.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                args.lr,
                betas=args.betas,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )

        # lr_scheduler
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
            gamma=0.1,
        )

        if fold_name.find("1") != -1:
            save_json(f"{checkpoint_dir}/log.json", log)
            log_training.write("===> model <===\n" + str(model))
            log_training.write(
                "\nLoss Function: {0}.\n Optimizer: {1}.\n".format(
                    str(lossFunction), str(optimizer)
                )
            )
            log_training.flush()

        log_training.write(f"===> {fold_name}\n")
        print(f"===> {fold_name}")
        log_training.flush()

        # training
        best_acc_loss = np.inf
        best_acc = 0.0
        cudnn.benchmark = True
        for epoch in range(args.epochs):
            # training and validation
            train_loss, train_acc, test_loss, test_acc = train_one_epoch(
                epoch,
                model,
                train_lossFunc,
                lossFunction,
                optimizer,
                device,
                trainloader,
                testloader,
                tf_writer,
                log_training,
            )
            lr_scheduler.step()
            if np.isnan(train_loss) or np.isinf(train_loss):
                print("train loss is nan or inf !!! stop training!!!")
                break

            # save model
            if test_acc >= best_acc:
                best_acc = test_acc
                best_acc_loss = test_loss
                print(
                    "===> Epoch: {0}, Best Accuracy: {1:.2%}, Loss: {2:.4f}, SAVE MODEL!".format(
                        epoch + 1, best_acc, best_acc_loss
                    )
                )

                torch.save(
                    {
                        "model_params": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(model_save_folder, f"bestacc_{fold_name}.pth"),
                )
                log["test"][
                    fold_name
                ] = "[Epoch] {0:0>2d} [Accuracy] {1:.2%} [Loss] {2:.4f}".format(
                    epoch + 1, best_acc, best_acc_loss
                )

                if args.evaluate:
                    metrics = validate(model, device, validateloader, True)
                    log["validate"][fold_name] = metrics

        save_json(f"{checkpoint_dir}/log.json", log)
        print(f"{fold_name} Done.")
    print("Done.")


if __name__ == "__main__":
    main()
