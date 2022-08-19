import argparse
import os
from typing import List

import cv2
import numpy as np
from sklearn.manifold import TSNE
from utils.dataset import ThreePhaseBone, normalize
from matplotlib import pyplot as plt
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    XGradCAM,
)
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from utils.model import create_model, load_model, trained_model_output
from utils.utils import (
    calculate_benefit_treat_all,
    calculate_net_benefit,
    get_convlstm_tensor,
    load_json,
    mkdir,
    softmax,
)

###################
###  Constants  ###
###################
TITLE_FONT = {
    "fontsize": 24,
    "color": "black",
    "weight": "light",
}
LABEL_FONT = {
    "fontsize": 24,
    "color": "black",
    "weight": "light",
}
LEGEND_FONT = {
    "weight": "normal",
    "size": 15,
}
LINE_WIDTH = 2
MARKER_SIZE = 150
COLORS = [
    "#63b2ee",
    "#76da91",
    "#f8cb7f",
    "#f89588",
    "#7cd6cf",
    "#9192ab",
    "#efa666",
    "#9394E7",
    "#5F97D2",
    "#9DC3E7",
]


##################
###  Function  ###
##################


# Receiver Operating Characteristic curve
def plot_ROC(log: str):
    DOCTOR = {
        "Knee": {"1": [0.4375, 1.0], "2": [0.5625, 0.9333], "3": [0.5, 1.0],},
        "Hip": {"1": [0.8929, 0.8125], "2": [0.8571, 0.8750], "3": [0.8571, 0.8750],},
    }
    file_types = ["test", "validate"]
    roc_dir = os.path.dirname(log)
    # ground truth, predicted score, predicted label
    result = trained_model_output(log, file_types)
    validate_fpr = np.linspace(0, 1, 101)
    validate_tpr = []
    test_result = {}
    for i, r in enumerate(result):
        t, v = r
        fpr, tpr, thresholds = roc_curve(t[0], t[1][:, 1])
        test_auc = auc(fpr, tpr)
        test_result[f"fold {i+1}"] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": test_auc,
        }
        fpr, tpr, thresholds = roc_curve(v[0], v[1][:, 1])
        validate_tpr.append(np.interp(validate_fpr, fpr, tpr))
        validate_tpr[-1][0] = 0.0
    validate_tpr = np.mean(validate_tpr, axis=0)
    validate_tpr[-1] = 1.0
    validate_auc = auc(validate_fpr, validate_tpr)

    title = "Knee" if log.find("knee") != -1 else "Hip"
    doctor = DOCTOR[title]
    # plot
    plt.figure(figsize=(8, 8))
    plt.title(title, fontdict=TITLE_FONT)
    plt.plot([0, 1], [0, 1], color=COLORS[5], lw=2, linestyle="--")
    for (fold_name, roc_auc), line_color in zip(test_result.items(), COLORS[0:5]):
        plt.plot(
            roc_auc["fpr"],
            roc_auc["tpr"],
            color=line_color,
            lw=LINE_WIDTH,
            linestyle="dotted",
            label="{0}: auc={1:.3f}".format(fold_name, roc_auc["auc"]),
        )
    plt.plot(
        validate_fpr,
        validate_tpr,
        color=COLORS[6],
        lw=LINE_WIDTH,
        label="independent: auc={:.3f}".format(validate_auc),
    )
    plt.scatter(
        1 - doctor["1"][0],
        doctor["1"][1],
        marker="o",
        label="Physician 1",
        c=COLORS[7],
        s=MARKER_SIZE,
    )
    plt.scatter(
        1 - doctor["2"][0],
        doctor["2"][1],
        marker="X",
        label="Physician 2",
        c=COLORS[8],
        s=MARKER_SIZE,
    )
    plt.scatter(
        1 - doctor["3"][0],
        doctor["3"][1],
        marker="P",
        label="Physician 3",
        c=COLORS[9],
        s=MARKER_SIZE,
    )
    plt.axis("square")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("1-Specificity", fontdict=LABEL_FONT)
    plt.ylabel("Sensitivity", fontdict=LABEL_FONT)
    plt.xticks(fontsize=LABEL_FONT["fontsize"])
    plt.yticks(fontsize=LABEL_FONT["fontsize"])
    plt.legend(loc="lower right", prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(roc_dir, "ROC"))
    plt.close()


# Decision Analysis Curve
def plot_DAC(log: str):
    file_types = ["test", "validate"]
    dac_dir = os.path.dirname(log)
    # ground truth, predicted score, predicted label
    result = trained_model_output(log, file_types)
    thresholds = np.arange(0, 1, 0.001)
    t_benefit = []
    v_benefit = []
    for r in result:
        t, v = r
        pred_score = softmax(t[1])
        t_benefit.append(calculate_net_benefit(thresholds, pred_score[:, 1], t[0]))
        pred_score = softmax(v[1])
        v_benefit.append(calculate_net_benefit(thresholds, pred_score[:, 1], v[0]))
    v_benefit_all = calculate_benefit_treat_all(thresholds, result[0][1][0])
    t_mean_benefit = np.mean(t_benefit, axis=0)
    v_benefit = np.mean(v_benefit, axis=0)

    title = "Knee" if log.find("knee") != -1 else "Hip"
    plt.figure(figsize=(10, 8))
    plt.title(title, fontdict=TITLE_FONT)
    for i, (benefit, line_color) in enumerate(zip(t_benefit, COLORS[0:5])):
        plt.plot(
            thresholds,
            benefit,
            color=line_color,
            lw=LINE_WIDTH,
            linestyle="dotted",
            label=f"fold {i+1}",
        )
    plt.plot(
        thresholds, t_mean_benefit, color=COLORS[7], lw=LINE_WIDTH, label="mean",
    )
    plt.plot(
        thresholds, v_benefit, color=COLORS[6], lw=LINE_WIDTH, label="independent",
    )
    plt.plot(
        thresholds, v_benefit_all, color=COLORS[9], lw=LINE_WIDTH, label="treat all",
    )
    plt.plot(
        (0, 1),
        (0, 0),
        color=COLORS[5],
        linestyle="--",
        lw=LINE_WIDTH,
        label="treat none",
    )
    plt.xlim(0, 1)
    plt.ylim(np.min(t_benefit) - 0.1, np.max(t_benefit) + 0.15)
    plt.xlabel("Threshold Probability", fontdict=LABEL_FONT)
    plt.ylabel("Net Benefit", fontdict=LABEL_FONT)
    plt.xticks(fontsize=LABEL_FONT["fontsize"])
    plt.yticks(fontsize=LABEL_FONT["fontsize"])
    plt.legend(loc="best", prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(dac_dir, "DAC"))
    plt.close()


# Calibration Curves
def plot_CC(log: str, file_types: List[str], n_bins: int, strategy: str, total: bool):
    cc_dir = os.path.dirname(log)
    print("compute trained model's output.")
    # ground truth, predicted score, predicted label
    result = trained_model_output(log, file_types)
    scores, truths, curves = [], [], []
    for r in result:
        score = np.concatenate([softmax(_[1]) for _ in r])
        truth = np.concatenate([_[0] for _ in r])
        if total:
            scores.append(score)
            truths.append(truth)
        else:
            curve = calibration_curve(truth, score[:, 1], n_bins=n_bins)
            curves.append(curve)
    if total:
        cc = calibration_curve(
            np.concatenate(truths), np.concatenate(scores)[:, 1], n_bins=n_bins
        )
    title = "Knee" if log.find("knee") != -1 else "Hip"
    print("draw Calibration Curves.")
    plt.figure(figsize=(8, 8))
    plt.title(title, fontdict=TITLE_FONT)
    plt.plot([0, 1], [0, 1], color=COLORS[5], lw=2, linestyle="--", label="")
    if total:
        plt.plot(cc[1], cc[0], color=COLORS[6], lw=LINE_WIDTH)
        plt.scatter(cc[1], cc[0], c=COLORS[6], marker="x")
    else:
        for i, (c, line_color) in enumerate(zip(curves, COLORS[0:5])):
            plt.plot(
                c[1], c[0], color=line_color, lw=LINE_WIDTH, label=f"fold {i+1}",
            )
            plt.scatter(c[1], c[0], c=line_color, marker="x")
    plt.xlabel("Predicted risk", fontdict=LABEL_FONT)
    plt.ylabel("Obeserved risk", fontdict=LABEL_FONT)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=LABEL_FONT["fontsize"])
    plt.yticks(fontsize=LABEL_FONT["fontsize"])
    # plt.legend(loc="best", prop=LEGEND_FONT)
    plt.tight_layout()
    name = "_".join(file_types)
    plt.savefig(os.path.join(cc_dir, f"CC_{strategy}_{n_bins}_{name}_{total}"))
    plt.close()
    print("Done.")


# CAM
def draw_cam(log: str, fold_name: str, method: str):
    methods = {
        "gradcam": GradCAM,  # ok
        "gradcam++": GradCAMPlusPlus,  # ok
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,  # ok
        "eigencam": EigenCAM,  # ok
        "eigengradcam": EigenGradCAM,  # ok
        "layercam": LayerCAM,  # ok
    }

    print("load arguments and model.")
    checkpoint_dir = os.path.dirname(log)
    log = load_json(log)
    arguments = argparse.Namespace(**log["arguments"])
    cam_save_dir = os.path.join(checkpoint_dir, "CAM_")
    mkdir(cam_save_dir)
    fusion = arguments.data_type == "none"
    model = create_model(arguments)
    model = load_model(
        model, checkpoint_dir + f"/model/bestacc_{fold_name}.pth", fusion
    )
    files = load_json(arguments.data_file)[fold_name]

    filelist = files["validate"]
    dataloader = DataLoader(
        dataset=ThreePhaseBone(
            filelist, arguments.data_type, arguments.dicom_window_ratio, is_train=False,
        )
    )

    print(f"choose cam method: {method}.")
    cam_method = methods[method]
    if fusion:
        target_layers = [model.convlstm1, model.convlstm2]
    else:
        target_layers = [model.convlstm]
    cams = [
        cam_method(
            model=model,
            target_layers=[target_layer],
            use_cuda=False,
            get_tensor=get_convlstm_tensor,
        )
        for target_layer in target_layers
    ]
    for i in range(len(cams)):
        cams[i].batch_size = arguments.batch_size

    print("start draw cam picture.")
    for idx, (input_tensor, targets) in enumerate(dataloader):
        print(f"counting: {idx+1}.")
        input_images = input_tensor.cpu().numpy()[0, 0]
        targets = [ClassifierOutputTarget(target) for target in targets.cpu().numpy()]
        grayscale_cams = [
            cam(
                input_tensor=input_tensor,
                targets=targets,
                aug_smooth=False,
                eigen_smooth=False,
            )
            for cam in cams
        ]
        heat_maps = []
        for grayscale_cam in grayscale_cams:
            heat_map = cv2.applyColorMap(
                np.uint8(255 * np.squeeze(grayscale_cam)), cv2.COLORMAP_JET
            )
            heat_map = np.float32(heat_map) / 255
            heat_maps.append(heat_map)

        # fuse heatmap and images
        show_cam_on_hip = arguments.dataset.find("hip") != -1
        if show_cam_on_hip:
            hips = np.load(filelist[idx])
            hip_images = hips["data"]
            hip_boundary = hips["boundary"]
            hip_images = normalize(hip_images)
            if input_images.shape[0] == 20:
                hip_images = hip_images[00:20]
            if input_images.shape[0] == 5:
                hip_images = hip_images[20:25]
            if fusion:
                hip_images = [hip_images[19], hip_images[23]]
            else:
                hip_images = [hip_images[-1]]

        if fusion:
            input_images = [input_images[19], input_images[23]]
        else:
            input_images = [input_images[-1]]

        for i, (input_image, heat_map) in enumerate(zip(input_images, heat_maps)):

            if show_cam_on_hip:
                hip_image = 1 - hip_images[i]
                hip_image = hip_image[:, :, np.newaxis]
                hip_image = np.repeat(hip_image, repeats=3, axis=2)
                cv2.imwrite(
                    os.path.join(cam_save_dir, f"{fold_name}_{idx}_{i}_hip.jpg"),
                    np.uint8(hip_image * 255),
                )
                cam_image = (
                    heat_map
                    + hip_image[
                        hip_boundary[0] : hip_boundary[1] + 1,
                        hip_boundary[2] : hip_boundary[3] + 1,
                        :,
                    ]
                )
                cam_image = cam_image / np.max(cam_image)
                hip_image[
                    hip_boundary[0] : hip_boundary[1] + 1,
                    hip_boundary[2] : hip_boundary[3] + 1,
                    :,
                ] = cam_image
                cv2.imwrite(
                    os.path.join(
                        cam_save_dir, f"{fold_name}_{idx}_{i}_hip_{method}.jpg"
                    ),
                    hip_image * 255,
                )

                # Normalize input image
                input_image -= np.min(input_image)
                input_image /= np.max(input_image) - np.min(input_image)
            else:
                input_image = 1 - input_image

            input_image = input_image[:, :, np.newaxis]
            input_image = np.repeat(input_image, repeats=3, axis=2)
            cam_image = input_image + heat_map
            cam_image = cam_image / np.max(cam_image)
            cv2.imwrite(
                os.path.join(cam_save_dir, f"{fold_name}_{idx}_{i}.jpg"),
                np.uint8(input_image * 255),
            )
            cv2.imwrite(
                os.path.join(cam_save_dir, f"{fold_name}_{idx}_{i}_{method}.jpg"),
                np.uint8(cam_image * 255),
            )
    print("Done.")


# t-SNE
def plot_t_sne(log: str, seed: int):
    # f2c205, #ff772c, #ff0863, #bf0099, #002ebb);
    # colors = ["#63B2EE", "#7898E1", "#EFA666", "#F8CB7F"]
    # colors = ["#63B2EE", "#63B2EE", "#EFA666", "#EFA666"]
    colors = ["#00F5FF", "#00F5FF", "#FFF68F", "#FFF68F"]
    markers = ["o", "^", "o", "^"]
    classes = [
        "test non-infected",
        "indepentent non-infected",
        "test infected",
        "indepentent infected",
    ]
    file_types = ["test", "validate"]
    checkpoint_dir = os.path.dirname(log)
    log = load_json(log)
    arguments = argparse.Namespace(**log["arguments"])
    model = create_model(args=arguments)
    data_file = load_json(arguments.data_file)
    fusion = arguments.data_type == "none"
    for fold_name, files in data_file.items():
        model = load_model(
            model, checkpoint_dir + f"/model/bestacc_{fold_name}.pth", fusion
        )
        feature_vectors = []
        marker_colors = []
        markers_ = []
        for i, type in enumerate(file_types):
            dataloader = DataLoader(
                dataset=ThreePhaseBone(
                    files[type],
                    arguments.data_type,
                    arguments.dicom_window_ratio,
                    is_train=False,
                )
            )
            feature_activations = ActivationsAndGradients(
                model, [model.classifier.act1], None, None
            )
            for inputs, targets in dataloader:
                feature_activations(inputs)
                feature_vectors.append(feature_activations.activations[0])
                marker_colors.append(colors[2 * targets + i])
                markers_.append(markers[2 * targets + i])
        feature_vectors = np.concatenate(feature_vectors, axis=0)
        features_vector_sne = TSNE(2, random_state=seed).fit_transform(feature_vectors)
        plt.style.use("dark_background")
        plt.figure(figsize=(4, 4))
        for i in range(len(marker_colors)):
            plt.scatter(
                features_vector_sne[i, 0],
                features_vector_sne[i, 1],
                c=marker_colors[i],
                marker=markers_[i],
                alpha=0.6,
            )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(checkpoint_dir + f"/t-sne_{fold_name}_{seed}__")

