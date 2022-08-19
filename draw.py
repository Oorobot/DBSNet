from sklearn.metrics import auc, classification_report, roc_curve
from utils.model import trained_model_output
from utils.plot import draw_cam, plot_CC, plot_DAC, plot_ROC, plot_t_sne
from utils.utils import calculate_metrics
import numpy as np

knee_log = "checkpoint_after0501/knee/none_ccm_fa1/2022-05-10_15:11:24/log.json"
hip_log = "checkpoint_after0501/hip_focus/none_tpb0/2022-06-08_22:57:00_20Rot+HorizontalFlip_Best/log.json"

# plot_ROC(hip_log)
# plot_ROC(knee_log)

# plot_DAC(hip_log)
# plot_DAC(knee_log)

# plot_t_sne(hip_log, 99)
# plot_t_sne(knee_log, 333)

# plot_CC(hip_log, ["train", "test", "validate"], 10, "quantile", True)
# plot_CC(hip_log, ["train", "test", "validate"], 10, "quantile", False)
# plot_CC(knee_log, ["train", "test", "validate"], 10, "quantile", True)
# plot_CC(knee_log, ["train", "test", "validate"], 10, "quantile", False)

# for fold_name in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:
#   draw_cam(hip_log, fold_name, "gradcam")
#   draw_cam(knee_log, fold_name, "gradcam")


# logs = [
#     "checkpoint/220801/hip_focus/none_vgg11_3d/2022-08-11_09:37:14/log.json",
#     "checkpoint/220801/hip_focus/none_vgg19_3d/2022-08-11_09:37:09/log.json",
#     "checkpoint/220801/hip_focus/none_resnet18_3d/2022-08-11_13:07:40/log.json",
#     "checkpoint/220801/hip_focus/none_resnet50_3d/2022-08-11_13:08:23/log.json",
#     "checkpoint/220801/hip_focus/none_densenet121_3d/2022-08-11_18:08:30/log.json",
#     "checkpoint/220801/hip_focus/none_convnext_tiny_3d/2022-08-11_13:09:10/log.json",
#     "checkpoint/220801/hip_focus/none_convnext_base_3d/2022-08-11_13:09:22/log.json",
#     #
#     "checkpoint_after0501/knee/none_vgg11_3d/2022-05-12 18:13:22/log.json",
#     "checkpoint_after0501/knee/none_vgg19_3d/2022-05-12 18:13:29/log.json",
#     "checkpoint_after0501/knee/none_resnet18_3d/2022-05-12 10:04:32/log.json",
#     "checkpoint_after0501/knee/none_resnet50_3d/2022-05-12 11:21:20/log.json",
#     "checkpoint_after0501/knee/none_densenet121_3d/2022-05-10 16:55:07/log.json",
#     "checkpoint_after0501/knee/none_convnext_tiny_3d/2022-05-18 14:33:36/log.json",
#     "checkpoint_after0501/knee/none_convnext_base_3d/2022-05-18 14:33:47/log.json",
# ]

# for log in logs:
#     result = trained_model_output(log, ["test"])
#     average_metric = {
#         "Acc": 0.0,
#         "Spec": 0.0,
#         "Sen": 0.0,
#         "F1": 0.0,
#         "PPV": 0.0,
#         "NPV": 0.0,
#         "AUC": 0.0,
#     }
#     for i, r in enumerate(result):
#         ground_truth, predicted_score, predicted_label = r[0]
#         classification_result = calculate_metrics(ground_truth, predicted_label)
#         fpr, tpr, thresholds = roc_curve(ground_truth, predicted_score[:, 1])
#         test_auc = auc(fpr, tpr)
#         classification_result["AUC"] = test_auc

#         for key in average_metric:
#             average_metric[key] += classification_result[key]
#     for key, value in average_metric.items():
#         average_metric[key] /= 5
#     print(log)
#     col_names = ["Acc", "Spec", "Sen", "F1", "PPV", "NPV", "AUC"]
#     metrics_infomation = (
#         "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}\n".format(*col_names)
#         + "{0:>8.2%}".format(average_metric["Acc"])
#         + "{0:>8.2%}".format(average_metric["Spec"])
#         + "{0:>8.2%}".format(average_metric["Sen"])
#         + "{0:>8.4f}".format(average_metric["F1"])
#         + "{0:>8.2%}".format(average_metric["PPV"])
#         + "{0:>8.2%}".format(average_metric["NPV"])
#         + "{0:>8.4f}".format(average_metric["AUC"])
#     )
#     print(metrics_infomation)


# calculate the 95% confidence interval，bootstrap
# bootstrap confidence intervals

from numpy.random import seed

from numpy.random import rand

from numpy.random import randint

from numpy import mean

from numpy import median

from numpy import percentile

# seed the random number generator

seed(1)

# generate dataset

dataset = trained_model_output(hip_log, ["test", "validate"])

five_fold_gt = np.concatenate([d[0][0] for d in dataset])
five_fold_score = np.concatenate([d[0][1] for d in dataset])
five_fold_label = np.concatenate([d[0][2] for d in dataset])
f_dataset_size = len(five_fold_gt)

independent_gt = np.concatenate([d[1][0] for d in dataset])
independent_score = np.concatenate([d[1][1] for d in dataset])
independent_label = np.concatenate([d[1][2] for d in dataset])
i_dataset_size = len(independent_gt)


def bootstrap(iterations: int, sample_size: int, gt, score, label):
    # bootstrap
    acc = list()
    sen = list()
    spec = list()
    f1 = list()
    ppv = list()
    npv = list()
    auc_ = list()

    for _ in range(iterations):

        # bootstrap sample

        indices = randint(0, sample_size, sample_size)

        sample_gt = gt[indices]
        sample_score = score[indices]
        sample_label = label[indices]

        classification = calculate_metrics(sample_gt, sample_label)
        fpr, tpr, thresholds = roc_curve(sample_gt, sample_score[:, 1])
        roc_auc = auc(fpr, tpr)
        acc.append(classification["Acc"])
        spec.append(classification["Spec"])
        sen.append(classification["Sen"])
        f1.append(classification["F1"])
        ppv.append(classification["PPV"])
        npv.append(classification["NPV"])
        auc_.append(roc_auc)

    alpha = 5.0
    lower_p = alpha / 2.0
    upper_p = (100 - alpha) + (alpha / 2.0)

    print("[Acc] 50th percentile (median) = %.4f" % median(acc))
    lower = max(0.0, percentile(acc, lower_p))
    print("[Acc] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(acc, upper_p))
    print("[Acc] %.1fth percentile = %.4f" % (upper_p, upper))

    print("[Spec] 50th percentile (median) = %.4f" % median(spec))
    lower = max(0.0, percentile(spec, lower_p))
    print("[Spec] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(spec, upper_p))
    print("[Spec] %.1fth percentile = %.4f" % (upper_p, upper))

    print("[Sen] 50th percentile (median) = %.4f" % median(sen))
    lower = max(0.0, percentile(sen, lower_p))
    print("[Sen] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(sen, upper_p))
    print("[Sen] %.1fth percentile = %.4f" % (upper_p, upper))

    print("[F1] 50th percentile (median) = %.4f" % median(f1))
    lower = max(0.0, percentile(f1, lower_p))
    print("[F1] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(f1, upper_p))
    print("[F1] %.1fth percentile = %.4f" % (upper_p, upper))

    print("[PPV] 50th percentile (median) = %.4f" % median(ppv))
    lower = max(0.0, percentile(ppv, lower_p))
    print("[PPV] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(ppv, upper_p))
    print("[PPV] %.1fth percentile = %.4f" % (upper_p, upper))

    print("[NPV] 50th percentile (median) = %.4f" % median(npv))
    lower = max(0.0, percentile(npv, lower_p))
    print("[NPV] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(npv, upper_p))
    print("[NPV] %.1fth percentile = %.4f" % (upper_p, upper))

    print("[AUC] 50th percentile (median) = %.4f" % median(auc_))
    lower = max(0.0, percentile(auc_, lower_p))
    print("[AUC] %.1fth percentile = %.4f" % (lower_p, lower))
    upper = min(1.0, percentile(auc_, upper_p))
    print("[AUC] %.1fth percentile = %.4f" % (upper_p, upper))


print("Hip Five-Fold: ")
bootstrap(1000, f_dataset_size, five_fold_gt, five_fold_score, five_fold_label)
print("Hip Independent: ")
bootstrap(1000, i_dataset_size, independent_gt, independent_score, independent_label)
