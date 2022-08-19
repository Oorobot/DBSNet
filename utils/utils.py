import os
import numpy as np
import json
from typing import List

from sklearn.metrics import confusion_matrix


# tool
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def mkdirs(dirs):
    if isinstance(dirs, list) and not isinstance(dirs, str):
        for dir in dirs:
            mkdir(dir)
    else:
        mkdir(dirs)


def save_json(save_path: str, data: dict):
    assert save_path.split(".")[-1] == "json"
    with open(save_path, "w") as file:
        json.dump(data, file)


def load_json(file_path: str) -> dict:
    assert file_path.split(".")[-1] == "json"
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def k_fold(files: List[str], K: int, num_classes: int, save_path: str = None):
    # Secondary classification: 0 -> non-infected, 1 -> infected
    # Triple classification: 0 -> normal, 1 -> non-infected after replacement surgery, 2 -> infected after replacement surgery
    samlpes = [[] for _ in range(num_classes)]

    for file in files:
        label = np.load(file)["label"]
        samlpes[label].append(file)
    print("the number of each class: {}.".format([len(s) for s in samlpes]))

    def avg_split_k_fold(K: int, file_list: List[str]):
        size = len(file_list) // K
        k_fold = [[] for _ in range(K)]
        for i in range(K):
            each_fold = np.random.choice(file_list, size, False)
            file_list = np.setdiff1d(file_list, each_fold)
            k_fold[i].extend(each_fold.tolist())
        for i in range(len(file_list)):
            k_fold[i].append(file_list[i])
        return k_fold

    samlpes_k_fold = [avg_split_k_fold(K + 1, s) for s in samlpes]
    validate_files = np.concatenate([s[-1] for s in samlpes_k_fold])
    standard_k_fold = {}
    for i in range(K):
        test_files = np.concatenate([s[i] for s in samlpes_k_fold])
        train_files = np.setdiff1d(files, np.concatenate((validate_files, test_files)))
        standard_k_fold[f"fold_{i+1}"] = {
            "train": train_files.tolist(),
            "test": test_files.tolist(),
            "validate": validate_files.tolist(),
        }

        print(
            "[fold {0}] train: {1}-{4}, test: {2}-{5}, validate: {3}-{6}.".format(
                i + 1,
                len(train_files),
                len(test_files),
                len(validate_files),
                [
                    len(s) - len(s_[i]) - len(s_[-1])
                    for s, s_ in zip(samlpes, samlpes_k_fold)
                ],
                [len(s[i]) for s in samlpes_k_fold],
                [len(s[-1]) for s in samlpes_k_fold],
            )
        )

    if save_path is not None:
        save_json(save_path, standard_k_fold)


# calculate classification metrics
def calculate_metrics(ground_truth, predicted_classes):
    binary = np.unique(ground_truth).shape[0] == 2
    if binary:
        tn, fp, fn, tp = confusion_matrix(ground_truth, predicted_classes).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        cm = np.reshape(np.array([tn, fp, fn, tp]), (2, 2))
    else:
        cm = confusion_matrix(ground_truth, predicted_classes)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        accuracy = fp.sum() / cm.sum()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    if binary:
        # two Categories
        col_names = ["Acc", "Spec", "Sen", "F1", "PPV", "NPV", "CM"]
        metrics_infomation = (
            "{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n".format(*col_names)
            + "{0:>8.2%}({1:>2}/{2:>2})".format(accuracy, tn + tp, tn + fp + fn + tp,)
            + "{0:>8.2%}({1:>2}/{2:>2})".format(specificity, tn, tn + fp,)
            + "{0:>8.2%}({1:>2}/{2:>2})".format(sensitivity, tp, tp + fn,)
            + "{:>15.4f}".format(f1)
            + "{0:>8.2%}({1:>2}/{2:>2})".format(ppv, tp, tp + fp)
            + "{0:>8.2%}({1:>2}/{2:>2})".format(npv, tn, tn + fn)
            + "{:>15}".format(str([tn, fp, fn, tp]))
        )
    else:
        # three Categories
        metrics_infomation = (
            "metrics\t{}\t{}\t{}\n".format(*np.arange(0, 3, 1))
            + "Acc\t\t\t{0:.2%}({1}/{2})\n".format(
                accuracy, tp.sum(), len(ground_truth)
            )
            + "Spec\t{0:.2%}({3}/{6})\t{1:.2%}({4}/{7})\t{2:.2%}({5}/{8})\n".format(
                *specificity, *tn, *(tn + fp)
            )
            + "Sen\t{0:.2%}({3}/{6})\t{1:.2%}({4}/{7})\t{2:.2%}({5}/{8})\n".format(
                *sensitivity, *tp, *(tp + fn)
            )
            + "F1\t{0:.4f}\t{1:.4f}\t{2:.4f}\n".format(*f1)
            + "PPV\t{0:.2%}({3}/{6})\t{1:.2%}({4}/{7})\t{2:.2%}({5}/{8})\n".format(
                *ppv, *tp, *(tp + fp)
            )
            + "NPV\t{0:.2%}({3}/{6})\t{1:.2%}({4}/{7})\t{2:.2%}({5}/{8})\n".format(
                *npv, *tn, *(tn + fn)
            )
            + f"CM:\n{str(cm)}"
        )
    # print(metrics_infomation)
    return {
        "Acc": accuracy,
        "Spec": specificity.tolist(),
        "Sen": sensitivity.tolist(),
        "F1": f1.tolist(),
        "PPV": ppv.tolist(),
        "NPV": npv.tolist(),
        "CM": cm.tolist(),
    }


def calculate_net_benefit(thresholds: List[float], predicted_score, labels):
    net_benefit = []
    for threshold in thresholds:
        predicted_label = predicted_score > threshold
        tn, fp, fn, tp = confusion_matrix(labels, predicted_label).ravel()
        n = tn + fp + fn + tp
        benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefit.append(benefit)
    return net_benefit


def calculate_benefit_treat_all(thresholds: List[float], labels):
    benefit_treat_all = []
    count = np.bincount(labels)
    ones, zeros, n = count[1], count[0], count[0] + count[1]
    for threshold in thresholds:
        benefit_ = (ones / n) - (zeros / n) * (threshold / (1 - threshold))
        benefit_treat_all.append(benefit_)
    return benefit_treat_all


# other
def get_convlstm_tensor(convlstm_output):
    _, last_state = convlstm_output
    return last_state[0][0]
