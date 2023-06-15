from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import generate_valid_label


def fbeta_score(preds, targets, threshold, beta=0.5, smooth=1e-5):
    preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
    y_true_count = targets.sum()

    ctp = preds_t[targets == 1].sum()
    cfp = preds_t[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def calc_fbeta(label, label_pred):
    label = label.astype(int).flatten()
    label_pred = label_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(50, 50+1, 5)) / 100:

        dice = fbeta_numpy(label, (label_pred >= th).astype(int), beta=0.5)

        if dice > best_dice:
            best_dice = dice
            best_th = th

    print(f'fbeta: {best_dice:0.4f}')
    return best_dice, best_th


def inference_evaluation(cfg, label_prefix="inklabels"):
    gt_area_list = []
    dice_list = []
    valid_label_gt_list = []
    label_pred_list = []

    for fold_i, fold in enumerate(glob(f"{cfg.train_dataset_path}/*")):
        fold = os.path.basename(fold).split(".")[0]
        valid_fragment_i = int(fold.split("_")[0])
        valid_split_i = int(fold.split("_")[1])

        if not os.path.exists(f"./{cfg.exp_name}/{cfg.exp_name}_fold{fold_i}.pth"):
            continue
        check_point = torch.load(f"./{cfg.exp_name}/{cfg.exp_name}_fold{fold_i}.pth", map_location=torch.device('cpu'))

        label_pred = check_point['preds']
        valid_label_gt = generate_valid_label(valid_fragment_i, valid_split_i, cfg, label_prefix)
        best_dice, best_th = calc_fbeta(valid_label_gt, label_pred)

        valid_label_gt_list.append(valid_label_gt)
        label_pred_list.append(label_pred)
        gt_area_list.append(valid_label_gt.shape[0]*valid_label_gt.shape[1])
        dice_list.append(best_dice)
    return gt_area_list, dice_list, valid_label_gt_list, label_pred_list


def dice_evaluation(gt_area_list, dice_list):
    dice_ = 0
    for gt_area, dice in zip(gt_area_list, dice_list):  # type: ignore
        dice_ += gt_area*dice

    dice_ = dice_/sum(gt_area_list)
    return dice_


def plot_inference(valid_label_gt_list, label_pred_list, cfg):

    fig, axes = plt.subplots(5, 3, figsize=(25, 20))
    for fragment_i, (valid_label_gt, label_pred) in enumerate(zip(valid_label_gt_list, label_pred_list)):

        axes[fragment_i][0].imshow(valid_label_gt)
        axes[fragment_i][0].axis('off')
        axes[fragment_i][1].imshow(label_pred)
        axes[fragment_i][1].axis('off')
        axes[fragment_i][2].imshow((label_pred >= 0.5).astype(int))
        axes[fragment_i][2].axis('off')

    plt.tight_layout()
    plt.savefig(f'/kaggle/working/notebook/experiment/stacked_unet/result/{cfg.exp_name}.png')
