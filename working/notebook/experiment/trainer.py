import time
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler # type: ignore
from torch.optim import AdamW
from loss import bce_loss
from model import build_model
from scheduler import get_scheduler
from dataset import generate_valid_label,get_train_valid_dataset, generate_dataloader
from scheduler import get_scheduler,scheduler_step
from metrics import calc_cv


def model_init(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    scheduler = get_scheduler(cfg, optimizer)

    loss_fn = bce_loss
    return model, optimizer, scheduler, loss_fn, device


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(train_loader, model, criterion, optimizer, device, cfg):
    model.train()

    scaler = GradScaler(enabled=cfg.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(cfg.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward() # type: ignore

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)# type: ignore

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_label_gt, cfg):
    label_pred = np.zeros(valid_label_gt.shape)
    label_count = np.zeros(valid_label_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole label
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        start_idx = step*cfg.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            label_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            label_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))

    label_pred /= label_count
    return losses.avg, label_pred




def train_and_evaluate(preprocess, cfg):
    for fold_i, fold in enumerate(glob(f"{cfg.train_dataset_path}/*")):
        fold = os.path.basename(fold).split(".")[0]
        valid_fragment_i = int(fold.split("_")[0])
        valid_split_i = int(fold.split("_")[1])
        print(f"fold: {fold_i}")

        train_images, train_labels, train_masks, valid_images, valid_labels, valid_xyxys = get_train_valid_dataset(valid_fragment_i, valid_split_i, cfg, preprocess)
        train_loader, valid_loader = generate_dataloader(train_images, train_labels,  valid_images, valid_labels, cfg)

        valid_label_gt = generate_valid_label(valid_fragment_i, valid_split_i, cfg)

        model, optimizer, scheduler, loss_fn, device = model_init(cfg)

        if cfg.metric_direction == 'minimize':
            best_score = np.inf
        elif cfg.metric_direction == 'maximize':
            best_score = -1

        best_loss = np.inf

        for epoch in range(cfg.epochs):

            start_time = time.time()

            # train
            avg_train_loss = train_fn(train_loader, model, loss_fn, optimizer, device, cfg)

            # eval
            avg_val_loss, label_pred = valid_fn(valid_loader, model, loss_fn, device, valid_xyxys, valid_label_gt, cfg)

            scheduler_step(scheduler, avg_val_loss, epoch)

            best_dice, best_th = calc_cv(valid_label_gt, label_pred)

            score = best_dice

            elapsed = time.time() - start_time

            print(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  avgScore: {score:.4f}  time: {elapsed:.0f}s')

            if cfg.metric_direction == 'minimize':
                update_best = score < best_score  # type: ignore
            elif cfg.metric_direction == 'maximize':
                update_best = score > best_score  # type: ignore

            if update_best:  # type: ignore
                best_loss = avg_val_loss
                best_score = score

                print(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f}  Best Score: {best_score:.4f} Model')

                model_path = f'./{cfg.exp_name}/{cfg.exp_name}_fold{fold_i}.pth'
                torch.save({'model': model.state_dict(),
                            'preds': label_pred},
                        model_path)
            print()
