import random
import gc
from glob import glob
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import matplotlib.pyplot as plt


def read_image_label_mask(stack_path, cfg, preprocess):
    fragment_i = int(stack_path.split(".")[-2].split("/")[-1].split("_")[0])
    split_i = int(stack_path.split(".")[-2].split("/")[-1].split("_")[1])

    image_stack = np.load(open(stack_path, 'rb'))
    pad0 = (cfg.tile_size - image_stack.shape[1] % cfg.tile_size)
    pad1 = (cfg.tile_size - image_stack.shape[2] % cfg.tile_size)
    image_stack = np.pad(image_stack, [(0, 0), (0, pad0), (0, pad1)], constant_values=0)
    image_stack = image_stack.transpose((1, 2, 0))
    image_stack = preprocess(image_stack)

    label = cv2.imread(f"/kaggle/working/dataset_train/inklabels_{fragment_i}_{split_i}.png", 0)
    label = np.pad(label, [(0, pad0), (0, pad1)], constant_values=0)
    label = label.astype('float32')
    label /= 255.0

    mask = cv2.imread(f"/kaggle/working/dataset_train/mask_{fragment_i}_{split_i}.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32')
    mask /= 255.0

    return image_stack, label, mask


def get_train_valid_dataset(valid_fragment_i, valid_split_i, cfg, preprocess):
    train_images = []
    train_labels = []
    train_masks = []

    valid_images = []
    valid_labels = []
    valid_xyxys = []

    for stack_path in glob(f"{cfg.train_dataset_path}/*"):
        fragment_i = int(stack_path.split(".")[-2].split("/")[-1].split("_")[0])
        split_i = int(stack_path.split(".")[-2].split("/")[-1].split("_")[1])
        image, label, mask = read_image_label_mask(stack_path, cfg, preprocess)
        x1_list = list(range(0, image.shape[1]-cfg.tile_size+1, cfg.stride))
        y1_list = list(range(0, image.shape[0]-cfg.tile_size+1, cfg.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size

                if (fragment_i == valid_fragment_i) & (split_i == valid_split_i):
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_labels.append(label[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_labels.append(label[y1:y2, x1:x2, None])
                    train_masks.append(mask[y1:y2, x1:x2, None])
    valid_xyxys = np.stack(valid_xyxys)
    return train_images, train_labels, train_masks, valid_images, valid_labels, valid_xyxys


def visualize_train_images(id, train_images, train_labels, train_masks):
    plt.subplot(1, 2, 1)
    plt.imshow(train_labels[id], cmap="gray", vmin=0, vmax=1)
    plt.title("label")
    plt.subplot(1, 2, 2)
    plt.imshow(train_masks[id], cmap="gray", vmin=0, vmax=1)
    plt.title("mask")

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(train_images[id][:, :, i], cmap="gray")

    plt.figure(figsize=(10, 6))
    plt.hist(train_images[id].flatten(), bins=255)


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    else:
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] # type: ignore

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image, label


def visualize_dataset(plot_dataset, cfg):

    transform = cfg.train_aug_list
    transform = A.Compose(transform)

    while True:
        i = random.randint(1, len(plot_dataset))

        image, label = plot_dataset[i]
        data = transform(image=image, mask=label)
        aug_image = data['image']
        aug_label = data['mask']

        if label.sum() == 0:
            continue

        fig, axes = plt.subplots(1, 4, figsize=(15, 8))
        axes[0].imshow(image[..., 0], cmap="gray")
        axes[0].set_title("image")
        axes[1].imshow(label, cmap="gray")
        axes[1].set_title("label")
        axes[2].imshow(aug_image[0, :, :], cmap="gray")
        axes[2].set_title("aug_image")
        axes[3].imshow(aug_label[0, :, :], cmap="gray")
        axes[3].set_title("aug_label")
        plt.figure()
        plt.hist(aug_image.flatten(), bins=256)
        plt.title("aug_image")
        break

    del plot_dataset
    gc.collect()

def generate_dataloader(train_images, train_labels,  valid_images, valid_labels, cfg):
    train_dataset = CustomDataset(
        train_images, cfg, labels=train_labels, transform=get_transforms(data='train', cfg=cfg))

    valid_dataset = CustomDataset(
        valid_images, cfg, labels=valid_labels, transform=get_transforms(data='valid', cfg=cfg))

    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train_batch_size,
                            shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                            )
    valid_loader = DataLoader(valid_dataset,
                            batch_size=cfg.valid_batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return train_loader, valid_loader


def generate_valid_label(valid_fragment_i, valid_split_i, cfg):
    valid_label_gt = cv2.imread(f"{cfg.dataset_path}/inklabels_{valid_fragment_i}_{valid_split_i}.png", 0)
    valid_label_gt = (valid_label_gt/255)  # type: ignore
    pad0 = (cfg.tile_size - valid_label_gt.shape[0] % cfg.tile_size)
    pad1 = (cfg.tile_size - valid_label_gt.shape[1] % cfg.tile_size)
    valid_label_gt = np.pad(valid_label_gt, [(0, pad0), (0, pad1)], constant_values=0)
    return valid_label_gt