from glob import glob
import cv2
import os
import gc
import shutil
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import path

GAUSSIAN_BLUR_TOPOGRAPHIC_MAP = False


def split_label_mask_train(input_dir, save_dir, split):
    # 1~64のsurface_volumeをstackし、npy（float32）で保存
    # fragmentサイズが大きい場合はsplit分割
    fragment_i = input_dir.split("/")[-1]
    for split_i in range(split):
        mask=cv2.imread(f"{input_dir}/mask.png", 0)
        label=cv2.imread(f"{input_dir}/inklabels.png", 0)
        image_height = mask.shape[0]
        split_height = image_height // split
        if split_i == split - 1:
            mask = mask[split_i*split_height:image_height, :]
            label = label[split_i*split_height:image_height, :]
        else:
            mask = mask[split_i*split_height:(split_i+1)*split_height, :]
            label = label[split_i*split_height:(split_i+1)*split_height, :]
        cv2.imwrite(save_dir + f"mask_{fragment_i}_{split_i}.png", mask)
        cv2.imwrite(save_dir + f"inklabels_{fragment_i}_{split_i}.png", label)


def split_label_mask_inference(input_dir, save_dir, fragment_i):
    shutil.copy(f"{input_dir}/mask.png", save_dir + f"mask_{fragment_i}.png")


def split_stack_image(input_dir, save_dir, split=1):
    fragment_i = input_dir.split("/")[-1]
    image_height = cv2.imread(f"{input_dir}/mask.png", -1).shape[0]
    split_height = image_height // split

    for split_i in range(split):
        image_stack = None
        images = []

        surfaces_path = sorted(glob(f"{input_dir}/surface_volume/*.tif"))
        save_npy_path = save_dir + f"image_stack_{fragment_i}_{split_i}.npy"
        if os.path.exists(save_npy_path):
            continue

        for surface_path in tqdm(surfaces_path):
            image = cv2.imread(surface_path, 0)
            if split_i < split - 1:
                image = image[split_i*split_height:(split_i+1)*split_height, :]
            else:
                image = image[split_i*split_height:image_height, :]
            images.append(image)
            del image
        image_stack = np.stack(images)

        del images
        gc.collect()

        image_stack = (image_stack).astype(np.uint8)
        with open(save_npy_path, 'wb') as f:
            np.save(f, image_stack, allow_pickle=True)
        del image_stack
        gc.collect()


def flatten(image_stack, x, y, range, z_buffer):
    clipped_stack = image_stack[:, x:x+range, y:y+range]  # smaller portion
    clipped_stack=(clipped_stack/255)
    clipped_stack=np.flip(clipped_stack,axis=0)
    gauss_stack = gaussian_filter(clipped_stack, sigma=1)  # blur data a little bit
    gauss_stack = ndimage.sobel(gauss_stack, axis=0)  # detect edges in top-down direction
    gauss_stack = gaussian_filter(gauss_stack, sigma=1)  # blur again

    filtered_stack = np.where(gauss_stack >= 0.5, 1, 0)  # type: ignore
    topographic_map = np.argmax(filtered_stack, axis=0)
    if GAUSSIAN_BLUR_TOPOGRAPHIC_MAP:
        topographic_map = gaussian_filter(topographic_map, sigma=1)

    is_idx = np.indices(clipped_stack.shape)
    flatten_stack = clipped_stack[(is_idx[0] + topographic_map-z_buffer) % clipped_stack.shape[0], is_idx[1], is_idx[2]]
    flatten_stack=(np.flip(flatten_stack,axis=0)*255).astype("uint8")

    # return clipped_stack, gauss_stack, filtered_stack, topographic_map, flatten_stack
    return flatten_stack


def whole_flatten(dataset_dir, image_stack_dir, fragment_i, split_i, delete=False):
    save_path = os.path.join(image_stack_dir + f"flatten_stack_{fragment_i}_{split_i}.npy")
    image_stack_path = dataset_dir + f"image_stack_{fragment_i}_{split_i}.npy"

    if os.path.exists(save_path):
        return

    image_stack = np.load(open(image_stack_path, 'rb'))

    _, image_stack_x, image_stack_y = image_stack.shape
    flatten_array = np.zeros_like(image_stack)

    for x in range(0, image_stack_x, 500):
        for y in range(0, image_stack_y, 500):
            flattened_stack = flatten(image_stack, x, y, 500, 5)
            flatten_array[:, x:x+500, y:y+500] = flattened_stack
            del flattened_stack
    with open(save_path, 'wb') as f:
        np.save(f, flatten_array, allow_pickle=True)

    del flatten_array
    gc.collect()

    if delete:
        os.remove(os.path.join(image_stack_path))


def extract_flatten_layers(input_dir, save_dir, fragment_i, split_i, start, stop, delete=False):
    input_stack_path = f"{input_dir}/flatten_stack_{fragment_i}_{split_i}.npy"
    output_stack_path=f"{save_dir}/{fragment_i}_{split_i}.npy"

    stack = np.load(open(input_stack_path, 'rb'))
    stack = stack[-stop-1:-start, :, :]

    with open(output_stack_path, 'wb') as f:
        np.save(f, stack, allow_pickle=True)
    del stack
    gc.collect()

    if delete:
        os.remove(os.path.join(input_stack_path))

def extract_nonflatten_layers(input_dir, save_dir, fragment_i, split_i, start, stop, delete=False):
    input_stack_path = f"{input_dir}/image_stack_{fragment_i}_{split_i}.npy"
    output_stack_path=f"{save_dir}/{fragment_i}_{split_i}.npy"

    stack = np.load(open(input_stack_path, 'rb'))
    stack = stack[start:stop+1, :, :]

    with open(output_stack_path, 'wb') as f:
        np.save(f, stack, allow_pickle=True)
    del stack
    gc.collect()

    if delete:
        os.remove(os.path.join(input_stack_path))


def concat_npy(save_dir, start, stop, fragment_i, delete=False):
    npy_list = []
    for npy in sorted(glob(f"{save_dir}/{start}-{stop}/{fragment_i}_*.npy")):
        print(npy)
        npy_list.append(np.load(open(npy, 'rb')))
        if delete:
            os.remove(npy)
    result = np.concatenate(npy_list, axis=1)
    output_stack_fname = f"{fragment_i}.npy"
    with open(f"{save_dir}/{start}-{stop}/{output_stack_fname}", 'wb') as f:
        np.save(f, result, allow_pickle=True)


def dataset_preprocess_nonflatten(input_dir, dataset_dir, split, start, stop, train=True, delete=True):
    image_stack_dir = f"{dataset_dir}/nonflatten/"
    extract_save_dir = f"{image_stack_dir}/{start}-{stop}/"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(image_stack_dir, exist_ok=True)
    os.makedirs(extract_save_dir, exist_ok=True)

    fragment_i = input_dir.split("/")[-1]
    print(input_dir)
    if train:
        split_label_mask_train(input_dir, dataset_dir, split)
    else:
        split_label_mask_inference(input_dir, dataset_dir, fragment_i)

    split_stack_image(input_dir, dataset_dir, split)

    for split_i in range(split):
        extract_nonflatten_layers(dataset_dir, extract_save_dir, fragment_i, split_i, start, stop, delete)

    if not train:
        concat_npy(extract_save_dir, start, stop, fragment_i, delete)



def dataset_preprocess_flatten(input_dir, dataset_dir, split, start, stop, train=True, delete=True):
    image_stack_dir = f"{dataset_dir}/flatten/"
    extract_save_dir = f"{image_stack_dir}/{start}-{stop}/"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(image_stack_dir, exist_ok=True)
    os.makedirs(extract_save_dir, exist_ok=True)

    fragment_i = input_dir.split("/")[-1]
    print(input_dir)
    if train:
        split_label_mask_train(input_dir, dataset_dir, split)
    else:
        split_label_mask_inference(input_dir, dataset_dir, fragment_i)

    split_stack_image(input_dir, dataset_dir, split)

    for split_i in range(split):
        whole_flatten(dataset_dir, image_stack_dir, fragment_i, split_i, delete=False)
        extract_flatten_layers(image_stack_dir, extract_save_dir, fragment_i, split_i, start, stop, delete)

    if not train:
        concat_npy(extract_save_dir, start, stop, fragment_i, delete)
