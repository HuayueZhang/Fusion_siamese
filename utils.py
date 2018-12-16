# -*- coding: utf-8 -*

import numpy as np
from os import listdir, mkdir, sep, remove
from os.path import join, exists, splitext, split
from scipy.misc import imread, imsave, imresize
import tensorflow as tf
from PIL import Image, ImageStat

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2

PATCH_PER_IMAGE = 20
BLUR_LEVEL = 5
PATCH_VAR_TH = 25


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()  # 转换字符串中所有大写字符为小写
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
    return images


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='L')
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')
        images.append(image)

    images = np.stack(images, axis=0)  # shape = (number_images, height, width, channel)
    # print('images shape gen:', images.shape)
    return images

## ----------------------------------get training images by preprocessing--------------------------------------------

def save_images(p, path, b, patch, mode, folder):
    if isinstance(path, str):
        paths = [path]

    pathname, ext = splitext(path)
    path1, imgname = split(pathname)

    imgname = imgname + '_' + str(p) + '_' + str(b) + ext

    rootpath, _ = split(path1)
    savepath = join(rootpath, mode, folder, imgname)

    imsave(savepath, patch)


def sample_patch(image):
    crop_heigh = 16
    crop_width = 16
    h, w = image.shape
    x = np.random.randint(0, w-crop_width-1, 1)
    y = np.random.randint(0, h-crop_heigh-1, 1)
    return image[int(y):int(y+crop_heigh), int(x):int(x+crop_width)]


def blur_patch(patch):
    return cv2.GaussianBlur(patch, (7, 7), 2)

def pre_op_get_train_images(paths):
    if isinstance(paths, str):
        paths = [paths]

    np.random.shuffle(paths)

    for path in paths[10000:]:
        remove(path)

    for i, path in enumerate(paths[0:8000]):
        print(i)
        image = imread(path, mode='L')

        p = 1
        while p < PATCH_PER_IMAGE+1:
            patch = sample_patch(image)
            if np.var(patch) < PATCH_VAR_TH:
                continue
            save_images(p, path, 0, patch, 'train', 'clear_x1')

            for b in range(BLUR_LEVEL):
                patch = blur_patch(patch)
                save_images(p, path, b+1, patch, 'train', 'blurd_x5')

            p = p + 1


    for i, path in enumerate(paths[8000:10000]):
        print(i)
        image = imread(path, mode='L')

        p = 1
        while p < PATCH_PER_IMAGE+1:
            patch = sample_patch(image)
            if np.var(patch) < PATCH_VAR_TH:
                continue
            save_images(p, path, 0, patch, 'eval', 'clear_x1')

            for b in range(BLUR_LEVEL):
                patch = blur_patch(patch)
                save_images(p, path, b+1, patch, 'eval', 'blurd_x5')

            p = p + 1


def main(directory):
    paths = list_images(directory)
    pre_op_get_train_images(paths)
    # i = get_images(paths[0:10])


if __name__ == '__main__':
    main('/home/zhy/fuse_cnn/ILSVRC2012/origin_images/')