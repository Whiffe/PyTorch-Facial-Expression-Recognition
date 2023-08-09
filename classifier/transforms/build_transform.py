# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-08-02 14:33:33
"""
import numbers
import random
import PIL.Image as Image
import numpy as np
from torchvision import transforms


def image_transform(input_size, rgb_mean=[0.5, 0.5, 0.5], rgb_std=[0.5, 0.5, 0.5], trans_type="train"):
    """
    不推荐使用：RandomResizedCrop(input_size), # bug:目标容易被crop掉
    :param input_size: [w,h]
    :param rgb_mean:
    :param rgb_std:
    :param trans_type:
    :return::
    """
    if trans_type == "train":
        transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[1] / 112), int(128 * input_size[0] / 112)]),
            transforms.RandomHorizontalFlip(),  # 随机左右翻转
            # transforms.RandomVerticalFlip(), # 随机上下翻转
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.RandomCrop([input_size[1], input_size[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ])
    elif trans_type == "val" or trans_type == "test":
        transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[1] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[1], input_size[0]]),
            # transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ])
    else:
        raise Exception("transform_type ERROR:{}".format(trans_type))
    return transform
