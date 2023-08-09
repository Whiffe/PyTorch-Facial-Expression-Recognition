# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-04-13 15:12:08
"""
import numpy as np
from classifier.dataloader import parser_imagefolder


def load_dataset(data_type,
                 filename,
                 transform,
                 class_name=None,
                 resample=False,
                 shuffle=True,
                 use_rgb=True,
                 check=False,
                 phase="train",
                 **kwargs):
    """
    :param data_type:
    :param filename:
    :param transform:
    :param resample:
    :param shuffle:
    :param check:
    :param phase:
    :param kwargs:
    :return:
    """
    if data_type.lower() == "folder":
        dataset = parser_imagefolder.ImageFolderDataset(filename,
                                                        transform=transform,
                                                        resample=resample,
                                                        class_name=class_name,
                                                        use_rgb=use_rgb,
                                                        shuffle=shuffle,
                                                        **kwargs)
    else:
        raise Exception("Error:data_type:{}".format(data_type))
    return dataset
