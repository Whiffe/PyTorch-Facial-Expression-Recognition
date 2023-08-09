# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import math
import PIL.Image as Image
import numpy as np
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pybaseutils import image_utils, file_utils
from classifier.dataloader import balanced_classes

IMG_POSTFIX = ['*.jpg', '*.jpeg', '*.png', '*.tif', "*.JPG", "*.bmp", "*.webp"]


class ImageFolderDataset(Dataset):
    """Pytorch Dataset"""

    def __init__(self, files, class_name=None, transform=None,
                 resample=False, use_rgb=True, shuffle=True, disp=False):
        """
        :param files: 图片数据根路径
        :param transform: torch transform
        :param resample: 是否使用重采样
        :param shuffle:
        :param disp:
        """
        self.resample = resample
        self.use_rgb = use_rgb
        self.class_name, self.class_dict = self.parser_classes(class_name)
        # 人脸识别中，多数据集合并训练only_id=False,格式：id/sub_dir
        # 其他情况，如果多个数据集的label文件是同一个类别，则only_id=True
        self.class_dict, self.item_list = self.parser_annotation(files, self.class_dict, shuffle, only_id=True)
        # 计算当前每个类别的分布
        self.class_count = self.count_class_nums(self.item_list)
        self.classes = list(self.class_count.keys())
        self.num_classes = max(self.classes) + 1
        print("Dataset have images:{},class_count:{}".format(len(self.item_list), self.class_count))
        print("class_dict: {}".format(self.class_dict))
        self.classes_weights = self.get_classes_weights(label_index=1)
        self.transform = transform
        self.num_images = len(self.item_list)

    def parser_classes(self, class_name):
        """
        class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        :param
        :return:
        """
        if isinstance(class_name, str):
            class_name = file_utils.read_data(class_name, split=None)
        if isinstance(class_name, list):
            class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        elif isinstance(class_name, dict):
            class_dict = class_name
        else:
            class_dict = None
        return class_name, class_dict

    def __getitem__(self, idx):
        '''
        :param idx:
        :return: RGB image,label id
        '''
        # idx= 0
        image_path = self.item_list[idx][0]
        label_id = self.item_list[idx][1]
        image = self.read_image(image_path, use_rgb=self.use_rgb)
        if self.transform and isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = self.transform(image)
        if image is None:
            print(f"bad image:{image_path}")
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        return image, label_id

    def __len__(self):
        self.num_images = len(self.item_list)
        return self.num_images

    def get_numclass(self):
        """
        get image num class
        :return:
        """
        return len(self.classes)

    def count_class_nums(self, images_list, label_index=1):
        """
        images_list=[[filename,label,...],[filename,label,...]]
        :param images:
        :param num_classes:
        :return:
        """
        class_count = {}
        for item in images_list:
            label = item[label_index]
            if label in class_count:
                class_count[label] += 1
            else:
                class_count[label] = 1
        return class_count

    def get_class_to_idx(self):
        """
        class -> ID
        :return:
        """
        return self.class_dict

    @staticmethod
    def get_data_labels(paths, only_id):
        if isinstance(paths, str): paths = [paths]
        image_lists = []
        image_labels = []
        for i, path in enumerate(paths):
            print("loading image from:{}".format(path))
            if not os.path.exists(path):
                raise Exception("image_dir:{}".format(path))
            image_list, label_list = file_utils.get_files_labels(path, postfix=IMG_POSTFIX)
            if not only_id:
                # dir_id = os.path.basename(image_dir)
                dir_id = str(i)
                label_list = [os.path.join(dir_id, l) for l in label_list]
            print("----have images:{},lable set:{}".format(len(image_list), len(set(label_list))))
            image_lists += image_list
            image_labels += label_list
        return image_lists, image_labels

    def parser_annotation(self, paths, class_dict=None, shuffle=True, only_id=True):
        """
        get image list and classes
        :param paths:
        :param shuffle:
        :param only_id: <bool>labels(classes)的名称格式,避免多个数据集的相同的ID
                        False:  label = "parent_dir/sub_dir",
                        True:   label = "sub_dir"
        :return:
        """
        image_lists, image_labels = self.get_data_labels(paths=paths, only_id=only_id)
        if isinstance(class_dict, dict):
            class_dict = class_dict
        elif isinstance(class_dict, list):
            class_dict = {str(class_dict[i]): i for i in range(len(class_dict))}
        else:
            classes = list(set(image_labels))
            classes.sort()
            class_dict = {classes[i]: i for i in range(len(classes))}
        item_list = self.get_item_list(image_lists, image_labels, class_dict)
        if shuffle:
            random.seed(100)
            random.shuffle(item_list)
        # max_nums = 17091657
        # max_nums = min(len(item_list), max_nums)
        # item_list = item_list[:max_nums]
        return class_dict, item_list

    @staticmethod
    def get_item_list(image_lists, image_labels, class_dict):
        """
        get images
        :param image_lists: image list
        :param image_labels: image label
        :param class_dict:
        :return:
        """
        item_list = []
        for image_path, label in zip(image_lists, image_labels):
            if not label in class_dict:
                continue
            label_id = class_dict[label]
            item_list.append((image_path, label_id))
        return item_list

    @staticmethod
    def read_image(path, use_rgb=True):
        """
        读取图片的函数
        :param path:
        :param use_rgb:
        :return:
        """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        # image = cv2.imread(path,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_UNCHANGED)
        if use_rgb and isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        return image

    def data_preproccess(self, image):
        """
        数据预处理
        :param data:
        :return:
        """
        image = self.transform(image)
        return image

    def get_classes_weights(self, label_index=1):
        """
        :param label_index:
        :return:
        """
        labels_list = []
        for item in self.item_list:
            label = item[label_index]
            labels_list.append(label)
        # weight = balanced_classes.create_sample_weight_torch(labels_list)
        weight = balanced_classes.create_class_sample_weight_custom(labels_list,
                                                                    balanced="auto",
                                                                    weight_type="sample_weight")
        return weight


if __name__ == '__main__':
    image_dir1 = "/home/dm/nasdata/dataset/csdn/emotion/MMAFEDB/test"
    input_size = [112, 112]
    from classifier.transforms import build_transform

    trans_type = "train"
    transform = build_transform.image_transform(input_size,
                                                rgb_mean=[0., 0., 0.],
                                                rgb_std=[1.0, 1.0, 1.0],
                                                trans_type=trans_type)
    batch_size = 1
    dataset_train = ImageFolderDataset(files=image_dir1,
                                       transform=transform,
                                       resample=False,
                                       shuffle=False,
                                       disp=True)
    dataloader = DataLoader(dataset_train, batch_size, shuffle=False, num_workers=16)
    epochs = 1
    for epoch in range(epochs):
        for batch_image, batch_label in iter(dataloader):
            image = batch_image[0, :]
            # image = image.numpy()  #
            image = np.array(image, dtype=np.float32)
            image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
            print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
            image_utils.cv_show_image("image", image, use_rgb=True)
