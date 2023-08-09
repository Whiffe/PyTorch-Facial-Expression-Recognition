# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-08-02 17:48:45
"""
import argparse
import numpy as np
import torch
import PIL.Image as Image
from classifier.models.build_models import get_models
from classifier.transforms.build_transform import image_transform
from basetrainer.utils import log, setup_config
from pybaseutils import file_utils, image_utils


class Inference(object):
    def __init__(self, cfg):
        self.class_name, self.class_dict = file_utils.parser_classes(cfg.class_name)
        if self.class_name:cfg.num_classes = len(self.class_name)
        self.num_classes = cfg.num_classes
        self.device = cfg.device
        self.model = self.build_model(cfg)
        self.transform = image_transform(input_size=cfg.input_size,
                                         rgb_mean=cfg.rgb_mean,
                                         rgb_std=cfg.rgb_std,
                                         trans_type=cfg.test_transform)

    def build_model(self, cfg, **kwargs):
        """build_model"""
        model = get_models(cfg.net_type,
                           cfg.input_size,
                           cfg.num_classes,
                           pretrained=False,
                           )
        state_dict = torch.load(cfg.model_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()  # set to val mode
        return model

    def pre_process(self, images):
        """
        :param images:
        :return:
        """
        if not isinstance(images, list):
            images = [images]
        image_tensors = []
        for img in images:
            image_tensor = self.transform(Image.fromarray(img))
            image_tensors.append(torch.unsqueeze(image_tensor, dim=0))
        image_tensors = torch.cat(image_tensors)
        return image_tensors

    def post_process(self, output):
        """
        :param output:
        :return:pred_index，pred_score
        """
        prob_scores = self.softmax(np.asarray(output), axis=1)
        # prob_scores = torch.nn.functional.softmax(output, dim=1)
        if isinstance(prob_scores, torch.Tensor):
            prob_scores = prob_scores.cpu().detach().numpy()
        pred_index = np.argmax(prob_scores, axis=1)
        pred_score = np.max(prob_scores, axis=1)
        return pred_index, pred_score

    @staticmethod
    def softmax(x, axis=1):
        """
        outputs = torch.nn.functional.softmax(output, dim=1)
        :param x:
        :param axis:
        :return:
        """
        # 计算每行的最大值
        row_max = x.max(axis=axis)
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max = row_max.reshape(-1, 1)
        x = x - row_max
        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensor
        :return:
        """
        with torch.no_grad():
            out_tensor = self.model(input_tensor.to(self.device))
        return out_tensor

    def inference(self, images):
        """
        :param images: input RGB image or image list
        :return:pred_index: predict label index
                pred_score: predict label score
        """
        # 图像预处理
        input_tensor = self.pre_process(images)
        output = self.forward(input_tensor)
        # 模型输出后处理
        pred_index, pred_score = self.post_process(output.cpu())
        return pred_index, pred_score

    def label2class_name(self, pred_index):
        if self.class_name:
            pred_index = [self.class_name[i] for i in pred_index]
        return pred_index

    def image_dir_predict(self, image_dir, isshow=True, shuffle=False):
        """
        :param image_dir: list,*.txt ,image path or directory
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir, shuffle=shuffle)
        for path in image_list:
            image = image_utils.read_image(path, use_rgb="RGB")
            pred_index, pred_score = self.inference(image)
            pred_index = self.label2class_name(pred_index)
            info = "path:{} pred_index:{},pred_score:{}".format(path, pred_index, pred_score)
            print(info)
            if isshow:
                image_utils.cv_show_image("predict", image)
