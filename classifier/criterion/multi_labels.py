# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-image-classification-pipeline
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-11-13 10:03:04
# --------------------------------------------------------
"""

import torch
import numpy as np


class MultiLabelLoss(object):
    def __init__(self, loss_type="BCE"):
        """
        nn.BCELoss: 二分类用的交叉熵，用的时候需要在该层前面加上Sigmoid函数
        nn.BCEWithLogitsLoss: nn.BCELoss+Sigmoid层,这样做能够利用log_sum_exp trick，使得数值结果更加稳定
        nn.MultiLabelMarginLoss: 多类别（multi-class）多分类（multi-classification）的 Hinge 损失

        :param loss_type:
        """
        if loss_type == "BCE":
            self.crition = torch.nn.BCEWithLogitsLoss()
            print("use BCEWithLogitsLoss")
        elif loss_type == "SoftMargin":
            self.crition = torch.nn.MultiLabelSoftMarginLoss()
            print("use MultiLabelSoftMarginLoss")
        else:
            raise Exception("Error:{}".format(loss_type))

    def __call__(self, input, target):
        loss = self.crition(input, target)
        return loss




if __name__ == "__main__":
    pred = np.array([[-0.4089, -1.2471, 0.5907],
                     [-0.4897, -0.8267, -0.7349],
                     [0.5241, -0.1246, -0.4751]])
    label = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [1, 0, 1]])

    pred = torch.from_numpy(pred).float()
    label = torch.from_numpy(label).float()
    crition1 = MultiLabelLoss(loss_type="BCE")
    crition2 = MultiLabelLoss(loss_type="SoftMargin")
    mloss1 = crition1(pred, label)
    mloss2 = crition2(pred, label)
    print("mloss1:{}".format(mloss1))
    print("mloss2:{}".format(mloss2))
