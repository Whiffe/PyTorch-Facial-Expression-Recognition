# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Callable, ByteString
from classifier.criterion.focal import FocalLoss, MultiFocalLoss
from classifier.criterion.label_smooth import LabelSmooth, LabelSmoothing, LabelSmoothingCrossEntropy
from basetrainer.utils import log


class KLDivLoss(nn.Module):
    """Norm Loss(criterion)相对熵 """

    def __init__(self):
        """
        :param norm: 1 or 2
        """
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduce=False)

    def forward(self, input, target):
        input = torch.log(input)
        loss = self.loss(input, target)
        loss = loss.sum() / loss.shape[0]
        return loss


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum() / loss.shape[0]
    return loss


class ComposeLoss(object):
    def __init__(self, criterions: Dict[str, Callable], weights: Dict[str, float] = None):
        """
        联合LOSS函数
        :param criterions: Dict[str, Callable], Loss function
        :param  weights: Dict[str, float]
        """
        if isinstance(weights, dict):
            assert criterions.keys() == weights.keys(), \
                Exception("Key Error:criterions:{},weights:{}".format(criterions.keys(), weights.keys()))
        self.weights = weights
        self.criterions = criterions

    def __call__(self, logits, labels):
        losses = {}
        for name, criterion in self.criterions.items():
            loss = criterion(logits, labels)
            if isinstance(loss, dict):
                losses = {k: v * self.weights[name] for k, v in loss.items()} if self.weights else loss
            else:
                losses[name] = loss * self.weights[name] if self.weights else loss
        return losses


def build_criterion(loss_type: str or List[str] or Dict[str, float],
                    num_classes=None,
                    class_weight=None,
                    device="cuda:0"):
    """
    使用nn.BCELoss需要在该层前面加上Sigmoid函数
    使用nn.CrossEntropyLoss会自动加上Softmax层,所以最后一层不需要加上Softmax()
    :param loss_type: loss_type={loss_name: loss_weigth}
                      FocalLoss,CrossEntropyLoss,LabelSmooth
    :param num_classes:
    :param class_weight: 类别loss权重， a manual rescaling weight given to each class.
                         If given, has to be a Tensor of size `Class`
    :param ignore_index: 忽略label=ignore_index的值
    :return:
    """
    logger = log.get_logger()
    if isinstance(class_weight, np.ndarray):
        class_weight = torch.from_numpy(class_weight.astype(np.float32)).to(device)
    if isinstance(loss_type, str):
        loss_type = [loss_type]
    if isinstance(loss_type, list):
        loss_type = {loss: 1.0 for loss in loss_type}
    assert isinstance(loss_type, dict)
    criterions = {}
    weights = {}
    for loss, loss_weight in loss_type.items():
        criterion = get_criterion(loss, num_classes, class_weight, device=device)
        criterions[loss] = criterion
        weights[loss] = loss_weight
    criterions = ComposeLoss(criterions=criterions, weights=weights)
    logger.info("use criterions:{}".format(weights))
    # print("criterions:{}".format(weights))
    return criterions


def get_criterion(loss_type: str, num_classes=None, class_weight=None, ignore_index=255, device="cuda:0"):
    """
    使用nn.BCELoss需要在该层前面加上Sigmoid函数
    使用nn.CrossEntropyLoss会自动加上Softmax层,所以最后一层不需要加上Softmax()
    :param loss_type: FocalLoss,CrossEntropyLoss,LabelSmooth
    :param num_classes:
    :param loss_weights: loss权重， a manual rescaling weight given to each class.
                         If given, has to be a Tensor of size `Class`
    :return:
    """
    if isinstance(class_weight, np.ndarray):
        class_weight = torch.from_numpy(class_weight.astype(np.float32)).to(device)
    if loss_type.lower() == "FocalLoss".lower():
        criterion = FocalLoss()
    elif loss_type.lower() == "MultiFocalLoss".lower():
        criterion = MultiFocalLoss(num_classes)
    elif loss_type.lower() == "CrossEntropyLoss".lower():
        criterion = nn.CrossEntropyLoss(class_weight)
    elif loss_type.lower() == "LabelSmooth".lower() or loss_type.lower() == "LabelSmoothing".lower():
        criterion = LabelSmooth(num_classes, device=device)
        # loss = LabelSmoothing()
        # loss = LabelSmoothingCrossEntropy() # 分类作用不大
    elif loss_type.lower() == "BCELoss".lower():
        # 用于二分类和多标签分类，一般前面需要加Sigmoid，或者直接使用BCEWithLogitsLoss
        criterion = nn.BCELoss(reduction='mean')
        criterions = {"BCELoss": criterion}
        weights = {"BCELoss": 1.0}
    elif loss_type.lower() == "BCELogit".lower():
        # 用于二分类和多标签分类
        # BCEWithLogitsLoss = Sigmoid + BCELoss
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        criterions = {"BCELogit": criterion}
        weights = {"BCELogit": 1.0}
    elif loss_type.lower() == "L1Loss".lower():
        criterion = nn.L1Loss(reduction='mean')
    elif loss_type.lower() == "mse".lower():
        # loss = nn.MSELoss(reduce=True, size_average=True)
        criterion = nn.MSELoss(reduction='mean')
    elif loss_type.lower() == "KLDivLoss".lower() or loss_type.lower() == "LabelDistribution".lower():
        # Label Distribution
        criterion = KLDivLoss()
    else:
        raise Exception("Error:{}".format(loss_type))
    print("loss_type:{}".format(loss_type))
    return criterion


if __name__ == "__main__":
    # inputs = torch.randn(4, 3)
    # target = torch.tensor([0, 1, 1, 2])  # 必须为Long类型，是类别的序号
    inputs = np.array([[-0.4089, -1.2471, 0.5907],
                       [-0.4897, -0.8267, -0.7349],
                       [0.5241, -0.1246, -0.4751]])
    target = np.array([0, 1, 1])
    inputs = torch.from_numpy(inputs)
    target = torch.from_numpy(target)
    print(inputs.shape)
    print(target.shape)
    loss_type = {"CrossEntropyLoss": 1.0, "TripletLoss": 1.0}
    crition = get_criterion(loss_type="CrossEntropyLoss")
    # crition = build_criterion(loss_type="TripletLoss")
    # crition = build_criterion(loss_type=loss_type)
    loss = crition(inputs, target)
    print(loss)
