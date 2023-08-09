# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-pipeline
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-06-03 17:21:18
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmooth(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, eps=0.1, device="cuda:0", label_smooth=True):
        super(LabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.eps = eps if label_smooth else 0
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(self.device)
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        return (-targets * log_probs).mean(0).sum()


def label_smooth_cross_entropy_loss(pred_class_outputs, gt_classes, eps=0.1, alpha=0.2):
    num_classes = pred_class_outputs.size(1)
    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt

    return loss


class LabelSmoothing(nn.Module):
    """
    https://www.jianshu.com/p/b9684ced5e33
    NLL loss with label smoothing.
    new_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes
    CrossEntropyLoss()=log_softmax() + NLLLoss()
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, input, target):
        """
        :param input: torch.Size([batch_size, num_classes]),torch.float32
        :param target: torch.Size([batch_size]),torch.int64
        :return:
        """
        logprobs = torch.nn.functional.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    https://www.jianshu.com/p/b9684ced5e33
    NLL loss with label smoothing.
    new_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes
    CrossEntropyLoss()=log_softmax() + NLLLoss()
    """

    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        """
        :param input: torch.Size([batch_size, num_classes]),torch.float32
        :param target: torch.Size([batch_size]),torch.int64
        :return:
        """
        c = input.size()[-1]
        log_preds = F.log_softmax(input, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        # CrossEntropyLoss() = log_softmax() + NLLLoss()
        l = (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction) + loss * self.eps / c
        return l


if __name__ == "__main__":
    import numpy as np

    l1 = LabelSmoothing()
    l2 = LabelSmoothingCrossEntropy()
    input = np.zeros(shape=(64, 2), dtype=np.float32) + (0.2, 0.05)
    target = np.zeros(shape=(64), dtype=np.int64)
    input = torch.from_numpy(input)
    target = torch.from_numpy(target)
    loss1 = l1(input, target)
    loss2 = l2(input, target)
    print(loss1)
    print(loss2)
