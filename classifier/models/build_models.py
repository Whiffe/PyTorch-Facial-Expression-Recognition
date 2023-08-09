# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-01-17 17:46:38
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from torchvision.models.resnet import model_urls
from torch.hub import load_state_dict_from_url

root = os.path.dirname(__file__)
MODEL_URL = {
    "mobilenet_v2": os.path.join(root, "pretrained/mobilenet_v2-b0353104.pth"),
    "resnet18": os.path.join(root, "pretrained/resnet18-5c106cde.pth"),
    "resnet34": os.path.join(root, "pretrained/resnet34-333f7ec4.pth"),
}


def get_pretrained(net_type, pretrained):
    pretrained = MODEL_URL[net_type] if (pretrained and net_type in MODEL_URL) else pretrained
    return pretrained


def get_models(net_type, input_size, num_classes, width_mult=1.0, is_train=True, pretrained=True, **kwargs):
    """
    :param net_type:  resnet18,resnet34,resnet50, mobilenet_v2
    :param input_size: 模型输入大小
    :param num_classes: 类别数
    :param width_mult:
    :param is_train:
    :param pretrained:
    :param kwargs:
    :return:
    """
    if net_type.lower().startswith("resnet"):
        model = resnet_model(net_type,
                             num_classes=num_classes,
                             pretrained=get_pretrained(net_type, pretrained))
    elif net_type.lower() == "googlenet":
        model = googlenet(num_classes=num_classes, pretrained=pretrained)
    elif net_type.lower() == "inception_v3":
        model = inception_v3(num_classes=num_classes, pretrained=pretrained)
    elif net_type.lower() == "mobilenet_v2":
        model = mobilenet_v2(num_classes=num_classes,
                             width_mult=width_mult,
                             pretrained=get_pretrained(net_type, pretrained))
    else:
        raise Exception("Error: net_type:{}".format(net_type))
    return model


def resnet_model(net_type, num_classes, pretrained=True):
    """
    :param net_type: resnet18,resnet34
    :param num_classes: if None ,return no-classifier-layers backbone
    :param pretrained: <bool> pretrained
    :return:
    """
    if net_type.lower() == "resnet18":
        backbone = models.resnet18(pretrained=False)
        out_channels = 512
        expansion = 1
    elif net_type.lower() == "resnet34":
        backbone = models.resnet34(pretrained=False)
        out_channels = 512
        expansion = 1
    elif net_type.lower() == "resnet50":
        backbone = models.resnet50(pretrained=False)
        out_channels = 512
        expansion = 4
    else:
        raise Exception("Error: net_type:{}".format(net_type))

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu") if isinstance(pretrained, str) else None
        state_dict = state_dict if state_dict else load_state_dict_from_url(model_urls[net_type])
        backbone.load_state_dict(state_dict)

    if num_classes:
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        assert backbone.fc.in_features == out_channels * expansion
        backbone.fc = nn.Linear(out_channels * expansion, num_classes)
    else:
        # remove mobilenet_v2  classifier layers
        model_dict = OrderedDict(backbone.named_children())
        model_dict.pop("avgpool")
        model_dict.pop("fc")
        backbone = torch.nn.Sequential(model_dict)
        # if attention:
        #     backbone.add_module("attention", ChannelAttention(input_size=last_channel))
    return backbone


def mobilenet_v2(num_classes=None, width_mult=1.0, pretrained=False):
    """
    :param pretrained: <bool> pretrained
    :param num_classes: if None ,return no-classifier-layers backbone
    :param last_channel:
    :param width_mult:
    :return:
    """
    model = models.mobilenet_v2(pretrained=False, width_mult=width_mult)
    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu") if isinstance(pretrained, str) else None
        state_dict = state_dict if state_dict else load_state_dict_from_url(model_urls['mobilenet_v2'])
        model.load_state_dict(state_dict)
    # state_dict1 = model.state_dict()
    if num_classes:
        last_channel = model.last_channel
        # replace mobilenet_v2  classifier layers
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        model.classifier = classifier
    else:
        # remove mobilenet_v2  classifier layers
        model_dict = OrderedDict(model.named_children())
        model_dict.pop("classifier")
        model = torch.nn.Sequential(model_dict)
        # state_dict2 = model.state_dict()
    return model


def googlenet(num_classes=None, pretrained=False):
    """
    :param num_classes: if None ,return no-classifier-layers backbone
    :param pretrained: <bool> pretrained
    :return:
    """
    model = models.googlenet(pretrained=pretrained, aux_logits=False)
    if num_classes:
        last_channel = model.fc.in_features
        model.fc = nn.Linear(last_channel, num_classes)
    else:
        model_dict = OrderedDict(model.named_children())
        model_dict.pop("dropout")
        model_dict.pop("fc")
        model = torch.nn.Sequential(model_dict)
        # state_dict2 = model.state_dict()
    return model


def inception_v3(num_classes=None, pretrained=False):
    """
    :param num_classes: if None ,return no-classifier-layers backbone
    :param pretrained: <bool> pretrained
    :return:
    """
    model = models.inception_v3(pretrained=pretrained)
    if num_classes:
        last_channel = model.fc.in_features
        model.fc = nn.Linear(last_channel, num_classes)
    else:
        model_dict = OrderedDict(model.named_children())
        model_dict.pop("dropout")
        model_dict.pop("fc")
        model = torch.nn.Sequential(model_dict)
        # state_dict2 = model.state_dict()
    return model


if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 1
    width_mult = 1.0
    num_classes = 10
    input_size = [224, 224]
    x = torch.randn(size=(batch_size, 3, input_size[0], input_size[1])).to(device)
    net_type = 'resnet18'
    model = get_models(net_type, input_size, num_classes, width_mult=width_mult, pretrained=True, is_train=True)
    model = model.to(device)
    model.eval()
    out = model(x)
    print("x.shape:{}".format(x.shape))
    print("out.shape:{}".format(out.shape))
