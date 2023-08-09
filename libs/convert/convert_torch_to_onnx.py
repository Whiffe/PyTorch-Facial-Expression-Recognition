"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import torch.onnx
import onnx
from classifier.models.build_models import get_models
from basetrainer.utils import torch_tools


def build_net(model_file, net_type, input_size, num_classes, width_mult=1.0):
    """
    :param model_file: 模型文件
    :param net_type: 模型名称
    :param input_size: 模型输入大小
    :param num_classes: 类别数
    :param width_mult:
    :return:
    """
    model = get_models(net_type, input_size, num_classes, width_mult=width_mult, is_train=False, pretrained=False)
    state_dict = torch_tools.load_state_dict(model_file)
    model.load_state_dict(state_dict)
    return model


def convert2onnx(model_file, net_type, input_size, num_classes, width_mult=1.0, device="cpu", onnx_type="default"):
    model = build_net(model_file, net_type, input_size, num_classes, width_mult=width_mult)
    model = model.to(device)
    model.eval()
    model_name = os.path.basename(model_file)[:-len(".pth")] + ".onnx"
    onnx_path = os.path.join(os.path.dirname(model_file), model_name)
    # dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0]).to(device)
    # torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
    #                   input_names=['input'],output_names=['scores', 'boxes'])
    do_constant_folding = True
    if onnx_type == "default":
        torch.onnx.export(model, dummy_input, onnx_path, verbose=False, export_params=True,
                          do_constant_folding=do_constant_folding,
                          input_names=['input'],
                          output_names=['output'])
    elif onnx_type == "det":
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          do_constant_folding=do_constant_folding,
                          export_params=True,
                          verbose=False,
                          input_names=['input'],
                          output_names=['scores', 'boxes', 'ldmks'])
    elif onnx_type == "kp":
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          do_constant_folding=do_constant_folding,
                          export_params=True,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'])
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx_path)


if __name__ == "__main__":
    net_type = "mobilenet_v2"
    width_mult = 1.0
    input_size = [112, 112]
    num_classes = 7
    model_file = "../../data/pretrained/mobilenet_v2_1.0_CrossEntropyLoss_20230313090258/model/latest_model_099_94.7200.pth"
    convert2onnx(model_file, net_type, input_size, num_classes, width_mult=width_mult)
