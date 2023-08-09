import os
import sys
import numpy as np
import tensorflow as tf

print("TF:{}".format(tf.__version__))

import numpy as np
import torch
import torch
import keras
import cv2
from keras_preprocessing import image
from torch.autograd import Variable
from pytorch2keras import converter
from models.nets.build_nets import build_nets
from utils import torch_tools


def get_image_tensor(image_file="test.jpg", input_size=[112, 112], transpose=True):
    " image = np.random.uniform(0, 1, (1, 3, input_size[1], input_size[0]))"
    image = cv2.imread(image_file)
    image = cv2.resize(image, dsize=tuple(input_size))
    image_tensor = image / 255.0
    if transpose:
        image_tensor = image_tensor.transpose(2, 0, 1)  # NHWC->NCHW
    image_tensor = image_tensor[np.newaxis, :]
    return image_tensor


def torch2keras(net_type,
                model_path,
                input_size,
                num_classes,
                width_mult=1.0,
                out_keras_model=None,
                device="cpu"):
    """
    https://github.com/nerox8664/pytorch2keras
    pip install tensorflow==2.3.0  keras==2.3.1 pytorch2keras
    =======================================================================
    To use the converter properly, please, make changes in your ~/.keras/keras.json:

    {
        "floatx": "float32",
        "epsilon": 1e-07,
        "backend": "tensorflow",
        "image_data_format": "channels_last"
    }
     =======================================================================
    :param model_path: torch model file
    :param input_size: torch model input_size
    :param out_keras_model: out_keras_model
    :param device: cpu
    :return:
    """

    if not os.path.exists(model_path):
        raise Exception("Error:{}".format(model_path))
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    if not out_keras_model:
        model_name = model_name[:-len(".pth")] + ".h5"
        out_keras_model = os.path.join(model_dir, model_name)

    # load torch Model
    t_model = build_model(net_type, model_path, input_size, num_classes, width_mult=width_mult)

    # create random inputs datas
    np.random.seed(200)
    inputs = get_image_tensor()
    t_inputs = Variable(torch.FloatTensor(inputs)).to(device)
    k_inputs = inputs.transpose(0, 2, 3, 1)  # [B,C,H,W]-->[B,H,W,C]

    # forward torch
    t_model = t_model.to(device)
    t_model = t_model.eval()
    t_output = t_model(t_inputs)

    # convert torch weight to keras weight
    k_model = converter.pytorch_to_keras(model=t_model,
                                         args=t_inputs,
                                         input_shapes=[(3, input_size[1], input_size[0],)],
                                         verbose=True,
                                         change_ordering=True,  # change CHW to HWC
                                         )
    k_model.summary()
    # 保存模型
    k_model.save(out_keras_model)
    # 重新载入模型
    del k_model
    # load keras model
    k_model = tf.keras.models.load_model(out_keras_model)
    k_output = k_model(k_inputs, training=False)

    t_output = np.asarray(t_output.detach().numpy(), dtype=np.float32)
    k_output = np.asarray(k_output, dtype=np.float32)
    # print("t_output:{}".format(t_output.shape))
    # print("k_output:{}".format(k_output.shape))
    print("t_output:{},{}".format(t_output.shape, t_output[0,:10]))
    print("k_output:{},{}".format(k_output.shape, k_output[0,:10]))
    print("successfully convert to keras model")
    print("torch model at: {}".format(model_path))
    print("save  model at: {}".format(out_keras_model))


def build_model(net_type, model_path, input_size, num_classes, width_mult=1.0, **kwargs):
    """
    build model
    :param net_type:
    :param model_path:
    :return:
    """
    model = build_nets(net_type, input_size, num_classes, width_mult=width_mult, pretrained=False, **kwargs)
    state_dict = torch_tools.load_state_dict(model_path, module=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    """
    t_output:(1, 2),[[-0.8178028   0.81784195]]
    k_output:(1, 2),[[-0.81780267  0.8178419 ]]
    """
    net_type = "mobilenet_v2"
    model_path = "/home/dm/data3/git_project/torch-image-classification-pipeline/work_space/card/mobilenet_v2/mobilenet_v2_1.0_84_84_SGD_LabelSmooth_resample_text_log_20210702100732/model/best_model_100_99.5726.pth"
    input_size = [84, 84]
    num_classes = 156
    width_mult = 1.0
    torch2keras(net_type,
                model_path,
                input_size,
                num_classes,
                width_mult=width_mult,
                out_keras_model=None)
