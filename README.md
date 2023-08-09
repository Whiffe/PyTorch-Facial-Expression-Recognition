# PyTorch-Classification-Trainer

## 1.介绍

基于PyTorch的图像分类Pipeline,该训练框架采用`Pytorch-Base-Trainer(PBT)`,
整套训练代码非常简单操作，用户只需要将相同类别的数据放在同一个目录下，并填写好对应的数据路径，即可开始训练了。

- Github: Pytorch-Base-Trainer: [Pytorch分布式训练框架](https://github.com/PanJinquan/Pytorch-Base-Trainer)
- pip安装包： [basetrainer](https://pypi.org/project/basetrainer/)
- Pytorch基础训练库Pytorch-Base-Trainer(支持模型剪枝 分布式训练)[使用教程](https://panjinquan.blog.csdn.net/article/details/122662902)


## 2.安装
- 依赖Python包：[requirements.txt](./requirements.txt)

```bash
# 先安装Anaconda3
# 在conda新建虚拟环境pytorch-py36(如果已经存在，则无需新建)
conda create -n pytorch-py36 python==3.6.7
# 激活虚拟环境pytorch-py36(每次运行都需要运行)
conda activate  pytorch-py36
# 安装工程依赖的包(如果已经安装，则无需安装)
pip install -r requirements.txt
```

## 3.数据：准备Train和Test数据

- Train和Test数据集，要求相同类别的图片，放在同一个文件夹下；且子目录文件夹命名为类别名称
  
![](docs/98eb1599.png)

- 类别文件：一行一个列表： [class_name.txt](data/dataset/class_name.txt) (最后一行,请多回车一行)

![](docs/37081789.png)

- 修改配置文件数据路径：[config.yaml](configs/config.yaml)
```yaml
train_data: # 可添加多个数据集
  - 'data/dataset/train' 
  - 'data/dataset/train2'
test_data: 'data/dataset/test'
class_name: 'data/dataset/class_name.txt'
```

## 4.训练
```bash
python train.py -c configs/config.yaml 
```

- 目标支持的backbone有：googlenet,inception_v3,resnet[18,34,50],mobilenet_v2等，详见[backbone](classifier/models/build_models.py)等
  ，其他backbone可以自定义添加
- 训练参数可以通过[config.yaml](configs/config.yaml)配置文件

| **参数**      | **类型**      | **参考值**   | **说明**                                       |
|:-------------|:------------|:------------|:---------------------------------------------|
| train_data   | str, list   | -           | 训练数据文件，可支持多个文件                               |
| test_data    | str, list   | -           | 测试数据文件，可支持多个文件                               |
| class_name   | str         | -           | 类别文件                               |
| work_dir     | str         | work_space  | 训练输出工作空间                                     |
| net_type     | str         | resnet18    | backbone类型,{resnet18/34/50,mobilenet_v2,googlenet,inception_v3} |
| input_size   | list        | [128,128]   | 模型输入大小[W,H]                                  |
| batch_size   | int         | 32          | batch size                                   |
| lr           | float       | 0.1         | 初始学习率大小                                      |
| optim_type   | str         | SGD         | 优化器，{SGD,Adam}                               |
| loss_type    | str         | CELoss      | 损失函数                                         |
| scheduler    | str         | multi-step  | 学习率调整策略，{multi-step,cosine}                  |
| milestones   | list        | [30,80,100] | 降低学习率的节点，仅仅scheduler=multi-step有效            |
| momentum     | float       | 0.9         | SGD动量因子                                      |
| num_epochs   | int         | 120         | 循环训练的次数                                      |
| num_warn_up  | int         | 3           | warn_up的次数                                   |
| num_workers  | int         | 12          | DataLoader开启线程数                              |
| weight_decay | float       | 5e-4        | 权重衰减系数                                       |
| gpu_id       | list        | [ 0 ]       | 指定训练的GPU卡号，可指定多个                             |
| log_freq     | in          | 20          | 显示LOG信息的频率                                   |
| finetune     | str         | model.pth   | finetune的模型                                  |
| progress     | bool        | True        | 是否显示进度条                                      |
| distributed  | bool        | False       | 是否使用分布式训练                                    |

## 5.测试Demo

- 先修改[demo.py](demo.py)

```python 配置文件
def get_parser():
    # 配置文件
    config_file = "configs/config.yaml"
    # 模型文件
    model_file = "work_space/mobilenet_v2/model/latest_model_099_97.5248.pth"
    # 待测试图片目录
    image_dir = "data/test_image"
    parser = argparse.ArgumentParser(description="Inference Argument")
    parser.add_argument("-c", "--config_file", help="configs file", default=config_file, type=str)
    parser.add_argument("-m", "--model_file", help="model_file", default=model_file, type=str)
    parser.add_argument("--device", help="cuda device id", default="cuda:0", type=str)
    parser.add_argument("--image_dir", help="image file or directory", default=image_dir, type=str)
    return parser
```

- 然后运行demo.py

```bash
python demo.py
```

## 6.可视化

目前训练过程可视化工具是使用Tensorboard，使用方法：

```bash
tensorboard --logdir=path/to/log/
```

## 7.其他

| 作者        | AI吃大瓜               |
|:------------|:--------------------|
| 联系方式    | 390737991@qq.com | 


![](copyright.png)