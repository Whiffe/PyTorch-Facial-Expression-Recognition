# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import sysrsync
from datetime import datetime


def get_polyaxon_output(dir=None):
    from polyaxon_client.tracking import get_outputs_path
    polyaxon_output = get_outputs_path()
    if dir:
        polyaxon_output = os.path.join(polyaxon_output, dir)
    return polyaxon_output


def polyaxon_root(dir, data_node):
    from polyaxon_client.tracking import get_data_paths, get_outputs_path
    out = os.path.join(get_data_paths()[data_node], dir)
    print("root:{}====>{}".format(dir, out))
    return out


def get_polyaxon_dataroot(dir="", data_node="ceph", use_local=False):
    """
    datauser@192.168.68.79:/nasdata/atp/data(上传至nas数据存储节点，代码中可使用 get_data_paths()['data-pool'] 访问）
    datauser@192.168.68.79:/sharedata06 （上传至SSD数据存储节点，代码中可使用 get_data_paths()['ssd'] 访问）
    datauser@192.168.68.79:/sharedata（上传至SSD数据存储节点，代码中可使用 get_data_paths()['ssd20'] 访问）
    datauser@192.168.68.79:/atpcephdata(上传至ceph存储，代码中可使用get_data_paths()['ceph']访问)
    :param dir: str or list
    :param data_node: data_node APT数据节点,ceph or newceph
    :param use_local: 是否将数据拷贝到本地数据节点
    :return:
    """
    if isinstance(dir, str):
        dir = polyaxon_root(dir, data_node) if dir else dir
    elif isinstance(dir, list):
        dir = [polyaxon_root(d, data_node) for d in dir if d]
    else:
        raise Exception("Error:{}".format(dir))
    if use_local:
        dir = rsync_dataset_node(dir)
    return dir


def polyaxon_env(data_root, val_root, val_dataset, update=False, use_local=True):
    """
    get_data_paths()['ssd']指向的是SSD数据存储节点，在每个训练节点的挂载路径为：/sharedata06；
    get_data_paths()['data-pool']指向nasdata节点，在每个训练节点的挂载路径为：/nasdata/atp/data，
    get_data_paths()['ssd20']指向的是20T的SSD数据存储节点，在每个训练节点的挂载路径为：/sharedata；
    # 训练机器本地盘路径
    host_path = get_data_paths()['host-path']
    :param data_root:
    :param val_root:
    :param val_dataset:
    :param update : data
    :return:
    """
    from polyaxon_client.tracking import get_data_paths, get_outputs_path
    print("The environment is polyaxon")
    # polyaxon_dataroot = os.path.join(get_data_paths()["data-pool"], 'FaceData')  # upload data to TFR file
    host_path = os.path.join(get_data_paths()['host-path'], "FaceData")
    # src_path = os.path.join(get_data_paths()['ssd'], 'FaceData')  # upload data to TFR file
    src_path = os.path.join(get_data_paths()['ceph'], 'FaceData')  # upload data to TFR file
    polyaxon_output = get_outputs_path()

    if isinstance(data_root, str):
        data_root = [data_root]
    if use_local:
        dst_data_root = [os.path.join(host_path, dataset) for dataset in data_root]
        # sync train data
        for i, (image_root, name) in enumerate(zip(dst_data_root, data_root)):
            if update or not os.path.exists(image_root):
                dst_data_root[i] = rsync(src_path, host_path, name)

        # sync val data
        dst_val_root = os.path.join(host_path, val_root)
        # file_processing.remove_dir(dst_val_root)
        for val_name in val_dataset:
            val_name = "{}.bin".format(val_name)
            local_val_path = os.path.join(dst_val_root, val_name)
            if update or not os.path.exists(local_val_path):
                val_src_root = os.path.join(src_path, val_root)
                local_val_path = rsync(val_src_root, dst_val_root, val_name)
    else:
        dst_data_root = [os.path.join(src_path, dataset) for dataset in data_root]
        dst_val_root = os.path.join(src_path, val_root)

    return dst_data_root, dst_val_root, polyaxon_output


def rsync_data_node(dir, dst_node='host-path'):
    """
    同步数据到指定数据节点
    :param dir: 原始数据路径
    :param dst_node: 需要拷贝到的数据节点
    :return:
    """
    from polyaxon_client.tracking import get_data_paths, get_outputs_path
    time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    dst_root = os.path.join(get_data_paths()[dst_node], time)
    # fix a Bug: ATP无法通过os.path.isfile判断是否是文件
    # isfile = os.path.isfile(dir)
    isfile = len(dir.split(".")) > 1
    name = os.path.basename(dir)
    print("isfile:{},{}".format(isfile, dir))
    if isfile:
        dir = os.path.dirname(dir)
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    print("copy data from:{}".format(dir))
    print("destination   :{}".format(dst_root))
    print("rsync data ...")
    start = datetime.now()
    sysrsync.run(source=dir, destination=dst_root, options=['-a'])
    end = datetime.now()
    print("rsync data done,run time:{}".format(end - start))
    if isfile:
        dst_root = os.path.join(dst_root, name)
    return dst_root


def rsync_dataset_node(dir, dst_node='host-path'):
    """
    同步数据到指定数据节点
    :param src_path: 原始数据路径
    :param dst_node: 需要拷贝到的数据节点
    :return:
    """
    if isinstance(dir, str):
        dir = rsync_data_node(dir, dst_node) if dir else dir
    elif isinstance(dir, list):
        dir = [rsync_data_node(d, dst_node) for d in dir if d]
    else:
        raise Exception("Error:{}".format(dir))
    return dir


def rsync_data():
    """
    rsync dataset
    :return:
    """
    from polyaxon_client.tracking import get_data_paths, get_outputs_path

    source = os.path.join(get_data_paths()['ceph'], 'FaceData')  # upload data to TFR file
    destination = os.path.join(get_data_paths()['host-path'], "FaceData")
    if not os.path.exists(destination):
        os.makedirs(destination)
    print("copy data from:{}".format(source))
    print("destination   :{}".format(destination))
    print("rsync data ...")
    start = datetime.now()
    sysrsync.run(source=source, destination=destination, options=['-a'])
    end = datetime.now()
    print("rsync data done,run time:{}".format(end - start))


def rsync(src_root, host_root, name):
    """
    rsync dataset
    :param src_root:
    :param host_root:
    :param name:
    :return:
    """
    source = os.path.join(src_root, name)  # upload data to TFR file
    destination = os.path.join(host_root, name)
    if not os.path.exists(host_root):
        os.makedirs(host_root)
    print("copy data from:{}".format(source))
    print("destination   :{}".format(destination))
    print("rsync data ...")
    start = datetime.now()
    sysrsync.run(source=source, destination=destination, options=['-a'])
    end = datetime.now()
    print("rsync data done,run time:{}".format(end - start))
    return destination


def rsync_test(name):
    """
    rsync data test
    :param name:
    :return:
    """
    source = os.path.join('utils', name)  # upload data to TFR file
    destination = os.path.join('FaceData', name)
    if not os.path.exists(destination):
        os.makedirs(destination)
    print("copy data from:{}".format(source))
    print("destination   :{}".format(destination))
    print("rsync data ...")
    start = datetime.now()
    sysrsync.run(source=source, destination=destination, options=['-a'])
    end = datetime.now()
    print("rsync data done,run time:{}".format(end - start))
    return destination


if __name__ == "__main__":
    val_src_root = "/media/dm/dm/FaceRecognition/torch-Face-Recognize-Pipeline/data/val/"
    local_val_root = "/media/dm/dm/FaceRecognition/torch-Face-Recognize-Pipeline/data/val1/val"
    val_name = "X4"
    local_val_path = rsync(val_src_root, local_val_root, val_name)
