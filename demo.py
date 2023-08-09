# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-08-03 17:39:23
"""
import os
import sys

sys.path.append("libs")
import argparse
import numpy as np
import torch
import cv2
from basetrainer.utils import log, setup_config
from pybaseutils import file_utils, image_utils, coords_utils
from classifier import inference
from libs.detector import Detector


class Predictor(inference.Inference):
    def __init__(self, cfg):
        super(Predictor, self).__init__(cfg)
        self.detector = Detector(detect_type="face", prb_thresh=0.85, iou_thresh=0.2)

    def detect_face(self, rgb_image):
        """进行人脸检测"""
        dets, labels = self.detector.detect(rgb_image)  # dets is bbox_score
        faces = []
        if len(dets) > 0:
            boxes = dets[:, 0:4]
            boxes = coords_utils.extend_xyxy(boxes, scale=[1.1, 1.1])
            # boxes = coords_utils.get_square_bboxes(boxes, use_max=True)
            faces = image_utils.get_bboxes_crop(rgb_image, boxes)
        return faces, dets

    def predict(self, faces):
        """分类预测"""
        pred_index, pred_score = np.zeros((0,)), np.zeros((0,))
        if len(faces) > 0:
            pred_index, pred_score = self.inference(faces)
            # pred_index = self.label2class_name(pred_index)
        return pred_index, pred_score

    def task(self, image):
        faces, dets = self.detect_face(image)
        pred_index, pred_score = self.predict(faces)
        return pred_index, pred_score, dets

    def start_capture(self, video_file, save_video=None, interval=1, vis=True):
        """
        start capture video
        :param video_file: *.avi,*.mp4,... 视频文件或摄像头ID
        :param save_video: *.avi 保存视频文件路径
        :param interval: 视频抽帧间隔
        :return:
        """
        # cv2.moveWindow("test", 1000, 100)
        video_cap = image_utils.get_video_capture(video_file)
        width, height, num_frames, fps = image_utils.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_utils.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            if count % interval == 0:
                # 设置抽帧的位置
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                isSuccess, frame = video_cap.read()
                if not isSuccess:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred_index, pred_score, dets = self.task(image)
                image = self.show_image(image, dets, pred_score, pred_index, use_rgb=True, vis=vis, delay=10)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if save_video:
                    self.video_writer.write(frame)
            count += 1
        video_cap.release()

    def image_dir_predict(self, image_dir, out_dir=None, use_rgb=True, vis=True):
        """
        :param image_dir: list,*.txt ,image path or directory
        :param out_dir: 保存输出结果
        :param use_rgb: 输入图片格式
        :param vis: 是否可视化识别结果
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir, shuffle=False,
                                                postfix=['*.jpg', '*.jpeg', '*.png', '*.tif', "*.JPG", "*.bmp"])
        for path in image_list:
            image = image_utils.read_image(path, use_rgb=use_rgb)
            # image = image_utils.resize_image(image, size=(480, None))
            pred_index, pred_score, dets = self.task(image)
            info = "path:{} pred_index:{},pred_score:{}".format(path, pred_index, pred_score)
            print(info)
            image = self.show_image(image, dets, pred_score, pred_index, use_rgb=use_rgb, vis=vis)
            if out_dir:
                out_file = file_utils.create_dir(out_dir, None, os.path.basename(path))
                print("save result：{}".format(out_file))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_file, image)

    def show_image(self, image, dets, pred_score, pred_index, use_rgb=True, delay=0, vis=True):
        image = image_utils.draw_image_detection_bboxes(image, dets, pred_score, pred_index,
                                                        class_name=self.class_name, fontScale=-1)
        if vis:
            image_utils.cv_show_image("det-result", image, use_rgb=use_rgb, delay=delay)
        return image


def get_parser():
    # 配置文件
    config_file = "configs/config.yaml"
    # 模型文件
    model_file = "data/pretrained/mobilenet_v2_1.0_CrossEntropyLoss_20230313090258/model/latest_model_099_94.7200.pth"
    # 待测试图片目录
    image_dir = "data/test_image"
    video_file = None
    # video_file = "data/video-test.mp4" # 视频文件
    # video_file = "0" # 摄像头
    parser = argparse.ArgumentParser(description="Inference Argument")
    parser.add_argument("-c", "--config_file", help="configs file", default=config_file, type=str)
    parser.add_argument("-m", "--model_file", help="model_file", default=model_file, type=str)
    parser.add_argument("--device", help="cuda device id", default="cuda:0", type=str)
    parser.add_argument("--image_dir", help="image file or directory", default=image_dir, type=str)
    parser.add_argument('--video_file', type=str, default=video_file, help='camera id or video file')
    parser.add_argument('--out_dir', type=str, default="output", help='save det result image')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    print(parser.parse_args())
    cfg = setup_config.parser_config(parser.parse_args(), cfg_updata=False)
    t = Predictor(cfg)
    if isinstance(cfg.video_file, str):
        if len(cfg.video_file) == 1: cfg.video_file = int(cfg.video_file)
        save_video = os.path.join(cfg.out_dir, "result.avi") if cfg.out_dir else None
        t.start_capture(cfg.video_file, save_video, interval=1, vis=True)
    else:
        t.image_dir_predict(cfg.image_dir, cfg.out_dir, vis=True)
