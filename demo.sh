#!/usr/bin/env bash
# 测试图片
image_dir='data/test_image' # 测试图片的目录
model_file="data/pretrained/mobilenet_v2_1.0_CrossEntropyLoss_20230313090258/model/latest_model_099_94.7200.pth" # 模型文件
out_dir="output/" # 保存检测结果
python demo.py --image_dir $image_dir --model_file $model_file --out_dir $out_dir

# 测试视频文件
video_file="data/video-test.mp4" # 测试视频文件，如*.mp4,*.avi等
model_file="data/pretrained/mobilenet_v2_1.0_CrossEntropyLoss_20230313090258/model/latest_model_099_94.7200.pth" # 模型文件
out_dir="output/" # 保存检测结果
#python demo.py --video_file $video_file --model_file $model_file --out_dir $out_dir


# 测试摄像头
video_file=0 # 测试摄像头ID
model_file="data/pretrained/mobilenet_v2_1.0_CrossEntropyLoss_20230313090258/model/latest_model_099_94.7200.pth" # 模型文件
out_dir="output/" # 保存检测结果
#python demo.py --video_file $video_file --model_file $model_file --out_dir $out_dir
