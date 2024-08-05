import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import NDCNet as net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import os

def dehaze_image(image_path):
    data_hazy = Image.open(image_path).convert('L')  # 转换为灰度图像

    # 转换为 numpy 数组，并添加通道维度
    data_hazy = np.array(data_hazy)
    data_hazy = np.expand_dims(data_hazy, axis=0)

    # 数据归一化
    data_hazy = (data_hazy / 255.0).astype(np.float32)

    # 转换为 PyTorch Tensor，并添加批处理维度
    data_hazy = torch.from_numpy(data_hazy).unsqueeze(0).cuda()

    # 加载模型
    dehaze_net = net.NDCNet().cuda()
    dehaze_net.load_state_dict(torch.load('pre_trained_model/NDCNet.pth'))

    # 运行模型
    clean_image = dehaze_net(data_hazy)

    # 保存结果图像
    result_folder = "DC_image/DC_clear_image"
    print(result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    save_path = os.path.join(result_folder, os.path.basename(image_path))
    torchvision.utils.save_image(clean_image.squeeze(0), save_path)


if __name__ == '__main__':

    test_list = glob.glob("DC_image/DC_haze_image//*")

    # 创建结果保存文件夹
    if not os.path.exists("DC_image/DC_clear_image"):
        os.makedirs("DC_image/DC_clear_image")

    for image in test_list:
        dehaze_image(image)
        print(image, "done!")
