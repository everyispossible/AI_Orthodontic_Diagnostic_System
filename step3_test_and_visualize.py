# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhanghongyuan2017@email.szu.edu.cn
import json
import os
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from albumentations import ToTensorV2
from PIL import Image
import albumentations as A
import warnings

from torch.utils.data import DataLoader

from utils.dataset import AarizDataset

warnings.filterwarnings('ignore')


from utils.model import load_model
from utils.landmarks_utils import check_and_make_dir, calculate_prediction_metrics, load_landmark_names, calculate_per_landmark_metrics
from utils.soft_argmax import get_windowed_soft_argmax
from utils.visualize import visualize_pred_orig

def main(config):
    gpu_id=config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ✅ 新增：显示当前使用的设备信息
    if device.type == 'cuda':
        print(f"🚀 使用显卡运行")
        print(f"   GPU 编号: {gpu_id}")
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  未检测到可用显卡，使用 CPU 运行（速度较慢）")
    model=load_model(model_name=config.model_name)
    # 1. 首先加载文件内容
    checkpoint = torch.load(config.load_weight_path, map_location=device, weights_only=False)

    # 2. 判断加载的内容是否为字典，并且包含 'model_state_dict' 键
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 如果是，则提取出真正的模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 成功从检查点加载模型权重。")
    else:
        # 如果不是，说明文件本身就是模型权重，直接加载
        model.load_state_dict(checkpoint)
        print("✅ 成功直接加载模型权重。")
    model.to(device)

    test_transform = A.Compose([
        A.Resize(512, 512),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['point_indices'], remove_invisible=False))
    test_dataset=AarizDataset(json_path=config.test_json_path,transform=test_transform)
    test_loader=DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2
    )

    with torch.no_grad():
        model.eval()
        test_result_dict={}

        for (images,data_packets) in tqdm.tqdm(test_loader):
            images=images.to(device)
            landmarks=data_packets['landmarks'].to(device)
            scale = data_packets['scale'].float().to(device)  # [B]
            pixel_size = data_packets['pixel_size'].float().to(device)  # [B]
            file_path = data_packets['file_path']  # [B]

            output_heatmaps = model(images)

    

         
            for b in range(output_heatmaps.shape[0]):

                current_output_heatmap = output_heatmaps[b]
                current_pixel_size = pixel_size[b]
                current_scale = scale[b]
                current_landmarks = landmarks[b]/current_scale
                current_file_path = file_path[b]
                predicted_landmarks=get_windowed_soft_argmax(current_output_heatmap)
                #必须再坐标计算出来后再还原
                predicted_landmarks=predicted_landmarks/current_scale
                test_result_dict[current_file_path] = {
                    'pixel_size': current_pixel_size,
                    'orig_landmarks': current_landmarks,
                    'predicted_landmarks': predicted_landmarks,
                }
        mean_radius_dist_error, mean_radius_mm_error, sdr = calculate_prediction_metrics(test_result_dict)
        print(f'Mean Radius dist Error (MRE): {mean_radius_dist_error}| Mean Radius mm Error (MRE): {mean_radius_mm_error} | 2mm Success Detection Rate (SDR): {sdr * 100}%')
        landmarks_labels=load_landmark_names(r'Aariz/test/Annotations/Cephalometric Landmarks/Senior Orthodontists/cks2ip8fp29yl0yuf6ry9266i.json')
        calculate_per_landmark_metrics(test_result_dict,landmarks_labels)

        if config.save_image:
            check_and_make_dir(os.path.join(config.save_image_dir,config.model_name))
            visualize_pred_orig(test_result_dict,os.path.join(config.save_image_dir,config.model_name))








if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters | 数据文件路径
    parser.add_argument('--test_json_path', type=str,default='./processed_data/test_index.json')

    # model load dir path | 存放模型的文件夹路径
    parser.add_argument('--load_weight_path', type=str,default='./model/Swin_Unet0/best_model.pth')

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='Swin_Unet')

    # result & save
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--save_image_dir', type=str, default='./visualize')

    experiment_config = parser.parse_args()
    main(experiment_config)

