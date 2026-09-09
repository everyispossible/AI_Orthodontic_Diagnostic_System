# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhanghongyuan2017@email.szu.edu.cn

import os
import logging
import tqdm
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')

from utils.dataset import AarizDataset
from utils.model import load_model
from utils.losses import load_loss

from utils.landmarks_utils import check_and_make_dir, calculate_prediction_metrics
from utils.soft_argmax import get_windowed_soft_argmax


def main(config):
    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # train and valid dataset
    train_transform = A.Compose([
        # 缩放到统一尺度
        # A.LongestMaxSize(max_size=512),
        # A.PadIfNeeded(min_height=512,min_width=512,border_mode=0),
        A.Resize(512, 512),
        # 限制对比度自适应直方图均衡化CLAHE
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        # 针对7种不同设备的图像差异进行亮度对比/对比度增强
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        # 仿射变换,shift_limit平移限制 ，scale缩放范围，rotate旋转范围，p执行概率
        A.Affine(translate_percent={'x': 0.05, 'y': 0.05}, scale=(0.95, 1.05), rotate=(-5, 5), p=0.3),
        # 标准化 (ImageNet 统计值)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # 转换为 PyTorch 张量
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['point_indices'], remove_invisible=False))

    train_dataset = AarizDataset(json_path=config.train_json_path, transform=train_transform)

    valid_transform = A.Compose([
        # A.LongestMaxSize(max_size=512),
        # A.PadIfNeeded(min_height=512,min_width=512,border_mode=0),
        A.Resize(512, 512),
        # 验证集也开启CLAE保证数据集分布一致
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['point_indices'], remove_invisible=False))
    valid_dataset = AarizDataset(json_path=config.valid_json_path, transform=valid_transform)

    # train and valid dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size_valid,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=config.num_workers)

    # 加载模型
    model = load_model(model_name=config.model_name)
    model = model.to(device)

    # 定义AdamW优化器，负责更新模型权重
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.lr,
                                  betas=(config.beta1, config.beta2),
                                  weight_decay=1e-2)

    # 定义学习率调整器，按照步长衰减学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.scheduler_step_size,
                                                gamma=config.scheduler_gamma)

    # 加载损失函数
    loss_fn = load_loss(config.loss_name)

    best_mean_dists_error = float('inf')
    num_epoch_no_improvement = 0

    # 日志文件
    log_file = os.path.join('./log', f'{config.model_name}_Train_log.txt')

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()
                                  ]
                        )
    logger = logging.getLogger(__name__)

    start_epoch = 0
    resume_path = os.path.join(config.save_model_dir,f'{config.model_name}', 'last_checkpoint.pth')

    if os.path.exists(resume_path):
        logger.info(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        num_epoch_no_improvement=checkpoint['num_epoch_no_improvement']

        start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        best_mean_dists_error = checkpoint['best_mean_dists_error']
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        check_and_make_dir(os.path.join(config.save_model_dir,f'{config.model_name}'))
        logger.info("No checkpoint found, starting from scratch.")

    # 训练和测试
    for epoch in range(start_epoch, config.train_max_epoch):
        train_loss = []
        valid_loss = []

        model.train()
        for (images, data_packets) in tqdm.tqdm(train_loader):
            images, heapmaps = images.float().to(device), data_packets['heatmaps'].float().to(device)
           

           
            output = model(images)
            loss = loss_fn(output, heapmaps)

            # # === 新增：打印与检查 Loss ===
            # current_loss = loss.item() # 提取具体的数字
            
            # # 1. 普通打印（如果你想看每一步的变化，可以取消下面这行的注释）
            # # print(f"Current Batch Loss: {current_loss}")
            
            # # 2. 异常拦截（强烈推荐！专门抓 NaN 或 Inf）
            # import math
            # if math.isnan(current_loss) or math.isinf(current_loss):
            #     print("\n🚨 警告：发现异常 Loss (NaN 或 Inf)！")
            #     print("模型输出的最大值:", fp32_output.max().item())
            #     print("模型输出的最小值:", fp32_output.min().item())
            #     break # 发现异常直接打断当前循环，方便排查

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        with torch.no_grad():
            model.eval()
            print('Validating...')
            valid_result_dict = {}
            for (images, data_packets) in tqdm.tqdm(valid_loader):
                images, heapmaps = images.float().to(device), data_packets['heatmaps'].float().to(device)
                landmarks = data_packets['landmarks'].float().to(device)  # [B,29,2]
                scale = data_packets['scale'].float().to(device)  # [B]
                pixel_size = data_packets['pixel_size'].float().to(device)  # [B]
                file_path = data_packets['file_path']  # [B]

                output_heatmaps = model(images)
                loss = loss_fn(output_heatmaps, heapmaps)
                valid_loss.append(loss.item())



                for batch_index in range(output_heatmaps.shape[0]):
                    #当前图像的缩放因子新图/旧图
                    current_scale = scale[batch_index]
                    #当前图像的物理像素尺寸
                    current_pixel_size = pixel_size[batch_index]
                    #当前图像的存储地址
                    current_file_path = file_path[batch_index]
                    #当前图像的标注的29个标记点坐标
                    current_landmarks = landmarks[batch_index]
                    #切记转换大小尺度
                    current_landmarks=current_landmarks/current_scale
            
                    #用于存储当前图像的29个预测标记点
                    predicted_landmarks=get_windowed_soft_argmax(output_heatmaps[batch_index])

                    predicted_landmarks=predicted_landmarks/current_scale
                    
                    valid_result_dict[current_file_path] = {
                        'pixel_size': current_pixel_size,
                        'orig_landmarks': current_landmarks,
                        'predicted_landmarks': predicted_landmarks,
                    }
            mean_radius_dist_error, mean_radius_mm_error, sdr = calculate_prediction_metrics(valid_result_dict)

        #             img_predicts_pts = []
        #             pred_landmarks = []
        #             current_scale_up = scale_up[batch_index]
        #             output_heatmap = output_heatmaps[batch_index]
        #             for i in range(output_heatmap.shape[0]):
        #                 pre_heatmap = output_heatmap[i]
        #                 yy, xx = np.where(pre_heatmap == np.max(pre_heatmap))
        #                 x0, y0 = np.mean(xx), np.mean(yy)
        #
        #                 pred_landmarks.append([x0, y0])
        #
        #             pred_landmarks = np.array(pred_landmarks)
        #             # 每张图像的29个点的分别的欧几里得距离
        #             dists = np.linalg.norm(pred_landmarks - landmarks[batch_index], axis=1) * current_scale_up
        #             # 每张图像的29个点的实际距离差
        #             error_mm = dists * scale[batch_index]
        #             # 计算每张图像中实际误差小于2.0mm的点
        #             sdr_num = sdr_num + np.sum(error_mm < 2.0)
        #             # 将每张图像的每个点误差添加，算整个验证集的
        #             all_errors_dist.append(dists)
        #             all_errors_mm.append(error_mm)
        # # 转换成numpy
        # all_errors_dist = np.array(all_errors_dist)
        # all_errors_mm = np.array(all_errors_mm)
        #
        # avg_dist = np.mean(all_errors_dist)  # 平均像素误差
        # avg_mre = np.mean(all_errors_mm)  # 平均物理误差
        # # 计算SDR整个验证集的
        # sdr_2mm = sdr_num / (len(valid_loader.dataset) * num_landmarks) * 100  # 整个验证集中所有点的误差小于2mm的占比

        valid_loss = np.mean(valid_loss)

        log_msg = (f'Epoch [{epoch + 1}/{config.train_max_epoch}] '
                   f'Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | '
                   f'Mean Radius dist Error (MRE): {mean_radius_dist_error}| Mean Radius mm Error (MRE): {mean_radius_mm_error} | 2mm Success Detection Rate (SDR): {sdr * 100}%'
                   )
        logger.info(log_msg)

        if (epoch + 1) % config.save_model_step == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_dists_error': best_mean_dists_error,
                'num_epoch_no_improvement':num_epoch_no_improvement,
                'mre': mean_radius_mm_error,
                'sdr': sdr
            }
            torch.save(checkpoint, os.path.join(config.save_model_dir,f'{config.model_name}', 'last_checkpoint.pth'))
            torch.save(checkpoint, os.path.join(config.save_model_dir,f'{config.model_name}',f'checkpoint_epoch_{epoch + 1}.pth'))
            logger.info(f'Checkpoint saved at epoch {epoch + 1}')

        if mean_radius_dist_error < best_mean_dists_error:
            logger.info(f'Mean dists loss decreased ({best_mean_dists_error:.6f} --> {mean_radius_dist_error:.6f}). Saving Best Model!')
            best_mean_dists_error = mean_radius_dist_error 
            num_epoch_no_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.save_model_dir,f'{config.model_name}', 'best_model.pth'))
        else:
            num_epoch_no_improvement += 1
            logger.info(f'No improvement for {num_epoch_no_improvement} epochs.')

        scheduler.step()  # 更新学习率

        if num_epoch_no_improvement == config.epoch_patience:
            logger.info('Early stopping triggered. Training finished.')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters | 数据文件路径
    parser.add_argument('--train_json_path', type=str, default='./processed_data/train_index.json')
    parser.add_argument('--valid_json_path', type=str, default='./processed_data/valid_index.json')

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model training hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='UNet_SCN')
    parser.add_argument('--train_max_epoch', type=int, default=400)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_valid', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--save_model_step', type=int, default=2)

    # data augmentation
    parser.add_argument('--flip_augmentation_prob', type=float, default=0.5)

    # model loss function
    parser.add_argument('--loss_name', type=str, default='focalLoss')

    # early stop mechanism
    parser.add_argument('--epoch_patience', type=int, default=10 )

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-4)

    # Adam optimizer parameters
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Step scheduler parameters
    parser.add_argument('--scheduler_step_size', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)

    # result & save
    parser.add_argument('--save_model_dir', type=str, default='./model/')

    experiment_config = parser.parse_args()
    main(experiment_config)
  
 