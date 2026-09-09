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
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from utils.dataset import AarizDataset
from utils.model import load_model
from utils.losses import load_loss
from utils.landmarks_utils import check_and_make_dir, calculate_prediction_metrics
from utils.soft_argmax import get_windowed_soft_argmax

warnings.filterwarnings('ignore')


def main(config):
    # === 多卡修改 1：支持识别多个 GPU ===
    gpu_id=config.cuda_ids
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================================
    print(f"检测到可用显卡数量: {torch.cuda.device_count()}")
    # train and valid dataset
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        A.Affine(translate_percent={'x': 0.05, 'y': 0.05}, scale=(0.95, 1.05), rotate=(-5, 5), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['point_indices'], remove_invisible=False))

    train_dataset = AarizDataset(json_path=config.train_json_path, transform=train_transform)

    valid_transform = A.Compose([
        A.Resize(512, 512),
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

    warmup_epoches=10
    

    # === 多卡修改 2：使用 DataParallel 包装模型 ===
    # 自动检测当前可见的 GPU 数量，如果大于 1，则开启多卡模式
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    # ==============================================

    # 定义AdamW优化器，负责更新模型权重
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.lr,
                                  betas=(config.beta1, config.beta2),
                                  weight_decay=1e-2)

    # # 定义学习率调整器，按照步长衰减学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=config.scheduler_step_size,
    #                                             gamma=config.scheduler_gamma)
    scheduler_warmup=LinearLR(
        optimizer,start_factor=0.1,total_iters=warmup_epoches
    )
    scheduler_cosine=CosineAnnealingLR(
        optimizer,T_max=config.train_max_epoch-warmup_epoches,eta_min=1e-7
    )
    scheduler=SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup,scheduler_cosine],
        milestones=[warmup_epoches]
    )
    
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
    resume_path = os.path.join(config.save_model_dir, f'{config.model_name}', 'last_checkpoint.pth')

    if os.path.exists(resume_path):
        logger.info(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        # model.load_state_dict(checkpoint['model_state_dict'])
        if isinstance(model,torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        num_epoch_no_improvement = checkpoint['num_epoch_no_improvement']

        start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        best_mean_dists_error = checkpoint['best_mean_dists_error']
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        check_and_make_dir(os.path.join(config.save_model_dir, f'{config.model_name}'))
        logger.info("No checkpoint found, starting from scratch.")

    # 训练和测试
    for epoch in range(start_epoch, config.train_max_epoch):
        train_loss = []
        valid_loss = []

        model.train()
        for (images, data_packets) in tqdm.tqdm(train_loader):
            images, heapmaps = images.float().to(device), data_packets['heatmaps'].float().to(device)

            output = model(images)
            epoch_threshold=250
            current_pos_weight=50.0 if epoch < epoch_threshold else 15.0
            loss = loss_fn(output, heapmaps,pos_weight=current_pos_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
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
                loss = loss_fn(output_heatmaps, heapmaps,pos_weight=1.0)
                valid_loss.append(loss.item())

                for batch_index in range(output_heatmaps.shape[0]):
                    current_scale = scale[batch_index]
                    current_pixel_size = pixel_size[batch_index]
                    current_file_path = file_path[batch_index]
                    current_landmarks = landmarks[batch_index]
                    current_landmarks = current_landmarks / current_scale

                    predicted_landmarks = get_windowed_soft_argmax(output_heatmaps[batch_index])
                    predicted_landmarks = predicted_landmarks / current_scale

                    valid_result_dict[current_file_path] = {
                        'pixel_size': current_pixel_size,
                        'orig_landmarks': current_landmarks,
                        'predicted_landmarks': predicted_landmarks,
                    }
            mean_radius_dist_error, mean_radius_mm_error, sdr = calculate_prediction_metrics(valid_result_dict)

        valid_loss = np.mean(valid_loss)

        log_msg = (f'Epoch [{epoch + 1}/{config.train_max_epoch}] '
                   f'Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | '
                   f'Mean Radius dist Error (MRE): {mean_radius_dist_error}| Mean Radius mm Error (MRE): {mean_radius_mm_error} | 2mm Success Detection Rate (SDR): {sdr * 100}%'
                   )
        logger.info(log_msg)
        # === 多卡修改 3：保存模型时剥离 module. 前缀 ===
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        # ==============================================
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state, # 这里使用剥离后的权重
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_dists_error': best_mean_dists_error,
                'num_epoch_no_improvement': num_epoch_no_improvement,
                'mre': mean_radius_mm_error,
                'sdr': sdr
            }
        torch.save(checkpoint, os.path.join(config.save_model_dir, f'{config.model_name}', 'last_checkpoint.pth'))
        if (epoch + 1) % config.save_model_step == 0:
            
            torch.save(checkpoint,
                       os.path.join(config.save_model_dir, f'{config.model_name}', f'checkpoint_epoch_{epoch + 1}.pth'))
            logger.info(f'Checkpoint saved at epoch {epoch + 1}')

        if mean_radius_dist_error < best_mean_dists_error:
            logger.info(
                f'Validation loss decreased ({best_mean_dists_error:.6f} --> {mean_radius_dist_error:.6f}). Saving Best Model!')
            best_mean_dists_error = mean_radius_dist_error
            num_epoch_no_improvement = 0
            
            # === 多卡修改 3 (同上)：保存 Best Model ===
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_state,
                       os.path.join(config.save_model_dir, f'{config.model_name}', 'best_model.pth'))
            # =========================================
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

    # === 多卡修改 4：修改传入的参数名为 cuda_ids，并传入两张卡的 ID ===
    parser.add_argument('--cuda_ids', type=str, default='0,1') # 默认使用 0号 和 1号 卡
    # =============================================================

    parser.add_argument('--model_name', type=str, default='Swin_Unet')
    parser.add_argument('--train_max_epoch', type=int, default=400)

    # 提示：多卡训练时，batch_size 是两张卡的总和。
    # 比如这里设置为 8，那么每张卡会分到 4 个样本。你可以根据显存情况适当将其调大（如 16）。
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--batch_size_valid', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--save_model_step', type=int, default=4)

    # data augmentation
    parser.add_argument('--flip_augmentation_prob', type=float, default=0.5)

    # model loss function
    parser.add_argument('--loss_name', type=str, default='focalLoss')

    # early stop mechanism
    parser.add_argument('--epoch_patience', type=int, default=50)

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