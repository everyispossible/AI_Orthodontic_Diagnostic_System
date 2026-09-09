import os
import argparse
import numpy as np
import timm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import logging
import tqdm
import json
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler
import cv2
from utils.dataset import CvmDataset
from utils.landmarks_utils import check_and_make_dir
from utils.cvm_utils import get_ordinal_labels,ordinal_to_class

from utils.coral import CoralLayer

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 数据增强：新增亮度/对比度随机变化 ---
    train_transform = A.Compose([
        A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(p=0.5), # 针对 X 光片曝光差异的增强
        A.Affine(translate_percent={'x': 0.05, 'y': (-0.15, 0.15)}, scale=(0.9, 1.1), rotate=(-7, 7), p=0.8),
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0, p=0.7),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1.0),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # --- 优化 JSON 加载与采样器 ---
    with open(config.train_json_path, 'r') as f:
        train_data_dict = json.load(f)
    train_labels = [int(item['cvm_stage']) - 1 for item in train_data_dict.values()]
    
    class_counts = np.bincount(train_labels, minlength=6)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float).clamp(min=1)
    samples_weights = weights[train_labels]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_dataset = CvmDataset(config.train_json_path, transform=train_transform)
    valid_dataset = CvmDataset(config.valid_json_path, transform=valid_transform)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=config.train_batch_size, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=config.num_workers)

    check_and_make_dir('./log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('./log/cvm_train.log'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    # --- 模型初始化 ---
    model = timm.create_model(
        'swinv2_base_window16_256.ms_in1k', 
        pretrained=False, 
        num_classes=0, 
        global_pool='avg', 
        drop_rate=0.3
    )
    
    logger.info(f"正在加载基础权重: {config.local_weights_path}")
    state_dict = load_file(config.local_weights_path) if config.local_weights_path.endswith('.safetensors') else torch.load(config.local_weights_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    clean_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k and 'head' not in k}
    model.load_state_dict(clean_dict, strict=False)
    
    # 注入 CORAL 层
    model.head = CoralLayer(in_features=model.num_features, num_classes=6)
    model = model.to(device)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    total_epochs = config.num_epochs
    warmup_epochs = 15
    best_acc, start_epoch, num_no_improve = 0., 0, 0
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- 断点恢复逻辑 ---
    resume_path = os.path.join(config.save_model_dir, 'cvm_model', 'last_checkpoint.pth')
    if os.path.exists(resume_path):
        logger.info(f"♻️ 恢复训练...")
        ckpt = torch.load(resume_path, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        num_no_improve = ckpt['num_epoch_no_improvement']
        (model.module if hasattr(model, 'module') else model).load_state_dict(ckpt['model_state_dict'])
        
        if start_epoch >= warmup_epochs:
            for p in model.parameters(): p.requires_grad = True
            # 断点恢复时，也使用降低后的学习率 2e-5
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
        else:
            for p in model.parameters(): p.requires_grad = False
            m_head = (model.module.head if hasattr(model, 'module') else model.head)
            for p in m_head.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs)
        
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    else:
        logger.info("🆕 开启第一阶段：冻结主干。")
        check_and_make_dir(os.path.join(config.save_model_dir, 'cvm_model'))
        for p in model.parameters(): p.requires_grad = False
        m_head = (model.module.head if hasattr(model, 'module') else model.head)
        for p in m_head.parameters(): p.requires_grad = True
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs)

    # --- 训练循环 ---
    for epoch in range(start_epoch, total_epochs):
        if epoch == warmup_epochs:
            logger.info("\n🚀 触发切换：进入全局全微调阶段！(开启 2e-5 慢火炖煮模式)")
            for p in model.parameters(): p.requires_grad = True
            # 【关键修改】全局微调学习率下调至 2e-5
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

        # Train Step
        model.train()
        train_losses = []
        for imgs, labels, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            targets = get_ordinal_labels(labels, 6).to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation Step
        model.eval()
        val_losses = []
        correct, total = 0, 0
        all_preds, all_labels, all_probs, all_paths = [], [], [], []
        with torch.no_grad():
            for imgs, labels, paths in tqdm.tqdm(valid_loader, desc=f"Epoch {epoch+1} Valid"):
                imgs, labels = imgs.to(device), labels.to(device)
                targets = get_ordinal_labels(labels, 6).to(device)
                
                logits = model(imgs)
                v_loss = criterion(logits, targets)
                val_losses.append(v_loss.item())
                
                preds = ordinal_to_class(logits)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append((torch.sigmoid(logits) > 0.5).int())
                all_paths.extend(paths)

        # 汇总指标
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        acc = 100 * correct / total
        
        # 异常序列检测
        epoch_probs = torch.cat(all_probs, dim=0)
        bad_mask = ((epoch_probs[:, :-1] - epoch_probs[:, 1:]) < 0).any(dim=1)
        if bad_mask.any():
            logger.warning(f"🚨 发现 {bad_mask.sum().item()} 个异常序列！")

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | Acc: {acc:.2f}%")

        # 保存断点
        m_state = (model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
        checkpoint = {
            'epoch': epoch, 'model_state_dict': m_state, 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_acc': best_acc, 'num_epoch_no_improvement': num_no_improve
        }
        torch.save(checkpoint, os.path.join(config.save_model_dir, 'cvm_model', 'last_checkpoint.pth'))

        # 最好成绩处理
        if acc > best_acc:
            logger.info(f"🏆 最好成绩更新: {best_acc:.2f}% -> {acc:.2f}%")
            best_acc, num_no_improve = acc, 0
            torch.save(m_state, os.path.join(config.save_model_dir, 'cvm_model', 'best_model.pth'))
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'CS{i+1}' for i in range(6)], yticklabels=[f'CS{i+1}' for i in range(6)])
            check_and_make_dir(os.path.join(config.save_model_dir, 'confusion_matrices'))
            plt.savefig(os.path.join(config.save_model_dir, 'confusion_matrices', f'cm_epoch_{epoch+1}.png'))
            plt.close()
        else:
            num_no_improve += 1
            # 【关键修改】耐心值延长至 25 轮
            if num_no_improve >= 25: 
                logger.info(f'连续 {num_no_improve} 轮无提升，训练结束！')
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_path', type=str, default='./processed_data/train_index.json')
    parser.add_argument('--valid_json_path', type=str, default='./processed_data/valid_index.json')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--local_weights_path', type=str, default='./model/model.safetensors')
    parser.add_argument('--save_model_dir', type=str, default='./model')
    config = parser.parse_args()
    main(config)
