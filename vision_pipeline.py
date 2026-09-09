# !/usr/bin/env python
# -*- coding:utf-8 -*-
# 功能：对单张LCR头影X光图像（支持 DICOM/NIfTI/PNG）进行关键点预测 + CVM颈椎骨龄分期
# 特性：支持手动输入 Pixel Spacing (物理比例)，输出真实物理毫米坐标 + 可视化图像

import os
import torch
import argparse
import numpy as np
import pandas as pd
import pydicom
import timm
from PIL import Image, ImageDraw
import albumentations as A
from albumentations import ToTensorV2
from safetensors.torch import load_file
import warnings

from utils.model import load_model
from utils.soft_argmax import get_windowed_soft_argmax
from utils.cvm_utils import crop_cvm_from_lcr, ordinal_to_class
from utils.coral import CoralLayer

warnings.filterwarnings('ignore')

# ── 29个标准头影测量点名称 ────────────────────────────────────────
DEFAULT_LANDMARK_NAMES = [
    "A-point (A)", "Anterior Nasal Spine (ANS)", "B-point (B)", "Menton (Me)",
    "Nasion (N)", "Orbitale (Or)", "Pogonion (Pog)", "Posterior Nasal Spine (PNS)",
    "Pronasale (Pn)", "Ramus (R)", "Sella (S)", "Articulare (Ar)",
    "Condylion (Co)", "Gnathion (Gn)", "Gonion (Go)", "Porion (Po)",
    "Lower 2nd PM Cusp Tip (LPM)", "Lower Incisor Tip (LIT)", "Lower Molar Cusp Tip (LMT)",
    "Upper 2nd PM Cusp Tip (UPM)", "Upper Incisor Apex (UIA)", "Upper Incisor Tip (UIT)",
    "Upper Molar Cusp Tip (UMT)", "Lower Incisor Apex (LIA)", "Labrale inferius (Li)",
    "Labrale superius (Ls)", "Soft Tissue Nasion (N`)", "Soft Tissue Pogonion (Pog`)",
    "Subnasale (Sn)"
]

CS_DESCRIPTION = {
    0: "CS1 - 青春期前（Pre-pubertal）",
    1: "CS2 - 青春期前期（Pre-pubertal late）",
    2: "CS3 - 青春期加速（Pubertal acceleration）",
    3: "CS4 - 青春期减速（Pubertal deceleration）",
    4: "CS5 - 青春期后（Post-pubertal early）",
    5: "CS6 - 青春期后成熟（Post-pubertal mature）",
}

# ─────────────────────────────────────────────────────────────────
# 图像读取与预处理部分
# ─────────────────────────────────────────────────────────────────
def load_and_preprocess(image_path: str):
    """加载图像(支持DICOM/PNG)，并预处理为模型输入"""
    try:
        ds = pydicom.dcmread(image_path)
        img_np_16bit = ds.pixel_array.astype(np.float32)
        
        img_np_8bit = (img_np_16bit - img_np_16bit.min()) / (img_np_16bit.max() - img_np_16bit.min()) * 255.0
        img_np_8bit = img_np_8bit.astype(np.uint8)
        
        if len(img_np_8bit.shape) == 2:
            img_np = np.stack([img_np_8bit]*3, axis=-1)
        else:
            img_np = img_np_8bit
            
        pil_img = Image.fromarray(img_np)
        orig_h, orig_w = img_np.shape[:2]
            
    except Exception:
        pil_img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)

    transform = A.Compose([
        A.Resize(512, 512),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transformed = transform(image=img_np)
    tensor = transformed['image'].unsqueeze(0).float()  

    scale_x = 512 / orig_w
    scale_y = 512 / orig_h

    return tensor, pil_img, scale_x, scale_y

def predict_landmarks(model, tensor, device, scale_x, scale_y):
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        heatmaps = model(tensor)[0]
        coords_512 = get_windowed_soft_argmax(heatmaps)

    coords = coords_512.cpu().numpy().astype(np.float32)
    coords[:, 0] /= scale_x
    coords[:, 1] /= scale_y
    return coords

# ─────────────────────────────────────────────────────────────────
# CVM 颈椎分期部分
# ─────────────────────────────────────────────────────────────────
def load_cvm_model(cvm_weight_path: str, device):
    # 【修复1】：使用与训练集完全匹配的架构 (window16, num_classes=0 准备接自定义头)
    model = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=False, num_classes=0, global_pool='avg')
    
    # 【修复2】：注入 CoralLayer
    model.head = CoralLayer(in_features=model.num_features, num_classes=6)
    
    # 读取权重
    if cvm_weight_path.endswith('.safetensors'):
        state_dict = load_file(cvm_weight_path)
    else:
        weight = torch.load(cvm_weight_path, map_location='cpu', weights_only=False)
        state_dict = weight.get('model_state_dict', weight.get('model', weight.get('state_dict', weight)))

    # 清理多卡训练可能带来的 module. 前缀
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 【修复3】：千万不要过滤掉 'head'，否则模型会随机输出！只过滤 attn_mask
    clean_state_dict = {k: v for k, v in clean_state_dict.items() if 'attn_mask' not in k}

    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def predict_cvm(cvm_model, pil_img: Image.Image, coords: np.ndarray, device):
    img_np = np.array(pil_img.convert('RGB'))
    crop_np = crop_cvm_from_lcr(img_np, coords)
    if crop_np.size == 0:
        return None, 'N/A'

    cvm_transform = A.Compose([
        A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1),
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    tensor = cvm_transform(image=crop_np)['image'].unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = cvm_model(tensor)
        cs_stage = int(ordinal_to_class(logits)[0].item())
    return cs_stage, f'CS{cs_stage + 1}'

# ─────────────────────────────────────────────────────────────────
# 输出部分
# ─────────────────────────────────────────────────────────────────
def save_csv(coords: np.ndarray, landmark_names: list, output_csv: str,
             cs_stage: int, cs_label: str, pixel_spacing: list):
    n = coords.shape[0]
    names = landmark_names if len(landmark_names) == n else [f'P{i+1:02d}' for i in range(n)]

    rows = []
    for i, name in enumerate(names):
        pixel_x, pixel_y = float(coords[i, 0]), float(coords[i, 1])
        physical_x_mm = pixel_x * pixel_spacing[1] 
        physical_y_mm = pixel_y * pixel_spacing[0]
        
        rows.append({
            'index'    : f'P{i + 1:02d}',
            'landmark' : name,
            'pixel_x'  : round(pixel_x, 3),
            'pixel_y'  : round(pixel_y, 3),
            'physical_x_mm': round(physical_x_mm, 3),
            'physical_y_mm': round(physical_y_mm, 3),
        })

    df_landmarks = pd.DataFrame(rows)

    cvm_desc = CS_DESCRIPTION.get(cs_stage, 'N/A') if cs_stage is not None else 'N/A'
    cvm_row = pd.DataFrame([{
        'index'    : 'CVM',
        'landmark' : cvm_desc,
        'pixel_x'  : cs_label,  
        'pixel_y'  : cs_stage if cs_stage is not None else 'N/A',
        'physical_x_mm': '-',
        'physical_y_mm': '-'
    }])
    df = pd.concat([df_landmarks, cvm_row], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'\n✅ CSV已保存：{output_csv}')
    return df

def save_visualization(pil_img: Image.Image, coords: np.ndarray, landmark_names: list, 
                       output_img: str, cs_label: str, pixel_spacing: list, radius: int = 8):
    draw_img = pil_img.convert('RGB').copy()
    draw = ImageDraw.Draw(draw_img)
    
    # 仅绘制 29 个特征点和标号 (P01-P29)
    for i, (x, y) in enumerate(coords):
        x, y = float(x), float(y)
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill='red', outline='white', width=2)
        draw.text((x + radius + 2, y - radius), f'P{i+1:02d}', fill='yellow')


    # 保存图片
    os.makedirs(os.path.dirname(output_img) or '.', exist_ok=True)
    draw_img.save(output_img)


# ─────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────
def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{config.cuda_id}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pixel_spacing = config.pixel_spacing
    if len(pixel_spacing) == 1:
        pixel_spacing = [pixel_spacing[0], pixel_spacing[0]]
    elif len(pixel_spacing) > 2:
        pixel_spacing = pixel_spacing[:2]
    
    print(f"📐 使用手动设定的物理比例: Y={pixel_spacing[0]} mm/px, X={pixel_spacing[1]} mm/px")

    landmark_model = load_model(model_name=config.model_name)
    landmark_model.load_state_dict(torch.load(config.load_weight_path, map_location=device, weights_only=False).get('model_state_dict', torch.load(config.load_weight_path, map_location=device)))
    landmark_model.to(device)
    
    cvm_model = load_cvm_model(config.cvm_weight_path, device)

    tensor, pil_img, scale_x, scale_y = load_and_preprocess(config.image_path)
    
    coords = predict_landmarks(landmark_model, tensor, device, scale_x, scale_y)
    cs_stage, cs_label = predict_cvm(cvm_model, pil_img, coords, device)

    base_name = os.path.splitext(os.path.basename(config.image_path))[0]
    csv_path = config.output_csv or os.path.join(config.output_dir, f'{base_name}_landmarks.csv')
    img_path = config.output_img or os.path.join(config.output_dir, f'{base_name}_visualized.png')

    save_csv(coords, DEFAULT_LANDMARK_NAMES, csv_path, cs_stage, cs_label, pixel_spacing)
    if not config.no_visualize:
        save_visualization(pil_img, coords, DEFAULT_LANDMARK_NAMES, img_path, cs_label, pixel_spacing)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,default='./Aariz/test/Cephalograms/cks2ip8fp29yl0yuf6ry9266i.png')
    parser.add_argument('--load_weight_path', type=str, default='./model/Swin_Unet0/best_model.pth')
    parser.add_argument('--model_name', type=str, default='Swin_Unet')
    parser.add_argument('--cvm_weight_path', type=str, default='./model/cvm_model/best_model.pth')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_csv', type=str, default='')
    parser.add_argument('--output_img', type=str, default='')
    parser.add_argument('--no_visualize', action='store_true')
    
    parser.add_argument('--pixel_spacing', type=float, nargs='+', default=[1.0, 1.0],
                        help='手动输入的物理像素比例 [Y轴(行间距), X轴(列间距)]。如果只输入一个值，默认XY相等。例如: --pixel_spacing 0.084 0.084')
    
    cfg = parser.parse_args()
    main(cfg)