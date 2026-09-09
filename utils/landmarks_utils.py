# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import shutil
import numpy as np
import glob
import SimpleITK as sitk
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
import json

def check_and_make_dir(dir_path: str) -> None:
    """
    function to create a new folder, if the folder path dir_path in does not exist
    :param dir_path: folder path | 文件夹路径
    :return: None
    """
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError('Error, the provided path (%s) is a file path, not a folder path.' % dir_path)
        shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

def load_landmark_names(annotation_json_path: str) -> list:
    """
    从 Aariz 数据集的任意一个标注 JSON 文件中，按顺序提取全部关键点名称。
 
    标注文件格式
    -----------
    {
        "landmarks": [
            {"title": "A-point",  "symbol": "A",   "value": {"x": ..., "y": ...}},
            {"title": "Nasion",   "symbol": "N",   "value": {...}},
            ...
        ]
    }
 
    每个关键点的名称组合为 "Title (Symbol)"，例如 "A-point (A)"。
    文件内 landmarks 列表的顺序与数据集坐标张量的通道顺序完全一致，直接按序读取即可。
 
    参数
    ----
    annotation_json_path : str
        指向任意一个标注 JSON 文件的路径。
        典型路径：
            ./Aariz/test/Annotations/Cephalometric Landmarks/
                Senior Orthodontists/cks2ip8fp29yl0yuf6ry9266i.json
 
        也支持传入标注文件所在的 **目录**，此时自动取目录下第一个 .json 文件。
 
    返回
    ----
    list[str]
        长度 == 关键点数量，每个元素格式为 "Title (Symbol)"。
        示例（29 点数据集）：
            ['A-point (A)', 'Anterior Nasal Spine (ANS)', 'B-point (B)', ...]
 
    异常
    ----
    FileNotFoundError : 路径不存在或目录下无 .json 文件
    ValueError        : 文件中未找到 'landmarks' 字段
    """
    path = annotation_json_path
 
    # 若传入的是目录，自动取第一个 .json 文件
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.json")))
        if not candidates:
            raise FileNotFoundError(f"目录下未找到任何 .json 标注文件：{path}")
        path = candidates[0]
 
    if not os.path.isfile(path):
        raise FileNotFoundError(f"标注文件不存在：{path}")
 
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    landmarks = data.get("landmarks")
    if not landmarks:
        raise ValueError(f"标注文件中未找到 'landmarks' 字段：{path}")
 
    names = [f"{lm['title']} ({lm['symbol']})" for lm in landmarks]
 
    print(f"✅ 成功读取 {len(names)} 个关键点名称（来源：{os.path.basename(path)}）")
    for i, name in enumerate(names, 1):
        print(f"   P{i:02d}  {name}")
 
    return names


def load_train_stack_data(file_path: str) -> np.ndarray:
    """
    function to load train_stack.mha data file | 加载train_stack.mha数据文件的函数
    :param file_path: train_stack.mha filepath | 挑战赛提供的train_stack.mha文件路径
    :return: a 4-dim array containing 400 training set cephalometric images | 一个包含了400张训练集头影图像的四维的矩阵
    """
    sitk_stack_image = sitk.ReadImage(file_path)
    np_stack_array = sitk.GetArrayFromImage(sitk_stack_image)
    return np_stack_array


def remove_zero_padding(image_array: np.ndarray) -> np.ndarray:
    """
    function to remove zero padding in an image | 去除图像中的0填充函数
    :param image_array: one cephalometric image array, shape is (2400, 2880, 3) | 一张头影图像的矩阵，形状为(2400, 2880, 3)
    :return: image matrix after removing zero padding | 去除零填充部分的图像矩阵
    """
    row=np.sum(image_array,axis=(1,2))
    col=np.sum(image_array,axis=(0,2))

    non_zero_row_indices=np.argwhere(row!=0)
    non_zero_col_indices=np.argwhere(col!=0)

    last_row=int(non_zero_row_indices[-1])
    last_col=int(non_zero_col_indices[-1])

    image_array=image_array[:last_row+1,:last_col+1]
    return image_array







def calculate_prediction_metrics(result_dict: dict):
    """
    function to calculate prediction metrics | 计算评价指标
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
    :return: MRE and 2mm SDR metrics
    """
    n_landmarks=0
    sdr_landmarks=0
    n_landmarks_error=0
    dist_error=0

    for file_path,landmarks in result_dict.items():
        scale=landmarks['pixel_size']
        landmarks,predicted_landmarks=landmarks['orig_landmarks'],landmarks['predicted_landmarks']

        n_landmarks=n_landmarks+landmarks.shape[0]

        dist_landmarks_error=torch.sqrt(torch.sum(torch.square(landmarks-predicted_landmarks),dim=1))
        each_landmarks_error=dist_landmarks_error*scale

        dist_error=dist_error+torch.sum(dist_landmarks_error)
        n_landmarks_error=n_landmarks_error+torch.sum(each_landmarks_error)

        sdr_landmarks=sdr_landmarks+torch.sum(each_landmarks_error<2)

    mean_radius_dist_error=dist_error/n_landmarks
    mean_radius_mm_error=n_landmarks_error/n_landmarks
    sdr=sdr_landmarks/n_landmarks


    return mean_radius_dist_error,mean_radius_mm_error,sdr



from utils.soft_argmax import get_windowed_soft_argmax  # noqa: F401

def calculate_per_landmark_metrics(result_dict: dict, landmark_names: list = None) -> dict:
    """
    计算测试集上每个关键点的三项核心指标：
      - mean_px  : 平均像素误差
      - mean_mm  : 平均实际误差 (mm)
      - sdr_2mm  : 实际误差 < 2mm 的占比
 
    参数
    ----
    result_dict : dict
        {
            file_path: {
                'pixel_size'         : Tensor / float   (mm/pixel)
                'orig_landmarks'     : Tensor [N, 2]    (已还原到原图尺度)
                'predicted_landmarks': Tensor [N, 2]    (已还原到原图尺度)
            }
        }
    landmark_names : list[str], 可选
        长度 == N 的关键点名称列表，由 load_landmark_names() 提供。
        若为 None 或长度不匹配，自动生成 ['P01', 'P02', ...] 。
 
    返回
    ----
    dict:
        {
            'names'   : list[str]   # 关键点名称，长度 N
            'mean_px' : np.ndarray  # 每点平均像素误差，形状 [N]
            'mean_mm' : np.ndarray  # 每点平均实际误差 (mm)，形状 [N]
            'sdr_2mm' : np.ndarray  # 每点 2mm 成功率（0~1），形状 [N]
        }
    """
    all_px_errors = []   # list of np.ndarray [N]
    all_mm_errors = []   # list of np.ndarray [N]
 
    for _, entry in result_dict.items():
        pixel_size = entry['pixel_size']
        orig_lm    = entry['orig_landmarks']
        pred_lm    = entry['predicted_landmarks']
 
        if hasattr(orig_lm,    'cpu'): orig_lm    = orig_lm.detach().cpu().numpy()
        if hasattr(pred_lm,    'cpu'): pred_lm    = pred_lm.detach().cpu().numpy()
        if hasattr(pixel_size, 'cpu'): pixel_size = float(pixel_size.detach().cpu().numpy())
        else:                          pixel_size = float(pixel_size)
 
        orig_lm = np.array(orig_lm, dtype=np.float32)
        pred_lm = np.array(pred_lm, dtype=np.float32)
 
        px_err = np.linalg.norm(pred_lm - orig_lm, axis=1)   # [N]
        mm_err = px_err * pixel_size                           # [N]
 
        all_px_errors.append(px_err)
        all_mm_errors.append(mm_err)
 
    if not all_mm_errors:
        raise ValueError("result_dict 为空，无法计算逐点指标。")
 
    px_mat = np.stack(all_px_errors, axis=0)   # (M, N)
    mm_mat = np.stack(all_mm_errors, axis=0)   # (M, N)
    num_lm = mm_mat.shape[1]
 
    mean_px = px_mat.mean(axis=0)              # [N]
    mean_mm = mm_mat.mean(axis=0)              # [N]
    sdr_2mm = (mm_mat < 2.0).mean(axis=0)     # [N]
 
    if landmark_names is None or len(landmark_names) != num_lm:
        if landmark_names is not None:
            print(f"⚠️  名称数 ({len(landmark_names)}) ≠ 关键点数 ({num_lm})，使用通用编号。")
        landmark_names = [f'P{i+1:02d}' for i in range(num_lm)]
 
    # 终端打印汇总表
    sep = '─' * 68
    print(f'\n{sep}')
    print(f'{"ID":<5} {"Landmark":<35} {"MRE(px)":>8} {"MRE(mm)":>8} {"SDR<2mm":>8}')
    print(sep)
    for i in range(num_lm):
        print(f'P{i+1:02d}  {landmark_names[i]:<35s}'
              f'{mean_px[i]:>8.2f} {mean_mm[i]:>8.3f} {sdr_2mm[i]*100:>7.1f}%')
    print(sep)
    print(f'{"ALL":<5} {"Global Average":<35}'
          f'{mean_px.mean():>8.2f} {mean_mm.mean():>8.3f} {sdr_2mm.mean()*100:>7.1f}%')
    print(f'{sep}\n')
 
    return {
        'names'   : landmark_names,
        'mean_px' : mean_px,
        'mean_mm' : mean_mm,
        'sdr_2mm' : sdr_2mm,
    }




