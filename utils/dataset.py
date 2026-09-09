import torch
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import numpy as np
from utils.heatmap import generate_heatmap
from utils.cvm_utils import crop_cvm_from_lcr



class AarizDataset(Dataset):
    def __init__(self, json_path, transform=None, target_size=(512, 512)):
        """
        json_path: 预处理生成的 *_index.json 路径
        img_dir: 对应的 Cephalograms 文件夹路径
        transform: 图像增强
        target_size: 训练时的输入尺寸
        """
        with open(json_path, 'r') as f:
            self.data_index = json.load(f)

        self.ceph_ids = list(self.data_index.keys())
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.ceph_ids)

    def __getitem__(self, idx):
        pid = self.ceph_ids[idx]
        item = self.data_index[pid]

        image_path = item.get('image_path')
        #读取图像，怕图像是单通道加上convert
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size[:2]
        image_np = np.array(image)

        # 根据模式返回数据
        data_packet = {'id': pid}
        # 返回原始图像的大小
        data_packet['orig_size'] = torch.tensor([orig_w, orig_h], dtype=torch.float32)

        # 准备标志点和索引
        # landmarks=[]
        # point_indices=[]
        landmarks = np.array(item.get('landmarks'), dtype=np.float32)
        point_indices = list(range(len(landmarks)))

        # 同步缩放坐标,执行变换
        if self.transform:
            transformed = self.transform(
                image=image_np,
                keypoints=landmarks,
                point_indices=point_indices
            )
            transformed_image = transformed.get('image')
            transformed_landmarks = transformed.get('keypoints')

            # 标注坐标归一化处理
            h, w = transformed_image.shape[-2:]
            # transformed_landmarks返回的是一个list
            landmarks_tensor = torch.tensor(transformed_landmarks, dtype=torch.float32)
            # landmarks_tensor[:, 0] /= w
            # landmarks_tensor[:, 1] /= h
            # 保证数值在 [0, 1] 之间，防止 Affine 变换导致的轻微溢出边界,大概率不可能因为29个点不在边缘
            transformed_landmarks = torch.clamp(landmarks_tensor, 0.0, 512.0)
            heatmaps=generate_heatmap(transformed_landmarks, h, w)
        else:

            transformed_image = torch.from_numpy(image_np).permute(2, 0, 1).float()
            h, w = transformed_image.shape[-2:]
            heatmaps=generate_heatmap(transformed_image,h,w)
            # 2. 坐标使用原始宽高进行归一化
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
            # landmarks_tensor[:, 0] /= orig_w
            # landmarks_tensor[:, 1] /= orig_h

            transformed_landmarks = torch.clamp(landmarks_tensor, 0.0, 512.0)

        # 标志点逻辑（29个）
        # 使用as_tensor节省内存
        data_packet['file_path']=image_path
        scale_x=w/orig_w
        scale_y=h/orig_h
        data_packet['scale']=torch.tensor([scale_x, scale_y], dtype=torch.float32)
        data_packet['heatmaps'] = torch.as_tensor(heatmaps, dtype=torch.float32)
        data_packet['landmarks'] = torch.as_tensor(transformed_landmarks, dtype=torch.float32)
        # 保留物理尺寸用于后续计算MRE(mm)
        data_packet['pixel_size'] = torch.tensor(float(item.get('pixel_size')))

        # CVM逻辑（6个阶段）
        data_packet['cvm_label'] = torch.tensor(int(item.get('cvm_stage')), dtype=torch.long)

        return transformed_image, data_packet



class CvmDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data_index = json.load(f)

        self.ceph_ids = list(self.data_index.keys())
        self.transform = transform
    

    def __len__(self):
        return len(self.ceph_ids)

    def __getitem__(self, idx):
        pid = self.ceph_ids[idx]
        item = self.data_index[pid]
        
        # 图像读取与处理
        image_path = item.get('image_path')
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # 裁剪图像
        landmarks = item.get('landmarks')
        crop_image = crop_cvm_from_lcr(image_np, landmarks)

        # 图像数据增强/转换 (并修复了 transform=None 时的报错问题)
        if self.transform:
            transformed = self.transform(image=crop_image)
            transformed_image = transformed.get('image')
        else:
            transformed_image = crop_image

        # 获取标签
        cvm_label = int(item.get('cvm_stage')) - 1
        
        return transformed_image, cvm_label, image_path