import torch
import numpy as np

def crop_cvm_from_lcr(image_np, landmarks):
    landmarks = np.array(landmarks)
    Po_index = 15
    Go_index = 14
    H_orig, W_orig = image_np.shape[:2]
    

    Po_y = landmarks[Po_index, 1]
    Go_x = landmarks[Go_index, 0]

    # 左边缘和下边缘
    x1 = int(0)
    y2 = int(H_orig)

    # 上边缘和右边缘 (根据关键点)
    y1 = int(Po_y)
    x2 = int(Go_x)

    # 原始裁剪
    raw_crop = image_np[y1:y2+ 1, x1:x2 + 1,:]

    return raw_crop

def get_ordinal_labels(labels,num_classes):
    ordinal_labels = []
    for label in labels:
        layers=[1]*label.item()+[0]*(num_classes - label.item()-1)
        ordinal_labels.append(layers)
    return torch.tensor(ordinal_labels,dtype=torch.float)

def ordinal_to_class(logits):
    probs=torch.sigmoid(logits)
    preds=(probs>0.5).sum(dim=1)
    return preds

