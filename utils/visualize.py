
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import os
import numpy as np
from torchvision import transforms
import torch
def visualize_landmarks(image_tensor,landmarks):
    if image_tensor.dim()==4:
        image_tensor=image_tensor[0]
    image_to_convert=image_tensor.cpu().detach()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # 逆运算：原图 = 归一化图 * std + mean
    image_to_convert = image_to_convert * std + mean
    
    # 极其重要：把计算后可能微小越界的数值，强行限制在 0.0 到 1.0 之间
    image_to_convert = torch.clamp(image_to_convert, 0.0, 1.0)
    to_pil=transforms.ToPILImage()
    image_pil=to_pil(image_to_convert)
    draw=ImageDraw.Draw(image_pil)
    w,h=image_pil.size[:2]
    radius=2
    for i in range(landmarks.shape[0]):
        x,y=landmarks[i,0],landmarks[i,1]
        draw.ellipse([x-radius,y-radius,x+radius,y+radius],fill=(0, 255, 0), outline=(0, 255, 0))
    return image_pil


def visualize_pred_orig(result_dict: dict,save_image_dir:str):
    for file_path,landmarks_dict in result_dict.items():
        landmarks,predicted_landmarks=landmarks_dict['orig_landmarks'],landmarks_dict['predicted_landmarks']
        image=Image.open(file_path).convert('RGB')
        draw=ImageDraw.Draw(image)
        w,h=image.size[:2]
        radius=7
        line_width=5
        for i  in range(landmarks.shape[0]):
            gt_x,gt_y=landmarks[i,0],landmarks[i,1]
            pr_x,pr_y=predicted_landmarks[i,0],predicted_landmarks[i,1]
            #绘制真实标注点
            draw.ellipse([gt_x-radius,gt_y-radius,gt_x+radius,gt_y+radius],(255,0,0))
            #绘制预测标注点
            draw.ellipse([pr_x-radius,pr_y-radius,pr_x+radius,pr_y+radius],(0,255,0))
            #绘制俩点连线
            draw.line([(gt_x,gt_y),(pr_x,pr_y)],fill=(255,255,0),width=line_width)

        filename=os.path.basename(file_path)
        image.save(os.path.join(save_image_dir,filename))


def visualize_heatmap(image_tensor,heatmap_tensor,point_idx=None,alpha=0.5):
    """
    将热力图叠加在原图上进行可视化
    Args:
        image_tensor: 原图张量
        heatmap_tensor: 预测热力图
        point_idx: 选定标注点，None显示所有标注点
        alpha: 热力图透明度

    """
    if image_tensor.dim()==4:
        image_tensor=image_tensor[0]
    if heatmap_tensor.dim()==4:
        heatmap_tensor=heatmap_tensor[0]

    img=image_tensor.detach().cpu().numpy().transpose(1,2,0)
    img=np.clip(img,0,1)
    if img.shape[-1]==1:
        img=img.squeeze(-1)

    #热力图处理
    hm=heatmap_tensor.detach().cpu().numpy()
    if point_idx is not None:
        heatmap_to_show=hm[point_idx]
        title=f'Landmarks {point_idx} Heatmap'
    else:
        heatmap_to_show=np.max(hm,axis=0)
        title=f'All 29 Landmarks Heatmap'

    heatmap_to_show=(heatmap_to_show-heatmap_to_show.min())/(heatmap_to_show.max()-heatmap_to_show.min())
    fig,ax=plt.subplots(figsize=(10,10))
    ax.imshow(img,cmap='gray')
    #叠加热力图，数值越靠近1越红，越接近0越蓝
    im=ax.imshow(heatmap_to_show,cmap='jet',alpha=alpha)

    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    plt.title(title,fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    return fig