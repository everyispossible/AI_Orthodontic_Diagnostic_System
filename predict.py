import argparse
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
from utils.model import load_model
import torch
from utils.visualize import visualize_heatmap,visualize_landmarks
from utils.soft_argmax import get_windowed_soft_argmax

def main(config):
    save_dir = './visualization'
    os.makedirs(save_dir, exist_ok=True)
    model=load_model(model_name=config.model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(config.load_model_path, map_location=device))
    model.to(device)
    image_path=config.image_path
    image=Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    transform=A.Compose([
        A.Resize(512,512),
        A.CLAHE(clip_limit=2.0,tile_grid_size=(8,8),p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    augmented=transform(image=image_np)
    image_tensor=augmented['image']
    image_tensor=image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        heatmap_tensor=model(image_tensor)
        predicted=get_windowed_soft_argmax(heatmap_tensor[0])
        heatmap_fig=visualize_heatmap(image_tensor,heatmap_tensor)
        pred_map=visualize_landmarks(image_tensor,predicted)
        heatmap_path=os.path.join(save_dir,'heatmap.png')
        heatmap_fig.savefig(heatmap_path,dpi=300,bbox_inches='tight')
        plt.close(heatmap_fig)
        pred_map_path=os.path.join(save_dir,'pred_map.png')
        pred_map.save(pred_map_path)
        plt.close(heatmap_fig)
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--model_name',type=str,default='Swin_Unet')
    parser.add_argument('--load_model_path',type=str,default='model/Swin_Unet/best_model.pth')
    parser.add_argument('--image_path',type=str,default='Aariz/test/Cephalograms/cks2ip8fq2a1d0yuf82ote2y7.png')
    experiment_config = parser.parse_args()
    main(experiment_config)