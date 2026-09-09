from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn as nn
class FocalHeatmapHead(nn.Module):
    def __init__(self,original_out_block):
        super(FocalHeatmapHead,self).__init__()
        self.original_out=original_out_block
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        logits=self.original_out(x)
        heatmap=self.sigmoid(logits)
        return heatmap
Swin_Unet=SwinUNETR(
    in_channels=3,
    out_channels=29,
    spatial_dims=2
)
Swin_Unet.out=FocalHeatmapHead(Swin_Unet.out)