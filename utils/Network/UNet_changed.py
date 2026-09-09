import torch.nn as nn
import torch


class LUconv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(LUconv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    def forward(self,x):
        out=self.activation(self.batch_norm(self.conv(x)))
        return out

def make_n_layer(in_channels,depth,double_channel):
    if double_channel:
        layer1=LUconv(in_channels,32*(2**(depth+1)))
        layer2=LUconv(32*(2**(depth+1)),32*(2**(depth+1)))
    else:
         layer1=LUconv(in_channels,32*(2**depth))
         layer2=LUconv(32*(2**depth),32*(2**(depth+1)))
    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self,in_channels,depth):
        super(DownTransition,self).__init__()
        self.ops=make_n_layer(in_channels,depth,False)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.current_depth=depth

    def forward(self,x):
        if self.current_depth==3:
            out=self.ops(x)
            out_before_pool=out
        else:
            out_before_pool=self.ops(x)
            out=self.pool(out_before_pool)
        return out_before_pool,out

class UpTransition(nn.Module):
    def __init__(self,in_channels,out_channels,depth):
        super(UpTransition,self).__init__()
        self.current_depth=depth
        self.up_conv=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2)
        self.ops=make_n_layer(in_channels+out_channels//2,depth,True)

    def forward(self,skip_x,x):
        out_up_cov=self.up_conv(x)
        concat=torch.cat([out_up_cov,skip_x],dim=1)
        out=self.ops(concat)
        return out

class OutputTransition(nn.Module):
    def __init__(self,in_channels,n_landmarks):
        super(OutputTransition,self).__init__()
        self.conv_reg=nn.Conv2d(in_channels=in_channels,out_channels=n_landmarks,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out=self.conv_reg(x)
        out=self.sigmoid(out)
        return out


class UNet_changed(nn.Module):
    def __init__(self,in_channels,n_landmarks):
        super(UNet_changed,self).__init__()
        self.down_str64=DownTransition(in_channels,depth=0)
        self.down_str128=DownTransition(in_channels=64,depth=1)
        self.down_str256=DownTransition(in_channels=128,depth=2)

        self.down_str512=DownTransition(in_channels=256,depth=3)

        self.up_str256=UpTransition(in_channels=512,out_channels=512,depth=2)
        self.up_str128=UpTransition(in_channels=256,out_channels=256,depth=1)
        self.up_str64=UpTransition(in_channels=128,out_channels=128,depth=0)

        self.outlayer=OutputTransition(in_channels=64,n_landmarks=n_landmarks)
        self._initialize_weights()
    def forward(self,x):
        self.skip64,self.out64=self.down_str64(x)
        self.skip128,self.out128=self.down_str128(self.out64)
        self.skip256,self.out256=self.down_str256(self.out128)

        _,self.out512=self.down_str512(self.out256)

        self.up256=self.up_str256(self.skip256,self.out512)
        self.up128=self.up_str128(self.skip128,self.up256)
        self.up64=self.up_str64(self.skip64,self.up128)
        self.out=self.outlayer(self.up64)
        return self.out

    def _initialize_weights(self):
        for m in self.modules():
            # 同时涵盖 普通卷积 和 转置卷积
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)