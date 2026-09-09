import torch.nn as nn

class SCN(nn.Module):
    def __init__(self,in_channels,n_landmarks):
        super(SCN, self).__init__()
        #不断增大感受野，提取更多语义信息
        self.local_conv1=nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.local_conv1_dropout=nn.Dropout2d(p=0.5)
        self.local_conv1_activation=nn.LeakyReLU(0.1,inplace=True)

        self.local_conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.local_conv2_activation=nn.LeakyReLU(0.1,inplace=True)

        self.local_conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.local_conv3_activation=nn.LeakyReLU(0.1,inplace=True)

        #降低分辨率，关注范围更大的语义信息1/2分辨率
        self.local_conv4=nn.AvgPool2d(kernel_size=2,stride=2)
        self.local_conv5=nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2)
        self.local_conv5_dropout=nn.Dropout(p=0.5) #设置0.5抗干扰能力
        self.local_conv5_activation=nn.LeakyReLU(0.1,inplace=True)
        self.local_conv6=nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2)
        self.local_conv6_activation=nn.LeakyReLU(0.1,inplace=True)

        #再降低分辨率，达到1/4分辨率
        self.local_conv7=nn.AvgPool2d(kernel_size=2,stride=2)
        self.local_conv8=nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2)
        self.local_conv8_dropout=nn.Dropout(p=0.5)
        self.local_conv8_activation=nn.LeakyReLU(0.1,inplace=True)
        self.local_conv9=nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2)
        self.local_conv9_activation=nn.LeakyReLU(0.1,inplace=True)

        #1/8分辨率最深层特征
        self.local_conv21=nn.AvgPool2d(kernel_size=2,stride=2)
        self.local_conv22=nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2)
        self.local_conv22_dropout=nn.Dropout(p=0.5)
        self.local_conv22_activation=nn.LeakyReLU(0.1,inplace=True)
        self.local_conv23=nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2)
        self.local_conv23_activation=nn.LeakyReLU(0.1,inplace=True)

        #解码器，恢复分辨率,align_corners=True强制拉起
        self.local_conv24=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.local_conv10=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.local_conv12=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.local_conv14=nn.Conv2d(64,n_landmarks,kernel_size=5,stride=1,padding=2)

        #热力图缩小到原来的1/8
        self.local_conv15=nn.Upsample(scale_factor=1/8,mode='bilinear')
        #连续使用15*15的超大卷积核，用于捕捉远距离的空间以来关系,标记点的相对位置就是
        self.local_conv16=nn.Conv2d(n_landmarks,64,kernel_size=15,stride=1,padding=7)
        self.local_conv16_activation=nn.LeakyReLU(0.1,inplace=True)
        self.local_conv17=nn.Conv2d(64,64,kernel_size=15,stride=1,padding=7)
        self.local_conv17_activation=nn.LeakyReLU(0.1,inplace=True)
        self.local_conv18=nn.Conv2d(64,64,kernel_size=15,stride=1,padding=7)
        self.local_conv18_activation=nn.LeakyReLU(0.1,inplace=True)

        #映射回关键点通道数
        self.local_conv18_reg=nn.Conv2d(64,n_landmarks,kernel_size=15,stride=1,padding=7)
        self.local_conv18_referee=nn.Tanh()#限制输出在[-1,1]之间，作为空间权重

        #将空间权重图放大到8倍，恢复到与局部热力图相同的尺寸
        self.local_conv19=nn.Upsample(scale_factor=8,mode='bicubic')

        self._initialize_weights()

    def forward(self,x):
        x1=self.local_conv1_activation(self.local_conv1_dropout(self.local_conv1(x)))
        x2=self.local_conv2_activation(self.local_conv2(x1))
        x3=self.local_conv3_activation(self.local_conv3(x2))

        x4=self.local_conv4(x3)#256
        x5=self.local_conv5_activation(self.local_conv5_dropout(self.local_conv5(x4)))
        x6=self.local_conv6_activation(self.local_conv6(x5))

        x7=self.local_conv7(x6)#128
        x8=self.local_conv8_activation(self.local_conv8_dropout(self.local_conv8(x7)))
        x9=self.local_conv9_activation(self.local_conv9(x8))#局部细节信息还行

        x21=self.local_conv21(x8)#注意,64
        x22=self.local_conv22_activation(self.local_conv22_dropout(self.local_conv22(x21)))
        x23=self.local_conv23_activation(self.local_conv23(x22))

        x24=self.local_conv24(x23)#向上采样，语义信息强，但是，局部细节信息不行,维度128

        x10=x24+x9#特征融合，跳跃连接

        x10=self.local_conv10(x10)#向上采样，维度256

        x11=x10+x6

        x12=self.local_conv12(x11)#向上采样，维度512

        x13=x12+x3

        x14=self.local_conv14(x13)#输出通道数为标记点数

        x15=self.local_conv15(x14)

        x16=self.local_conv16(self.local_conv16_activation(x15))
        x17=self.local_conv17(self.local_conv17_activation(x16))
        x18=self.local_conv18(self.local_conv18_activation(x17))

        x18_reg=self.local_conv18_referee(self.local_conv18_reg(x18))

        x19=self.local_conv19(x18_reg)#输出空间掩码[-1,1]，筛选

        x20=x19*x14
        return x20
    #该代码科学地为卷积层初始化权重，防止梯度消失
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu',a=0.01)#何恺明算法，防止梯度消失，ReLU家族每一次都会消耗一半能量
                if m.bias is not None:
                        nn.init.constant_(m.bias,0.0)









