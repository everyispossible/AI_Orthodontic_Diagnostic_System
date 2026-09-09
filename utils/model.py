from utils.Network.UNet_changed import UNet_changed
from utils.Network.UNet import UNet
from utils.Network.SCN import SCN
from utils.Network.UNet_SCN import UNet_SCN
from utils.Network.Swin_Unet import Swin_Unet

def load_model(model_name):
    if model_name=='UNet':
        model=UNet(in_channels=3,n_landmarks=29)
    elif model_name=='SCN':
        model=SCN(in_channels=3,n_landmarks=29)
    elif model_name=='UNet_changed':
        model=UNet_changed(in_channels=3,n_landmarks=29)
    elif model_name=='Swin_Unet':
        model=Swin_Unet
    elif model_name=='UNet_SCN':
        model=UNet_SCN(in_channels=3,n_landmarks=29)
    else:
        raise ValueError('Please input valid model name, {} not in model zones.'.format(model_name))
    return model

if __name__ == '__main__':
    model = load_model('Swin_Unet')
    print(model)