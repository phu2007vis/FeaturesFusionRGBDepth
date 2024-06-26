from resources.i3d.pytorch_i3d import InceptionI3d
from resources.s3d.s3d_pytorch import S3D_pytorch
from resources.lstm.lstm_model import LSTMModel
from resources.SwinTransformer.SwinTransformer import SwinTransformer
from resources.i3d.MF_fusion import MF_fusion
from resources.i3d.LF_fusion import LF_fusion


def get_model(name,num_classes,fintuning,in_channles = 3, **kwargs):
    if name == 'i3d':
        model = InceptionI3d(400, in_channels=in_channles)
        model.replace_logits(num_classes)
        model.fintuning(from_layer=fintuning)
        return model
    elif name == 's3d':
        return S3D_pytorch(num_classes)
    elif name == 'lstm':
        return  LSTMModel(num_classes = num_classes, **kwargs)
    elif name.split('-')[0] == 'transformer':
        return SwinTransformer(model_name = name.split('-')[1],num_classes=num_classes)
def get_i3d_depth_model(name,num_classes,fintuning = -1):
    if name == 'i3d':
        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(num_classes)
        model.fintuning(from_layer=fintuning)
       
    elif name == 'lower_fusion':
        model = InceptionI3d(num_classes, in_channels=4)
       
    elif name ==  'middle_fusion':
        model =  MF_fusion(num_classes)
    elif name == 'late_fusion':
        model = LF_fusion(num_classes)
    total_params = 0
    for param in model .parameters():
        total_params += param.numel()
    print(f"Model {name} has {total_params} parameters!")
        
    
   
        
    
