from resources.i3d.pytorch_i3d import InceptionI3d
from resources.s3d.s3d_pytorch import S3D_pytorch
from resources.lstm.lstm_model import LSTMModel


def get_model(name,num_classes, **kwargs):
    if name == 'i3d':
        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(num_classes)
        return model
    elif name == 's3d':
        return S3D_pytorch(num_classes)
    elif name == 'lstm':
        return  LSTMModel(num_classes = num_classes, **kwargs)
    
