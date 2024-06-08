from torchvision.models.video import s3d,S3D_Weights
import torch.nn as nn


S3D_Weights
class S3D_pytorch(nn.Module):
    def __init__(self,
                 num_classes,
                 dropout = 0.2):
        super(S3D_pytorch,self).__init__()
        self.net = s3d(
                weights = S3D_Weights.KINETICS400_V1
        )
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),
        )
    def forward(self,x):
        return self.net(x)
if __name__ == "__main__":
    model = S3D_pytorch(120)
    import torch
    x = torch.randn(2,3,70,224,224)
    print(model(x).shape)