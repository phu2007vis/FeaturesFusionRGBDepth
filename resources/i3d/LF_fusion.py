import torch.nn as nn
from resources.i3d.pytorch_i3d import InceptionI3d
import torch

class LF_fusion(nn.Module):
    def __init__(self,num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.rgb_branch = InceptionI3d(in_channels=3,num_classes=num_classes)
        self.depth_branch = InceptionI3d(in_channels=1 ,pretrained=False,num_classes=num_classes)
        self.dropout  = nn.Dropout(0.35)
        
    def forward(self,inputs):
        
        rgb,depth = inputs
        rgb_feature_logits = self.rgb_branch(rgb).unsqueeze(-1)
        depth_feature_logits = self.depth_branch(depth).unsqueeze(-1)
        logits = torch.cat([rgb_feature_logits,depth_feature_logits],dim  = -1)
        return torch.mean(logits,-1).squeeze()

if __name__ == "__main__":
    model = LF_fusion(300)
    rgb = torch.randn(2,3,64,224,224)
    depth = torch.rand(2,1,64,224,224)
    inputs = (rgb,depth)
    print(model(inputs).shape)
    
        
        