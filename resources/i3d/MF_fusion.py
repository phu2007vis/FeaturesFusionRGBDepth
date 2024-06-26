import torch.nn as nn
from resources.i3d.pytorch_i3d import InceptionI3d,Unit3D
import torch

class MF_fusion(nn.Module):
    def __init__(self,num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.rgb_branch = InceptionI3d(in_channels=3,final_endpoint = 'Mixed_5c')
        self.depth_branch = InceptionI3d(in_channels=1 ,final_endpoint = 'Mixed_5c',pretrained=False)
        self.dropout  = nn.Dropout(0.35)
        self.logits =  Unit3D(  in_channels=2048,
                                output_channels=num_classes,
                                kernel_shape=[1, 1, 1],
                                padding=0,
                                activation_fn=None,
                                use_batch_norm=False,
                                use_bias=True,
                                name='logits' )
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
    def forward(self,inputs):
        
        rgb,depth = inputs
        rgb_feature = self.rgb_branch(rgb)
        depth_feature = self.depth_branch(depth)
      
        features = torch.cat([rgb_feature,depth_feature],dim = 1)
       
        features = self.logits(self.dropout(features))
        features = self.avg_pool(features)
        logits = features.squeeze(3).squeeze(3)
        logits = torch.mean(logits,dim=-1).squeeze()
        return logits

if __name__ == "__main__":
    model = MF_fusion(300)
    rgb = torch.randn(2,3,64,224,224)
    depth = torch.rand(2,1,64,224,224)
    inputs = (rgb,depth)
    
        
        