import torch.nn as nn
from resources.i3d.pytorch_i3d import InceptionI3d
import torch
import os
class LF_fusion(nn.Module):
    def __init__(self,num_classes,rgb_pretrained = r"/work/21013187/SignLanguageRGBD/all_code/results/i3d-72/13-00-00-02/i3d-72_best.pt", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.rgb_branch = InceptionI3d(in_channels=3,num_classes=num_classes)
        self.depth_branch = InceptionI3d(in_channels=1 ,pretrained=True,num_classes=num_classes)
        self.dropout  = nn.Dropout(0.35)
        if os.path.exists(rgb_pretrained):
            try:
                #this will have bug , i will fix it in future bug missmatch number of classes
                state_dict = torch.load(rgb_pretrained,map_location= 'cpu')
                copy_state_dict = {k:v for k,v in state_dict.items()}
                for key in copy_state_dict.keys():
                    new_key = key[7:]
                    value = state_dict[key]
                    state_dict.pop(key)
                    state_dict[new_key] = value
                    
                self.rgb_branch.load_state_dict(state_dict)
            except:
                self.rgb_branch  = torch.load(rgb_pretrained,torch.device('cpu'))
        self.fintuning_all()
        self.fituning_depth()
    def fituning_depth(self):
        print("Turn off rgb branch")
        for param in self.rgb_branch.parameters():
            param.requires_grad = False
    def fintuning_all(self):
        print("Turn on rgb branch")
        for param in self.rgb_branch.parameters():
            param.requires_grad = True
    def forward(self,inputs):
        
        rgb,depth = inputs
        rgb_feature_logits = self.rgb_branch(rgb).unsqueeze(-1)
        depth_feature_logits = self.depth_branch(depth).unsqueeze(-1)
        logits = torch.cat([rgb_feature_logits,depth_feature_logits],dim  = -1)
        return torch.mean(logits,-1).squeeze()
    
    
if __name__ == "__main__":
    model = LF_fusion(119)
    rgb = torch.randn(2,3,64,224,224)
    depth = torch.rand(2,1,64,224,224)
    inputs = (rgb,depth)
    print(model(inputs).shape)
    
        
        