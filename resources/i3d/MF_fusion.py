import torch.nn as nn
from resources.i3d.pytorch_i3d import InceptionI3d,Unit3D
import torch

class MF_fusion(nn.Module):
    def __init__(self,num_classes, rgb_pretrained = r"/work/21013187/SignLanguageRGBD/all_code/results/i3d-72/13-00-00-02/i3d-72_best.pt",*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.rgb_branch = InceptionI3d(in_channels=3,final_endpoint = 'Mixed_5c',num_classes=119)
        try:
                
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
        self.depth_branch = InceptionI3d(in_channels=1 ,final_endpoint = 'Mixed_5c',pretrained=True)
        self.dropout  = nn.Dropout(0.35)
        self.logits =  Unit3D( in_channels=2048,
                                output_channels=num_classes,
                                kernel_shape=[1, 1, 1],
                                padding=0,
                                activation_fn=None,
                                use_batch_norm=False,
                                use_bias=True,
                                name='logits' )
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
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
        rgb_feature = self.rgb_branch(rgb)
        depth_feature = self.depth_branch(depth)
      
        features = torch.cat([rgb_feature,depth_feature],dim = 1)
       
        features = self.logits(self.dropout(features))
        features = self.avg_pool(features)
        logits = features.squeeze(3).squeeze(3)
        logits = torch.mean(logits,dim=-1).squeeze()
        return logits
    def load_weight(self,model_path):
        state_dict  = torch.load(model_path,map_location='cpu')
        
        import pdb;pdb.set_trace()
        self.load_state_dict(state_dict)

if __name__ == "__main__":
    model = MF_fusion(119)
    model_path = "/work/21013187/SignLanguageRGBD/all_code/results/middle_fusion-72/27-15-22-47/middle_fusion-72_best.pt"
    model  = torch.load(model_path,map_location='cpu')
 
    rgb = torch.randn(2,3,64,224,224)
    depth = torch.rand(2,1,64,224,224)
    inputs = (rgb,depth)
    print(model(inputs).shape)
        
        