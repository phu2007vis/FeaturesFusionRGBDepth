import torch
from torchvision.models.video import swin3d_s,Swin3D_S_Weights,Swin3D_B_Weights,Swin3D_T_Weights,swin3d_b,swin3d_t
import torch.nn as nn



register_model = {'s':swin3d_s,
               't':swin3d_t,
               'b':swin3d_b}
weight = {
    's':Swin3D_S_Weights.KINETICS400_V1,
    't':Swin3D_T_Weights.KINETICS400_V1,
    'b':Swin3D_B_Weights.KINETICS400_V1
}
class SwinTransformer(nn.Module):
    def __init__(self,
                 model_name = 's',
                 num_classes = 400
                 ):
        super(SwinTransformer,self).__init__()
        
        weight_tag = weight[model_name]
        self.model = register_model[model_name](
            weights = weight_tag
        )
        self.model.head = nn.Linear(self.model.head.in_features,num_classes)
    def forward(self,x):
        # B C T H W
        return self.model(x)
    
if __name__ == "__main__":
    model = SwinTransformer('t')
    x = torch.randn(2,3,70,224,224)
    import pdb;pdb.set_trace()
        


