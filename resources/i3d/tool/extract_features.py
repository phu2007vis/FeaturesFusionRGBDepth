
import torch
from torch.autograd import Variable
from pytorch_i3d import InceptionI3d

def run(load_model=r"D:\pytorch-i3d\models\i3d_weight_pretrained.pt"):
   
    i3d = InceptionI3d(in_channels=3)
    i3d.load_state_dict(torch.load(load_model))
    i3d.eval()
    # i3d.cuda() uncoment this for gpu
    with torch.no_grad() :
        inputs = torch.randn(2,3,10,224,224)
        inputs = inputs.to("cpu")
        features = i3d.extract_features(inputs)
        import pdb;pdb.set_trace()

if __name__ == '__main__':
    run()
