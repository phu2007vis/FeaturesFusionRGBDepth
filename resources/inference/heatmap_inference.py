
from resources.i3d.pytorch_i3d import *
import torch
import argparse
from resources.utils.data_prepareation import video_loader
import numpy as np
from resources.utils.augumentaion import SpatialTransform
import torch.nn.functional as F
from resources.utils.augumentaion import TemperalAugument



def load_rgb_input(image_path,spatial_transformer,temperal_augument):
    sample = np.array(video_loader(image_path))
    X_spatial_augumented = spatial_transformer.transform_fn(sample)
    X_temperal_augumented = temperal_augument.get_augmented_data(X_spatial_augumented).permute(1,0,2,3).unsqueeze(0)
    return X_temperal_augumented

def load_model(model_name,
               pretrained_path,
               device,
               num_classes = 119
               ):
	if model_name == "i3d":
		state_dict = torch.load(pretrained_path,
                          		map_location= device)
		model = InceptionI3d(num_classes=num_classes)
		model.load_state_dict(state_dict=state_dict)
		model.to(device)
	return model







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model name s3d or i3d
    parser.add_argument("--model_name",type=str,default="i3d",help='i3d or s3d or lstm')
    parser.add_argument("--pretrained",type=str,default='/work/21013187/SignLanguageRGBD/all_code/results/heatmap_inference/21-16-01-50/heatmap_inference_best.pt')
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--img_size",type=int,default="224")
    parser.add_argument("--input_source",type=str,default="/work/21013187/SignLanguageRGBD/ViSLver2/Processed/A5P14/heatmap/101_A5P7_.avi")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    pretrained_path = args.pretrained
    device = args.device
    img_size = args.img_size
    input_source = args.input_source
    
    model = load_model(model_name,
                       pretrained_path,
                       device).eval()
    
    spatial_transform = SpatialTransform((img_size,img_size))
    temperal_transform = TemperalAugument(72,'valid')
    tensor_input =  load_rgb_input(input_source,spatial_transform,temperal_transform).to(device)
    output = model(tensor_input)
    prob,index = F.softmax(output,dim = 0).max(0)
    import pdb;pdb.set_trace()

    
    














