
from resources.i3d.pytorch_i3d import *
import torch
import argparse
from resources.utils.data_prepareation import video_loader
import numpy as np
from resources.utils.augumentaion import SpatialTransform
import torch.nn.functional as F
from resources.utils.augumentaion import TemperalAugument
from resources.utils.heatmap_preparation import extract_heatmap_from_array
from resources.utils.visualize import save_frames_as_video

def preprocessing_rgb_input(sample,spatial_transformer,temperal_augument):
    X_spatial_augumented = spatial_transformer.transform_fn(sample)
    X_temperal_augumented = temperal_augument.get_augmented_data(X_spatial_augumented).permute(1,0,2,3).unsqueeze(0)
    return X_temperal_augumented
    
def load_rgb_input(image_path,spatial_transformer,temperal_augument,input_type = 'heatmap'):
    assert(input_type in ('heatmap','rgb'))
    sample = np.array(video_loader(image_path))
    if input_type == 'rgb':
        sample = extract_heatmap_from_array(sample)
    return preprocessing_rgb_input(sample,spatial_transformer,temperal_augument),sample

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
    parser.add_argument("--input_type",type=str,default="rgb")
    parser.add_argument("--pretrained",type=str,default='/work/21013187/SignLanguageRGBD/all_code/results/heatmap_inference/21-16-01-50/heatmap_inference_best.pt')
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--img_size",type=int,default="224")
    parser.add_argument("--output_dir",type= str,default = "/work/21013187/SignLanguageRGBD/results_inference")
    parser.add_argument("--input_source",type=str,default="/work/21013187/SignLanguageRGBD/all_code/5820290365444.mp4")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    pretrained_path = args.pretrained
    device = args.device
    img_size = args.img_size
    input_source = args.input_source
    input_type = args.input_type
    output_dir = args.output_dir
    
    
    model = load_model(model_name,
                       pretrained_path,
                       device).eval()
    
    spatial_transform = SpatialTransform((img_size,img_size))
    temperal_transform = TemperalAugument(72,'valid')
    tensor_input,heatmap_video =  load_rgb_input(input_source,spatial_transform,temperal_transform,input_type)
    output = model(tensor_input.to(device))
    prob,index = F.softmax(output,dim = 0).max(0)
    
    name = f"output_cls_{index.item()}_{round(prob.item(),2)}.mp4"
    os.makedirs(output_dir,exist_ok=True)
    path = os.path.join(output_dir,name)
    print(f"Save result at : {path}")
    save_frames_as_video(heatmap_video,path)
    
    import pdb;pdb.set_trace()

    
    














