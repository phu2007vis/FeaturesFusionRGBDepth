import cv2
import numpy as np


def resize_depth(depth,max_value,min_value,size = 224):
    depth = np.expand_dims(depth,-1)
    depth = (depth-min_value)/(max_value-min_value)
    depth_frame = np.uint8(np.clip(depth*255,a_min=0,a_max=255))
    depth_frame_resize = cv2.resize(depth_frame,(size,size))
    return depth_frame_resize/255
def resize_array_depth(depth_array,size = 224):
    max_value = depth_array.max()
    min_value = depth_array.min()
    result = []
    for depth in depth_array:
        result.append(resize_depth(depth,max_value=max_value,min_value=min_value,size = size))
    return np.expand_dims(np.array(result),1)

# x = np.load('/work/21013187/SignLanguageRGBD/ViSLver2/Processed/A1P2/depth/40_A1P25_.npy')
# print(resize_and_normalize_list(x).shape)