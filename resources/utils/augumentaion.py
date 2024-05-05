import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
import torch
import cv2

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n))]

def sample_by_number(frame_num, out_frame_num, random_choice=True):
    full_frame_lists = split(list(range(frame_num)), out_frame_num)
    if random_choice:
            return [random.choice(i) for i in full_frame_lists]
    else:
        return [i[0] for i in full_frame_lists]
def time_augument(data,out_frames,random=True):
    if out_frames > data.shape[0]:
        data = np.repeat(data, out_frames // data.shape[0] + 1, axis=0)
    index_choose = sample_by_number(len(data),out_frames,random_choice=random)
    return data[index_choose]
class TemperalAugument:
    def  __init__(self,out_frames,mode):
        self.out_frames  = out_frames
        assert(mode in ['valid','train'])
        self.mode = mode
    def augment_data(self,data,frame_skip):
        duration = data.shape[0]
        crop_duration = duration // 7
        start =  np.random.choice(np.arange(crop_duration,crop_duration*3)) 
        end = np.random.choice(np.arange(duration-crop_duration*3,duration-crop_duration*1))
        data = data[start:end]

        if frame_skip < 0:
            data = np.repeat(data, abs(frame_skip), axis=0)
        elif frame_skip > 0:
            data = data[::frame_skip]
        return time_augument(data,self.out_frames)

    def get_augmented_data(self, data):
        if self.mode == "valid":
            duration = data.shape[0]
            crop_duration = duration // 7
            cropped_data = data[int(crop_duration*1.5):-int(crop_duration*1.5)]
            return time_augument(cropped_data,self.out_frames,random=False)
        speed = np.random.choice([-2,-1, 0, 1, 2])
        return self.augment_data(data, speed)
class SpatialTransform:
    def __init__(self, output_size=(224, 224),augument = None):
        self.output_size = output_size
        self.augument = augument
        self.transform_list = []
        if self.augument is not None:
            if self.augument.get('color') is not None:
                self.transform_list.append(transforms.ColorJitter(*augument['color']))
        self.transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(
            self.transform_list
        )
    def reset(self):
        if self.augument:
            if self.augument.get("h_flip") is  not None:
                self.h_flip = random.random() < self.augument.get("h_flip")
            if self.augument.get("rotation") is not None:   
                self.rotate_angle = random.uniform(-self.augument['rotation'],self.augument['rotation']) 
                self.rotation_matrix = cv2.getRotationMatrix2D((self.output_size[0] / 2, self.output_size[1] / 2), self.rotate_angle, 1)  
    def image_augument(self,image:np.array):
        if self.augument and self.augument.get('gausian_noise') is not None:
            noise = np.random.normal(self.augument['gausian_noise']['mean'] , self.augument['gausian_noise']['std'], image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        if self.augument and self.h_flip:
            image = cv2.flip(image,1)
        if self.augument and self.rotate_angle is not None:
            height, width = image.shape[:2]
            image = cv2.warpAffine(image, self.rotation_matrix, (width, height))
        image = Image.fromarray(image)
        return image
    def transform_fn(self, image_nps):
        self.reset()
        image_PILs = []
        for image_np in image_nps:
                image = cv2.resize(image_np,self.output_size)
                image_PIL = self.image_augument(image)
                image_PIL = self.transform(image_PIL)
                image_PILs.append(image_PIL)
        return torch.stack(image_PILs)
