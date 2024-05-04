import numpy as np
import random

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
class SpatialAugument:
    def __init__(self) -> None:
        pass