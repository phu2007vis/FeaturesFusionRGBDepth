import random
import cv2
import numpy as np
import os
import argparse

_VIDEO_EXT = ['.avi', '.mp4', '.mov']
_IMAGE_EXT = ['.jpg', '.png']
_IMAGE_SIZE = 224


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n))]


def resize_img(img, short_size=256):
    h, w, c = img.shape
    return cv2.resize(img,(short_size,short_size))
    # if (w <= h and w == short_size) or (h <= w and h == short_size):
    #     return img
    # if w < h:
    #     ow = short_size
    #     oh = int(short_size * h / w)
    #     return cv2.resize(img, (ow, oh))
    # else:
    #     oh = short_size
    #     ow = int(short_size * w / h)
    #     return cv2.resize(img, (ow, oh))


def video_loader(video_path, short_size):
    video = []
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    video.append(resize_img(image, short_size))
    while success:
        success, image = vidcap.read()
        if not success: break
        video.append(resize_img(image, short_size))
    vidcap.release()
    return video


def images_loader(images_path, transform=None):
    images_set = []
    images_list = [i for i in images_path.iterdir() if not i.stem.startswith('.') and i.suffix.lower() in _IMAGE_EXT]
    images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f.stem))))
    for image_path in images_list:
        image = cv2.imread(str(image_path), 3)
        if transform is not None:
            image = transform(image)
        images_set.append(image)
    return images_set, len(images_set)


def sample_by_number(frame_num, out_frame_num, random_choice=False):
    full_frame_lists = split(list(range(frame_num)), out_frame_num)
    if random_choice:
            return [random.choice(i) for i in full_frame_lists]
    else:
        return [i[0] for i in full_frame_lists]



class FrameGenerator(object):
    def __init__(self, input_path,
                 resize=None
                 ):
        """
        :param input_path: The input video file or image set path
        :param sample_num: The number of frames you hope to use, they are chosen evenly spaced
        :param slice_num: The number of blocks you want to divide the input file into, and frames
                            are randomly chosen from each block.
        """
       
        self.frames = video_loader(input_path, resize)
        self.counter = 0
        
    def __len__(self):
        return len(self.frames)

    def reset(self):
        self.counter = 0

    def get_frame(self):
        frame = self.frames[self.counter]  # cv2.resize(frame, (_IMAGE_SIZE, _IMAGE_SIZE))
        self.counter += 1
        return frame


def get_video_generator(video_path, opts):
    video_object = FrameGenerator(video_path,resize=opts.resize)
    return video_object

def compute_rgb(video_object, out_path):
    
    rgb = np.array(video_object.frames)[:,:]
    np.save(out_path, rgb)
    return rgb

def pre_process(param):
    video_path,outpath, opts = param 
    video_object = get_video_generator(video_path, opts)
    compute_rgb(video_object,outpath)
    video_object.reset()


def mass_process(opts):
    param_list = []
    for sub_folder in os.listdir(opts.data_root):
        sub_folder_path = os.path.join(opts.data_root,sub_folder)
        rgb_folder  = os.path.join(sub_folder_path,"rgb")
        out_folder = os.path.join(sub_folder_path,"rgb_raw")
        os.makedirs(out_folder,exist_ok=True)
        for rgb_file_name in os.listdir(rgb_folder):
            input_path = os.path.join(rgb_folder,rgb_file_name)
            output_name = rgb_file_name.replace(".avi",".npy")
            output_path = os.path.join(out_folder,output_name)
            param = [input_path,output_path,opts]
            param_list.append(param)
    pre_process(param_list[0])
    import multiprocessing as mp
    from tqdm import tqdm
    with mp.Pool() as pool:
        for result in tqdm(pool.imap_unordered(pre_process,param_list),total=len(param_list)):
            pass
            
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-process the video into formats which i3d uses.')
    parser.add_argument(
        '--data_root',
        type=str,
        default=r"/work/21013187/SignLanguageRGBD/data/ver2_all_rgb_only",
        help='Where you want to save the output input_folder')
    # Sample arguments
    parser.add_argument(
        '--sample_num',
        type=int,
        default='32',
        help='The number of the output frames after the sample, or 1/sample_rate frames will be chosen.')
    parser.add_argument(
        '--resize',
        type=int,
        default='224',
        help="Resize the short edge video to '--resize'. Mention that this is only the pre-process, random crop"
             "will be applied later when training or testing, so here 'resize' can be a little bigger.")
    
    args = parser.parse_args()
    mass_process(args)

