import tqdm
import random
import os
import cv2
import numpy as np

import torch
import math


DEFAULT_MAP = """A1 => nha_lau
A2 => nha_may_ngoi
A3 => nha_rong
A4 => nha_san
A5 => nha_biet_thu
A6 => nha_tren_cay
A7 => nha_go
A8 => nha_chung_cu
A9 => nha_tret
A10 => nha_ky_tuc_xa
A11 => tivi
A12 => den
A13 => dong_ho
A14 => cau_thang
A15 => chia_khoa
A16 => o_khoa
A17 => ban_ghe_sofa
A18 => ban_tho
A19 => dien_thoai_ban
A20 => tranh_anh_treo_tuong
A21 => ke_sach
"""




def filter_outliers(data):
    # Calculate the quartiles
    q1 = np.percentile(data, 20, axis=(1, 2))
    q3 = np.percentile(data, 70, axis=(1, 2))
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    # Calculate the lower and upper bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out the outliers, replace with the median
    filtered_data = np.where((data >= lower_bound[:, None, None]) & (data <= upper_bound[:, None, None]), data, np.median(data, axis=(1, 2), keepdims=True))

    return filtered_data






class DSL:
    def __init__(self,dataset_path,height=224,width=224,n_frames=320,batch_size=1,map_file=None,random_seed=42) -> None:
        '''
        Depth Sign Language Dataset Loader (Preprocessing and Data Augmentation)
        '''
        self.dataset_path = dataset_path
        self.height = height
        self.width = width
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.random_seed = random_seed

        if map_file is not None:
            # Load the map file
            with open(map_file, 'r') as f:
                self.map_txt = f.read()
        else:
            self.map_txt = DEFAULT_MAP
        self.map = {x.split(' => ')[0]: x.split(' => ')[1].replace('\n', '') for x in self.map_txt.split('\n') if '=>' in x}
        self.classes = []
        self.persons = []

        self.dataset = {}
        self.get_dataset_info()

    def get_dataset_info(self):
        # Get the dataset path
        data_batch = os.listdir(self.dataset_path)
        print(f"Gathering dataset information from {len(data_batch)} batches...\n")
        for i, batch in tqdm.tqdm(enumerate(data_batch)):
            # Get the batch path
            batch_path = os.path.join(self.dataset_path, batch)
            
            class_ = batch.split('P')[0]
            person_ = batch.split('P')[1]

            mapped_class = self.map[class_]
            if mapped_class not in self.classes:
                self.classes.append(mapped_class)
            
            if person_ not in self.persons:
                self.persons.append(person_)
            if mapped_class not in self.dataset:
                self.dataset[mapped_class] = {}
            if person_ not in self.dataset[mapped_class]:
                self.dataset[mapped_class][person_] = {'depth':[],'rgb':[]}

            for data_point in os.listdir(os.path.join(batch_path, 'depth')):
                data_point_path = os.path.join(batch_path,'depth', data_point)
                avi_file = os.path.join(batch_path,'rgb', data_point.replace('npy', 'avi'))
                #if both files exist, add to the dataset
                if os.path.exists(data_point_path) and os.path.exists(avi_file):
                    self.dataset[mapped_class][person_]['depth'].append(data_point_path)
                    self.dataset[mapped_class][person_]['rgb'].append(avi_file)
        print(f"Loaded {len(self.classes)} classes {self.classes}")
        print(f"Loaded {len(self.persons)} persons {self.persons}")
    def filter(self,classes:list=None,persons:list=None,randomize=True):
        filtered_list = {'depth':[],'rgb':[],'class':[],'len':0}
        #if no classes are specified, use all classes
        if classes is None:
            classes = self.classes
        #if no persons are specified, use all persons
        if persons is None:
            persons = self.persons
        for class_ in classes:
            for person in persons:
                if person not in self.dataset[class_]:
                    continue
                filtered_list['depth'].extend(self.dataset[class_][person]['depth'])
                filtered_list['rgb'].extend(self.dataset[class_][person]['rgb'])
                filtered_list['class'].extend([self.get_class_index(class_)]*len(self.dataset[class_][person]['depth']))
                filtered_list['len'] += len(self.dataset[class_][person]['depth'])
        if randomize:
            random.seed(self.random_seed)
            #shuffle the dataset
            zipped = list(zip(filtered_list['depth'],filtered_list['rgb'],filtered_list['class']))
            random.shuffle(zipped)
            filtered_list['depth'],filtered_list['rgb'],filtered_list['class'] = zip(*zipped)
        return filtered_list

    def get_class_index(self,class_):
        return self.classes.index(class_)
    def get_class_name(self,index):
        return self.classes[index]
    def get_classes(self):
        return self.classes
    def get_persons(self):
        return self.persons
    def get_generator(self,filtered_list=None,randomize=True,output='rgb'):
        if filtered_list is None:
            filtered_list = self.filter(randomize=randomize)
        return Generator(filtered_list,self.height,self.width,self.n_frames,self.batch_size,self.classes,output=output)
        
optical_flow = cv2.optflow.createOptFlow_DualTVL1()
class Generator(torch.utils.data.IterableDataset ):
    def __init__(self,data_paths,height,width,n_frames,batch_size,classes,output) -> None:
        super(Generator).__init__()
        self.data_paths = data_paths
        self.height = height
        self.width = width
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.output = output
        self.start = 0
        self.classes = classes
        self.end = len(data_paths['class'])

    def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
         return iter(self.get_data(iter_start, iter_end))
    

    def get_data(self,start,end):
        #create a list of labels
        labels = self.data_paths['class']
        depth_batch = self.data_paths['depth']
        rgb_batch = self.data_paths['rgb']

        #i = 0
        for i in range(start,end):
            #if i is greater than the length of the dataset, reset i
            #if i >= len(depth_batch):
            #    i = 0
            #print(f"\rReading depth data of {i}..",end='')

            if self.output == 'flow' or self.output == 'rgbd':
                read_depth = np.load(depth_batch[i]) #(time, height, width)
                read_depth = filter_outliers(read_depth)

                read_depth = read_depth[:-1] #remove the last frame
                read_depth = np.expand_dims(read_depth, axis=-1) #(time, height, width, 1)
                #normalize depth data to 0-255
                read_depth = (read_depth - np.min(read_depth)) / (np.max(read_depth) - np.min(read_depth)) * 255.0
                n_frames = read_depth.shape[0]

            if 'rgb' in self.output:
                read_rgb = cv2.VideoCapture(rgb_batch[i])
                #read avi video and store as np array
                read = []
                #print(f"Reading rgb data of {i}:{rgb_batch[i]}..",end='\n')
                while True:
                    ret, frame = read_rgb.read()
                    if not ret:
                        break
                    read.append(frame)
                combined = np.array(read) #(time, height, width, channels)
            #if dimensions 0 of read,read_depth are not equal, trim the larger one
            if self.output == 'rgbd':
                if read.shape[0] > read_depth.shape[0]:
                    read = read[:read_depth.shape[0]]
                    #print(f"Trimming rgb data of {rgb_batch[i]} to {read_depth.shape[0]}..",end='\n')
                elif read.shape[0] < read_depth.shape[0]:
                    read_depth = read_depth[:read.shape[0]]
                    #print(f"Trimming depth data of {depth_batch[i]} to {read.shape[0]}..",end='\n')
                #combine depth and rgb as channels 4
                combined = np.concatenate([read,read_depth],axis=-1) #RGBD
            if self.output == 'flow':
                read_depth = read_depth.astype(np.uint8)
                prev = read_depth[0]
                combined = np.zeros((n_frames, read_depth.shape[1], read_depth.shape[2], 2), dtype=np.float32)
                for _i in range(1, n_frames):
                    combined[_i] = optical_flow.calc(prev,read_depth[_i], None)
                    prev = read_depth[i]

            #to uint8
            combined = combined.astype(np.uint8)


            #resize X to HEIGHT, WIDTH
            X = np.array([cv2.resize(combined[j], (self.width, self.height))/255.0 for j in range(combined.shape[0])])
            #repeat X so X > n_frames
            if X.shape[0] < self.n_frames:
                X = np.tile(X, (self.n_frames//X.shape[0] + 1, 1, 1, 1))
            
            if X.shape[0] > self.n_frames:
                X = X[:self.n_frames]
            #Y is index of class
            y = labels[i]
            #normalize X
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
            X = X * 2.0 - 1.0

            print(f"Getting data {i}:{rgb_batch[i]} of class {y}.",end='\n')
            #reshape X to (C x T x H x W)
            X = np.transpose(X, (3, 0, 1, 2))
            #FloatTensor
            X = torch.FloatTensor(X)
            y = torch.nn.functional.one_hot(torch.tensor(y), len(self.classes))
            y = y.unsqueeze(0).expand(self.n_frames, -1).T.float() 
            #print(f"Y: {y.shape}")
            yield X, y
