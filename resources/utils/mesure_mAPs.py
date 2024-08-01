

import pandas as pd
import numpy as np
import os



def caculate_maps_all(file_result,file_output = None):
    with open(file_result,'r') as f:
        f.readline()
        data = [list(map(float,line.strip().split(',')[:-1]))[1:][::3] for line in f.readlines() if line.strip() != '']
    # for line in data
    value_map = np.mean(np.array(data),axis=1,keepdims=True)
    file_output = file_output if file_output is not None else os.path.join(os.path.dirname(file_result),'mAPs.txt')
    
    with open(file_output,'w') as f: 
        f.write("index,mAPs\n")
        for i,value in enumerate(value_map):
         text = f"{i},{value}\n"
         f.write(text)
if __name__ == "__main__":
	data_path = "/work/21013187/SignLanguageRGBD/all_code/results/late_fusion-72/30-07-54-09/val_result.csv"
	caculate_maps_all(data_path)
	