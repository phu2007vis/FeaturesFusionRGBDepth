import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision
from torchvision import transforms
import DSL.lstm_dataset as dsl

import numpy as np
import time
import datetime
from pytorch_i3d import InceptionI3d
from lstm import LSTM1



os.makedirs('logs',exist_ok=True)
import logging.handlers
# Set up logging
log_filename = datetime.datetime.now().strftime("sc_%d_%m_%H_%M_%S.log")
log_filepath = os.path.join("logs", log_filename)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(log_filepath, maxBytes=(1048576*5), backupCount=7),
        logging.StreamHandler(sys.stdout)
    ],force=True
)

def handle_exception(exc_type, exc_value, exc_traceback):
    # Custom exception handling logic here
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

def run(init_lr=0.0001, max_steps=2000000, mode='rgb', root="/work/21010294/DepthData/OutputSplitAbsoluteVer2/", batch_size=1,
        name='lstm', nframe=200, elog=None, cache=None):
    HEIGHT = 224
    WIDTH = 224
    dataset = dsl.DSL(root, height=HEIGHT, width=WIDTH, n_frames=nframe, cache_folder=cache)
    person_list = dataset.get_persons()
    val_index = int(len(person_list) * 0.7)
    test_index = int(len(person_list) * 0.8)
    train_persons = person_list[:val_index]
    val_persons =   person_list[val_index:test_index]
    test_persons =  person_list[test_index:]
    save_model = elog.get_path() + f"/{name}_"
    print(f"Train: {len(train_persons)}",
        f" Val: {len(val_persons)}",
        f" Test: {len(test_persons)}")

    train_filter = dataset.filter(persons=train_persons)
    val_filter = dataset.filter(persons=val_persons)
    test_filter = dataset.filter(persons=test_persons)

    print(f"Train set size: {len(train_filter['class'])}")
    print(f"Validation set size: {len(val_filter['class'])}")
    print(f"Test set size: {len(test_filter['class'])}")
    


    train_ds = dataset.get_generator(train_filter,output=mode)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    val_ds = dataset.get_generator(val_filter,output=mode)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    test_ds = dataset.get_generator(test_filter,output=mode)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    dataloaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}

    num_classes = len(dataset.get_classes())
    model  = LSTM1().cuda()
    saved_model_path = "/work/21013187/SignLanguageRGBD/i3d/30-22-15-39_i3d-rgb36/i3d-rgb36_031970.pt"
    model_state_dict = torch.load(saved_model_path)
    model.load_state_dict(model_state_dict)
    if True: 
        steps = -1
        for phase in ['val']:           
                model.eval()
                with torch.no_grad():
                    logits = []
                    for data in dataloaders[phase]:
                        inputs, labels = data
                        t_start = time.time()
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        _,logit = (nn.functional.softmax(model(inputs),dim = 1)).max(1)
                        labels = np.argmax(labels.cpu().numpy(),axis = 1)[0]
                        infertime = time.time() - t_start
                        logits.append([labels,logit,logit.detach().cpu().reshape(-1).tolist(),labels])
                elog.evaluate('test',steps,logits,dataset.get_classes())




if __name__ == '__main__':
    import eval as ev
    # need to add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='rgb or flow', default='rgb')
    parser.add_argument('-r', '--root', type=str, help='root directory of the dataset', default="/work/21013187/SignLanguageRGBD/OutputSplitAbsoluteVer2/")                                        
    parser.add_argument('-n','--nframe',type=int,help= 'n frame',default = 36)
    parser.add_argument('-c', '--cache', type=str, help='cache directory', default=None)
    

    args = parser.parse_args()
    mode = args.mode
    root = args.root
    nframe = args.nframe


    name = f"i3d-{mode}{nframe}"
    elog = ev.Eval(run_name=name)
    run(mode=args.mode, root=root, name=name, nframe=nframe, elog=elog,cache=args.cache)
